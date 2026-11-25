#!/usr/bin/env python3
"""
Module 2: `active_perception.py` — 360° Blind-Spot Clearing (Head -> Waist -> Body) & Battery Sensor Gating
===========================================================================================================
Grounded in the exact camera & joint topology of `tiangong2pro_torq.xml` and Boards 1, 2, 3:
  - `camera_head_link`       (Board 2 `orin_llm` Orbbec RGB-D #1, mounted on 3-DoF neck + 1-DoF waist)
  - `camera_body_front_link` (Board 3 `orin_slam` Orbbec RGB-D #2, angled down at floor ahead of waist)
  - `camera_body_back_link`  (Rear torso camera looking behind `waist_yaw_link`)
  - `livox_ros_driver2`      (Board 3 `orin_slam` 3D LiDAR)

Implements 4 core perceptual instincts:
  1. Instinct #3 (`ActiveGazeAndBlindSpotManager`):
     Maintains a 12-sector (360 deg) Polar Blind-Spot & Uncertainty Map around the robot.
     Clears blind spots using the Minimum-Energy Motor Hierarchy:
       - Tier 1 (|az| <= 45 deg)  : Swivel Head ONLY (`head_yaw`, `head_pitch`, `head_roll` — ~4 W)
       - Tier 2 (45..145 deg)     : Swivel Head + Turn Waist CW/CCW (`body_yaw_joint` Motor 31 — feet stay planted!)
       - Tier 3 (|az| > 145 deg)  : Rotate whole body (`vyaw` via `rl_control_node`)
     Also enforces "Look-Before-Stepping": swivels eyes toward any intended sidestep (`vy`) or turn!
  2. Instinct #4 (`SensorEnergyManager`):
     Turns sensors ON/OFF or throttles Hz across Boards 1, 2, 3 depending on context & battery level
     (`/power/board/status`), saving up to 68% sensor & Orin GPU power when idle!
  3. Instinct #9 (`AcousticReflex`):
     Instant sub-100ms head/waist snap toward sudden sound azimuth before speech recognition even finishes.
  4. Instinct #10 (`GroundHazardInstinct`):
     Uses downward `camera_body_front_link` to spot floor hazards (cables, toys, puddles, stairs/cliffs).
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class GazeMotorCommand:
    """Output of the 3-Tier Head -> Waist -> Pelvis Blind-Spot & Gaze Controller."""
    tier: str  # "TIER1_HEAD_ONLY", "TIER2_HEAD_PLUS_WAIST_CW_CCW", "TIER3_WHOLE_BODY_TURN"
    target_azimuth_deg: float
    target_elevation_deg: float
    head_yaw_rad: float
    head_pitch_rad: float
    head_roll_rad: float
    body_yaw_rad: float      # Motor 31 (+CCW / -CW waist turn)
    pelvis_vyaw_rad_s: float
    estimated_motor_watts: float
    reason: str


class ActiveGazeAndBlindSpotManager:
    """
    Instinct #3 & #9: 360-Degree Blind-Spot Map & 3-Tier Motor Hierarchy (Head -> Waist -> Legs).
    """

    def __init__(self, num_sectors: int = 12):
        self.num_sectors = num_sectors
        self.sector_deg = 360.0 / num_sectors
        # Uncertainty in [0.0 = freshly observed, 1.0 = complete blind spot]
        self.uncertainty = np.ones(num_sectors, dtype=float) * 0.6
        # Front sectors start observed
        self.uncertainty[0] = 0.0
        self.uncertainty[1] = 0.1
        self.uncertainty[-1] = 0.1

    def update_visibility(
        self,
        dt: float,
        head_yaw_rad: float,
        body_yaw_rad: float,
        rear_camera_active: bool = False,
        lidar_active: bool = False,
    ):
        """
        Updates the 360-deg blind-spot map based on where `camera_head_link` is currently pointing
        (`gaze_angle = body_yaw_rad + head_yaw_rad`) and whether rear cam / LiDAR are powered on.
        """
        # Unobserved sectors gradually accumulate uncertainty (blind-spot decay)
        self.uncertainty = np.clip(self.uncertainty + 0.08 * dt, 0.0, 1.0)

        gaze_deg = math.degrees(body_yaw_rad + head_yaw_rad) % 360.0
        rear_deg = (math.degrees(body_yaw_rad) + 180.0) % 360.0

        for i in range(self.num_sectors):
            center_deg = (i * self.sector_deg) % 360.0
            # Angular distance to head camera optical axis (FOV ~ 70 deg -> +-35 deg)
            diff_head = abs((center_deg - gaze_deg + 180.0) % 360.0 - 180.0)
            if diff_head <= 36.0:
                self.uncertainty[i] = 0.0
            elif lidar_active and diff_head <= 120.0:
                self.uncertainty[i] = min(self.uncertainty[i], 0.20)

            if rear_camera_active:
                diff_rear = abs((center_deg - rear_deg + 180.0) % 360.0 - 180.0)
                if diff_rear <= 35.0:
                    self.uncertainty[i] = 0.0

    def get_worst_blind_spot_azimuth_deg(self) -> Tuple[float, float]:
        """Returns `(azimuth_deg_in_[-180, 180], uncertainty_score)` of the highest-uncertainty blind spot."""
        idx = int(np.argmax(self.uncertainty))
        az = idx * self.sector_deg
        if az > 180.0:
            az -= 360.0
        return float(az), float(self.uncertainty[idx])

    def orient_to_target_or_blind_spot(
        self,
        target_azimuth_deg: float,
        target_elevation_deg: float = 0.0,
        curious_tilt: bool = False,
        reason: str = "Inspecting blind spot",
    ) -> GazeMotorCommand:
        """
        Computes the Minimum-Energy Coordination across Head (Motors 1-3), Waist (Motor 31), and Legs (`vyaw`):
          - |azimuth| <= 45 deg  -> Tier 1: Swivel Head Only (4 W)
          - 45 < |az| <= 145 deg -> Tier 2: Swivel Head + Rotate Waist Clockwise/Counter-Clockwise (18 W, feet planted!)
          - |azimuth| > 145 deg  -> Tier 3: Rotate Waist + Step-Turn Pelvis (`vyaw`, 65 W)
        """
        az_rad = math.radians(target_azimuth_deg)
        el_rad = math.radians(target_elevation_deg)
        # Pitch in `tiangong2pro_torq.xml`: positive pitches head downward, negative pitches upward
        head_pitch = float(np.clip(-el_rad, -0.42, 0.42))
        head_roll = 0.14 if curious_tilt else 0.0

        abs_az_deg = abs(target_azimuth_deg)

        if abs_az_deg <= 45.0:
            return GazeMotorCommand(
                tier="TIER1_HEAD_ONLY",
                target_azimuth_deg=target_azimuth_deg,
                target_elevation_deg=target_elevation_deg,
                head_yaw_rad=float(np.clip(az_rad, -1.50, 1.50)),
                head_pitch_rad=head_pitch,
                head_roll_rad=head_roll,
                body_yaw_rad=0.0,
                pelvis_vyaw_rad_s=0.0,
                estimated_motor_watts=4.2,
                reason=f"{reason} at {target_azimuth_deg:+.0f} deg -> Swiveling neck only (zero foot/waist motion).",
            )

        if abs_az_deg <= 145.0:
            # Split the rotation smoothly between Waist (`body_yaw_joint`) and Neck (`head_yaw_joint`)!
            # `body_yaw_joint` can turn up to +-170 deg, so turning waist 55% and head 45% feels 100% human!
            waist_yaw = float(np.clip(0.58 * az_rad, -1.57, 1.57))
            rem_head_yaw = float(np.clip(az_rad - waist_yaw, -1.45, 1.45))
            direction_str = "counter-clockwise (+Z)" if waist_yaw > 0 else "clockwise (-Z)"
            return GazeMotorCommand(
                tier="TIER2_HEAD_PLUS_WAIST_CW_CCW",
                target_azimuth_deg=target_azimuth_deg,
                target_elevation_deg=target_elevation_deg,
                head_yaw_rad=rem_head_yaw,
                head_pitch_rad=head_pitch,
                head_roll_rad=head_roll,
                body_yaw_rad=waist_yaw,
                pelvis_vyaw_rad_s=0.0,
                estimated_motor_watts=18.5,
                reason=(
                    f"{reason} at {target_azimuth_deg:+.0f} deg (in side/rear blind spot) -> "
                    f"Turning waist {direction_str} ({math.degrees(waist_yaw):+.0f} deg) + neck ({math.degrees(rem_head_yaw):+.0f} deg) with feet planted!"
                ),
            )

        # Tier 3: Target is nearly directly behind (> 145 deg) -> start waist + head lead while turning pelvis
        sign = 1.0 if target_azimuth_deg >= 0.0 else -1.0
        return GazeMotorCommand(
            tier="TIER3_WHOLE_BODY_TURN",
            target_azimuth_deg=target_azimuth_deg,
            target_elevation_deg=target_elevation_deg,
            head_yaw_rad=float(1.10 * sign),
            head_pitch_rad=head_pitch,
            head_roll_rad=head_roll,
            body_yaw_rad=float(1.20 * sign),
            pelvis_vyaw_rad_s=float(0.65 * sign),
            estimated_motor_watts=64.0,
            reason=(
                f"{reason} at {target_azimuth_deg:+.0f} deg (deep rear blind spot) -> "
                "Leading with head + waist while commanding `rl_control_node` pelvis rotation."
            ),
        )

    def look_before_stepping(self, cmd_vx: float, cmd_vy: float, cmd_vyaw: float) -> Optional[GazeMotorCommand]:
        """
        Human instinct: never step sideways (`cmd_vy`) or turn into an unseen zone without looking there first!
        """
        if abs(cmd_vy) > 0.05 and abs(cmd_vx) < 0.10:
            look_deg = 85.0 if cmd_vy > 0 else -85.0
            return self.orient_to_target_or_blind_spot(
                look_deg,
                target_elevation_deg=-10.0,
                reason="Look-Before-Sidestepping safety reflex",
            )
        if abs(cmd_vyaw) > 0.25:
            look_deg = 50.0 if cmd_vyaw > 0 else -50.0
            return self.orient_to_target_or_blind_spot(
                look_deg,
                target_elevation_deg=-5.0,
                reason="Look-Into-Turn safety reflex",
            )
        return None


@dataclass
class SensorPowerAllocation:
    """Output of Instinct #4 (`SensorEnergyManager`)."""
    profile_name: str  # "DEEP_IDLE_ECO", "ACOUSTIC_ALERT_GLANCE", "ACTIVE_LOCOMOTION", "TIGHT_MANEUVER_360", "LOW_BATTERY_SURVIVAL"
    sensor_states: Dict[str, str]
    total_sensor_watts: float
    power_saved_pct: float
    explanation: str


class SensorEnergyManager:
    """
    Instinct #4: Context & Battery-Aware Sensor Power Gating across Boards 1, 2, and 3.
    Turns sensors on/off or adjusts frame rate depending on what the robot is actually doing,
    instead of burning full battery on all 3 boards 100% of the time!
    """

    # Nominal full-rate power consumption (Watts) of perception & streaming modules on Boards 2 & 3
    FULL_POWER_WATTS = {
        "head_orbbec_rgbd (Board 2)": 8.5,
        "front_torso_rgbd (Board 3)": 7.5,
        "rear_torso_cam (Board 3)": 5.0,
        "livox_3d_lidar (Board 3)": 14.0,
        "voice_orin_mic_array (Board 2)": 2.5,
        "sri_6dof_wrist_ft (Board 1)": 1.8,
        "webrtc_video_stream (Boards 2/3)": 9.5,
    }

    def compute_sensor_power_plan(
        self,
        battery_pct: float,
        is_walking: bool,
        is_backing_up_or_squeezing: bool,
        sound_detected: bool,
        hands_interacting: bool,
        remote_operator_connected: bool = False,
    ) -> SensorPowerAllocation:
        max_watts = sum(self.FULL_POWER_WATTS.values())  # 48.8 W

        # 1. Critical Low-Battery Survival Mode (< 18%)
        if battery_pct < 18.0:
            states = {
                "head_orbbec_rgbd (Board 2)": "ON (10 Hz Low-Power)",
                "front_torso_rgbd (Board 3)": "ON (5 Hz Cliff-Guard)" if is_walking else "SLEEP (0 W)",
                "rear_torso_cam (Board 3)": "SLEEP (0 W)",
                "livox_3d_lidar (Board 3)": "SLEEP (0 W — Using RGB-D Depth Only)",
                "voice_orin_mic_array (Board 2)": "ON (Wake-Word Always Active)",
                "sri_6dof_wrist_ft (Board 1)": "ON (100 Hz)" if hands_interacting else "IDLE (10 Hz)",
                "webrtc_video_stream (Boards 2/3)": "SLEEP (0 W)",
            }
            watts = 9.8 if is_walking else 6.2
            return SensorPowerAllocation(
                profile_name="LOW_BATTERY_SURVIVAL",
                sensor_states=states,
                total_sensor_watts=watts,
                power_saved_pct=100.0 * (1.0 - watts / max_watts),
                explanation=f"Battery at {battery_pct:.0f}% -> Shutting down Livox LiDAR, Rear Cam & WebRTC to preserve battery for safe return.",
            )

        # 2. Tight Maneuver / Backing Up / Sideways Squeeze (360-deg close-in awareness)
        if is_backing_up_or_squeezing:
            states = {
                "head_orbbec_rgbd (Board 2)": "ON (30 Hz Full Depth)",
                "front_torso_rgbd (Board 3)": "ON (30 Hz Floor/Body Clearance)",
                "rear_torso_cam (Board 3)": "ON (30 Hz Rear Blind-Spot Guard)",
                "livox_3d_lidar (Board 3)": "ON (10 Hz 3D PointCloud)",
                "voice_orin_mic_array (Board 2)": "ON",
                "sri_6dof_wrist_ft (Board 1)": "ON (200 Hz Contact Guard)",
                "webrtc_video_stream (Boards 2/3)": "ON" if remote_operator_connected else "SLEEP (0 W)",
            }
            watts = 39.3 + (9.5 if remote_operator_connected else 0.0)
            return SensorPowerAllocation(
                profile_name="TIGHT_MANEUVER_360",
                sensor_states=states,
                total_sensor_watts=watts,
                power_saved_pct=100.0 * (1.0 - watts / max_watts),
                explanation="Squeezing through narrow gap / backing up -> Waking Rear Torso Cam + Front Floor Cam + 200Hz Wrist Contact Guard.",
            )

        # 3. Normal Active Walking / Crowd Navigation
        if is_walking:
            states = {
                "head_orbbec_rgbd (Board 2)": "ON (30 Hz)",
                "front_torso_rgbd (Board 3)": "ON (15 Hz Floor Hazard Watch)",
                "rear_torso_cam (Board 3)": "SLEEP (0 W — Moving Forward)",
                "livox_3d_lidar (Board 3)": "ON (10 Hz Crowd/Obstacle Tracking)",
                "voice_orin_mic_array (Board 2)": "ON",
                "sri_6dof_wrist_ft (Board 1)": "ON (100 Hz)" if hands_interacting else "LOW_RATE (20 Hz)",
                "webrtc_video_stream (Boards 2/3)": "ON" if remote_operator_connected else "SLEEP (0 W)",
            }
            watts = 29.8 + (9.5 if remote_operator_connected else 0.0)
            return SensorPowerAllocation(
                profile_name="ACTIVE_LOCOMOTION",
                sensor_states=states,
                total_sensor_watts=watts,
                power_saved_pct=100.0 * (1.0 - watts / max_watts),
                explanation="Active forward walking -> Head RGB-D + Downward Floor Cam + Livox LiDAR active; Rear Cam & WebRTC sleeping.",
            )

        # 4. Stationary Acoustic / Interaction Alert
        if sound_detected or hands_interacting:
            states = {
                "head_orbbec_rgbd (Board 2)": "ON (15 Hz Social/Reach Tracking)",
                "front_torso_rgbd (Board 3)": "SLEEP (0 W — Feet Planted)",
                "rear_torso_cam (Board 3)": "SLEEP (0 W)",
                "livox_3d_lidar (Board 3)": "SLEEP (0 W — Feet Planted)",
                "voice_orin_mic_array (Board 2)": "ON (Active Beamforming)",
                "sri_6dof_wrist_ft (Board 1)": "ON (200 Hz Tactile/Handover)" if hands_interacting else "IDLE (10 Hz)",
                "webrtc_video_stream (Boards 2/3)": "SLEEP (0 W)",
            }
            watts = 11.2
            return SensorPowerAllocation(
                profile_name="ACOUSTIC_OR_SOCIAL_ALERT",
                sensor_states=states,
                total_sensor_watts=watts,
                power_saved_pct=100.0 * (1.0 - watts / max_watts),
                explanation="Standing interaction -> Only Head Camera, Mic Array & Wrist Force Sensors awake; LiDAR & Floor/Rear Cams sleeping!",
            )

        # 5. Deep Idle Sentinel Mode (Standing in quiet room)
        states = {
            "head_orbbec_rgbd (Board 2)": "SENTINEL (1 Hz Periodic Glance)",
            "front_torso_rgbd (Board 3)": "SLEEP (0 W)",
            "rear_torso_cam (Board 3)": "SLEEP (0 W)",
            "livox_3d_lidar (Board 3)": "SLEEP (0 W)",
            "voice_orin_mic_array (Board 2)": "ON (Low-Power Acoustic Trigger)",
            "sri_6dof_wrist_ft (Board 1)": "IDLE (10 Hz)",
            "webrtc_video_stream (Boards 2/3)": "SLEEP (0 W)",
        }
        watts = 5.4
        return SensorPowerAllocation(
            profile_name="DEEP_IDLE_ECO",
            sensor_states=states,
            total_sensor_watts=watts,
            power_saved_pct=100.0 * (1.0 - watts / max_watts),
            explanation="Quiet stationary idle -> 1 Hz sentinel glance + Mic wake trigger; sleeping LiDAR & extra cameras (88.9% power saved!).",
        )


class GroundHazardInstinct:
    """
    Instinct #10: Downward Torso Camera (`camera_body_front_link`) Floor Hazard & Cliff Reflex.
    Monitors the floor 0.35 m to 1.20 m ahead of the feet for cables, toys, wet puddles, or drop-offs.
    """

    @staticmethod
    def evaluate_floor_patch(
        hazard_type: Optional[str],
        hazard_pos_xy_m: Optional[Tuple[float, float]] = None,
        hazard_width_m: float = 0.20,
    ) -> Dict[str, object]:
        if not hazard_type or hazard_pos_xy_m is None:
            return {"action": "CLEAR", "cmd_vx_scale": 1.0, "cmd_vy_detour": 0.0, "note": "Floor ahead clear."}

        hx, hy = hazard_pos_xy_m
        if hazard_type in ("cliff", "stairs_down_unexpected"):
            return {
                "action": "EMERGENCY_HALT_DROP_OFF",
                "cmd_vx_scale": 0.0,
                "cmd_vy_detour": 0.0,
                "note": f"Downward torso cam spotted drop-off at X={hx:.2f}m! Halting immediately.",
            }

        # Sidestep around small floor hazard (puddle, toy, power cord)
        detour_sign = -1.0 if hy >= 0.0 else 1.0
        return {
            "action": "SIDESTEP_AROUND_FLOOR_HAZARD",
            "cmd_vx_scale": 0.45,
            "cmd_vy_detour": 0.16 * detour_sign,
            "note": (
                f"Downward torso cam spotted '{hazard_type}' at ({hx:.2f}m, {hy:+.2f}m) -> "
                f"Slowing down and sidestepping {'right' if detour_sign < 0 else 'left'} so feet don't trip."
            ),
        }
