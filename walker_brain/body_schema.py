#!/usr/bin/env python3
"""
Module 1: `body_schema.py` — 3D Body Envelope, Narrow-Gap Sideways Squeeze, Reach-With-Waist & Finger Guard
===========================================================================================================
Implements 5 core somatic instincts grounded in the exact `tiangong2pro_torq.xml` URDF geometry:
  1. Instinct #1 (`BodySchemaEnvelope`):
     Knows the robot's exact physical dimensions in every posture:
       - Normal stance width (arms relaxed) : 0.58 m
       - Arms tucked tightly against ribs   : 0.48 m
       - Rotated 90 deg sideways (chest-to-back depth): 0.23 m
     Automatically switches between Forward Walk, Arm-Tuck Walk, and 90-deg Sideways Sidestep (`vy`)
     when encountering narrow corridors or doorways!
  2. Instinct #2 (`OverheadClearanceReflex`):
     Ducks head (`head_pitch_joint`) and crouches pelvis when passing under low overhangs (< 1.72 m).
  3. Instinct #5 (`VisuomotorBodySchemaReacher`):
     Reaches to 3D targets `[X, Y, Z]` intuitively without pre-recorded JSON files—twisting `body_yaw_joint`
     (Motor 31) to extend reach on far targets while swinging the opposite arm backward as a counterbalance!
  4. Instinct #6 (`WristForceAndFingerGuard`):
     Protects the delicate Inspire 6-DoF fingers using the real `/arm_6dof_left` & `/arm_6dof_right`
     SRI 6-axis wrist force/torque sensors:
       - Yields compliantly under > 12 N external push
       - Refuses finger-tip pinch & switches to forearm cradle if load > 18 N (preventing snapped linkages!)
       - Releases objects smoothly when feeling a gentle 2.5 N human pull during handovers
  5. Instinct #11 (`SomaticSelfCareInstinct`):
     Monitors motor currents (`MotorStatus.cur`) and holding duration—automatically relaxing idle raised
     arms back to a low-energy neutral hang to prevent motor overheating and battery drain.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BodyDimensions:
    """Exact physical bounding-box dimensions of TienKung 2.0 Pro from `tiangong2pro_torq.xml`."""
    standing_crown_height_m: float = 1.68
    crouched_crown_height_m: float = 1.52
    width_arms_relaxed_m: float = 0.58
    width_arms_tucked_m: float = 0.48
    depth_chest_to_back_m: float = 0.23
    safety_margin_m: float = 0.06  # 3 cm clearance per side
    shoulder_z_above_pelvis_m: float = 0.377
    shoulder_y_offset_m: float = 0.212
    max_single_arm_reach_m: float = 0.62
    waist_twist_extra_reach_m: float = 0.14


@dataclass
class PassageDecision:
    """Output of Instinct #1 (Narrow-Space Body Schema) & Instinct #2 (Overhead Ducking)."""
    mode: str  # "WALK_NORMAL", "TUCK_ARMS_WALK", "TURN_SIDEWAYS_SIDESTEP", "IMPASSABLE_HALT"
    can_pass: bool
    gap_width_m: float
    effective_body_profile_m: float
    clearance_margin_m: float
    cmd_vx: float
    cmd_vy: float
    cmd_vyaw: float
    target_body_yaw_rad: float
    target_head_yaw_rad: float
    target_head_pitch_rad: float
    pelvis_crouch_dz_m: float
    tuck_arms: bool
    explanation: str


class BodySchemaEnvelope:
    """
    Instinct #1 & #2: Spatial Body-Width & Overhead Height Common Sense.
    Evaluates any gap width and overhead clearance against the robot's own 3D body schema.
    """

    def __init__(self, dims: Optional[BodyDimensions] = None):
        self.dims = dims or BodyDimensions()

    def evaluate_passage(
        self,
        gap_width_m: float,
        overhead_height_m: float = 2.50,
        preferred_side: str = "left",
    ) -> PassageDecision:
        """
        Determines how a human-like embodied agent should pass through a gap of `gap_width_m`
        and ceiling/overhang height `overhead_height_m`.
        """
        d = self.dims
        need_normal = d.width_arms_relaxed_m + d.safety_margin_m       # 0.64 m
        need_tucked = d.width_arms_tucked_m + d.safety_margin_m        # 0.54 m
        need_sideways = d.depth_chest_to_back_m + d.safety_margin_m    # 0.29 m

        # Instinct #2: Check overhead clearance (duck head & crouch knees if low overhang)
        if overhead_height_m < d.crouched_crown_height_m + 0.04:
            return PassageDecision(
                mode="IMPASSABLE_LOW_OVERHANG",
                can_pass=False,
                gap_width_m=gap_width_m,
                effective_body_profile_m=d.standing_crown_height_m,
                clearance_margin_m=overhead_height_m - d.crouched_crown_height_m,
                cmd_vx=0.0,
                cmd_vy=0.0,
                cmd_vyaw=0.0,
                target_body_yaw_rad=0.0,
                target_head_yaw_rad=0.0,
                target_head_pitch_rad=0.25,
                pelvis_crouch_dz_m=0.0,
                tuck_arms=True,
                explanation=f"Overhang ({overhead_height_m:.2f}m) is lower than crouched height ({d.crouched_crown_height_m:.2f}m). Halting.",
            )

        duck_needed = overhead_height_m < (d.standing_crown_height_m + 0.06)
        crouch_dz = -0.14 if duck_needed else 0.0
        head_pitch = 0.35 if duck_needed else 0.05

        # Instinct #1: Compare gap width against Front Stance vs. Arms-Tucked vs. 90-deg Sideways Profile
        if gap_width_m >= need_normal:
            return PassageDecision(
                mode="WALK_NORMAL",
                can_pass=True,
                gap_width_m=gap_width_m,
                effective_body_profile_m=d.width_arms_relaxed_m,
                clearance_margin_m=gap_width_m - d.width_arms_relaxed_m,
                cmd_vx=0.35 if duck_needed else 0.50,
                cmd_vy=0.0,
                cmd_vyaw=0.0,
                target_body_yaw_rad=0.0,
                target_head_yaw_rad=0.0,
                target_head_pitch_rad=head_pitch,
                pelvis_crouch_dz_m=crouch_dz,
                tuck_arms=False,
                explanation=(
                    f"Gap ({gap_width_m:.2f}m) comfortably fits normal shoulder width ({d.width_arms_relaxed_m:.2f}m)."
                    + (" Ducking under low overhang." if duck_needed else "")
                ),
            )

        if gap_width_m >= need_tucked:
            return PassageDecision(
                mode="TUCK_ARMS_WALK",
                can_pass=True,
                gap_width_m=gap_width_m,
                effective_body_profile_m=d.width_arms_tucked_m,
                clearance_margin_m=gap_width_m - d.width_arms_tucked_m,
                cmd_vx=0.25,
                cmd_vy=0.0,
                cmd_vyaw=0.0,
                target_body_yaw_rad=0.0,
                target_head_yaw_rad=0.0,
                target_head_pitch_rad=head_pitch,
                pelvis_crouch_dz_m=crouch_dz,
                tuck_arms=True,
                explanation=(
                    f"Tight gap ({gap_width_m:.2f}m) < relaxed width ({d.width_arms_relaxed_m:.2f}m). "
                    f"Tucking elbows inward to {d.width_arms_tucked_m:.2f}m profile and slowing down."
                ),
            )

        if gap_width_m >= need_sideways:
            # Turn upper body / pelvis 90 deg sideways so 0.23m depth faces the narrow gap,
            # while counter-rotating head_yaw 90 deg so the head camera still looks in the direction of motion!
            sign = 1.0 if preferred_side == "left" else -1.0
            return PassageDecision(
                mode="TURN_SIDEWAYS_SIDESTEP",
                can_pass=True,
                gap_width_m=gap_width_m,
                effective_body_profile_m=d.depth_chest_to_back_m,
                clearance_margin_m=gap_width_m - d.depth_chest_to_back_m,
                cmd_vx=0.0,
                cmd_vy=0.15 * sign,
                cmd_vyaw=0.0,
                target_body_yaw_rad=-sign * (math.pi / 2.0),
                target_head_yaw_rad=sign * (math.pi / 2.0),  # Keep eyes looking into the gap!
                target_head_pitch_rad=head_pitch,
                pelvis_crouch_dz_m=crouch_dz,
                tuck_arms=True,
                explanation=(
                    f"Narrow gap ({gap_width_m:.2f}m) blocks front shoulders ({d.width_arms_tucked_m:.2f}m), "
                    f"but fits sideways chest depth ({d.depth_chest_to_back_m:.2f}m)! "
                    f"Rotating 90 deg sideways, tucking arms, swiveling head to look ahead, and sidestepping slowly."
                ),
            )

        return PassageDecision(
            mode="IMPASSABLE_HALT",
            can_pass=False,
            gap_width_m=gap_width_m,
            effective_body_profile_m=d.depth_chest_to_back_m,
            clearance_margin_m=gap_width_m - d.depth_chest_to_back_m,
            cmd_vx=0.0,
            cmd_vy=0.0,
            cmd_vyaw=0.0,
            target_body_yaw_rad=0.0,
            target_head_yaw_rad=0.0,
            target_head_pitch_rad=0.15,
            pelvis_crouch_dz_m=0.0,
            tuck_arms=True,
            explanation=(
                f"Gap ({gap_width_m:.2f}m) is narrower than even our 90-deg sideways torso depth "
                f"({d.depth_chest_to_back_m:.2f}m + margin). Physically impassable—seeking alternate route."
            ),
        )


class VisuomotorBodySchemaReacher:
    """
    Instinct #5: Fast Whole-Body Visuomotor Reach-With-Waist & Opposite-Arm Counterbalance.
    Maps a 3D target `[x, y, z]` (in pelvis frame) directly into:
      - Automatic hand selection (Left vs. Right vs. Bimanual)
      - Waist rotation (`body_yaw_joint`, Motor 31) when reaching far forward/across
      - 7-DoF arm joint angles for the reaching arm
      - Opposite-arm backward counterbalance swing so the Center of Mass stays over the feet!
    """

    def __init__(self, dims: Optional[BodyDimensions] = None):
        self.dims = dims or BodyDimensions()

    def compute_intuitive_reach(
        self,
        target_pelvis_xyz: np.ndarray,
        object_width_m: float = 0.08,
    ) -> Dict[str, float]:
        """
        Computes a whole-body upper-torso joint configuration to reach `target_pelvis_xyz`
        without any pre-recorded JSON trajectory.
        """
        tx, ty, tz = float(target_pelvis_xyz[0]), float(target_pelvis_xyz[1]), float(target_pelvis_xyz[2])
        bimanual = object_width_m >= 0.22
        active_side = "both" if bimanual else ("l" if ty >= 0.0 else "r")

        # 1. Instinctive Waist Twist (`body_yaw_joint`, Motor 31):
        # If reaching past 0.35 m forward with one hand, twist the waist toward the target so the shoulder advances!
        reach_dist_xy = math.hypot(tx, ty)
        if active_side == "l" and tx > 0.32:
            waist_yaw = np.clip((tx - 0.32) * 1.35, 0.0, 0.55)   # Turn left shoulder forward (CCW/CW)
            waist_yaw = -waist_yaw  # -Z yaw advances left (+Y) shoulder forward (+X)
        elif active_side == "r" and tx > 0.32:
            waist_yaw = np.clip((tx - 0.32) * 1.35, 0.0, 0.55)   # +Z yaw advances right (-Y) shoulder forward (+X)
        else:
            waist_yaw = 0.0

        # 2. Head naturally looks at the target being reached
        head_yaw = np.clip(math.atan2(ty, max(0.15, tx)) - waist_yaw, -1.4, 1.4)
        head_pitch = np.clip(math.atan2(0.56 - tz, max(0.20, reach_dist_xy)), -0.40, 0.40)

        joints: Dict[str, float] = {
            "body_yaw_joint": float(waist_yaw),
            "head_yaw_joint": float(head_yaw),
            "head_pitch_joint": float(head_pitch),
            "head_roll_joint": 0.0,
        }

        # 3. Compute reaching arm joints + Opposite-arm CoM counterbalance!
        for side in ("l", "r"):
            sign_y = 1.0 if side == "l" else -1.0
            if active_side in (side, "both"):
                # Account for shoulder advancement from waist_yaw
                sh_x = -sign_y * self.dims.shoulder_y_offset_m * math.sin(waist_yaw)
                sh_y = sign_y * self.dims.shoulder_y_offset_m * math.cos(waist_yaw)
                sh_z = self.dims.shoulder_z_above_pelvis_m

                dx = tx - sh_x
                dy = (ty + sign_y * 0.5 * object_width_m if bimanual else ty) - sh_y
                dz = tz - sh_z
                dist = np.clip(math.sqrt(dx * dx + dy * dy + dz * dz), 0.18, self.dims.max_single_arm_reach_m)

                # Analytical human-like elbow-down inverse kinematics
                l1, l2 = 0.30, 0.29  # Upper arm + forearm lengths from `tiangong2pro_torq.xml`
                cos_elbow = np.clip((dist * dist - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -0.98, 0.98)
                elbow_pitch = -math.acos(cos_elbow)  # Negative bends elbow forward

                alpha = math.atan2(dx, -dz)
                beta = math.acos(np.clip((l1 * l1 + dist * dist - l2 * l2) / (2.0 * l1 * dist), -0.98, 0.98))
                shoulder_pitch = -(alpha - beta)
                shoulder_roll = np.clip(math.atan2(dy, max(0.12, math.hypot(dx, dz))), -0.25, 0.85) if side == "l" else np.clip(math.atan2(dy, max(0.12, math.hypot(dx, dz))), -0.85, 0.25)

                joints[f"shoulder_pitch_{side}_joint"] = float(np.clip(shoulder_pitch, -2.2, 1.5))
                joints[f"shoulder_roll_{side}_joint"] = float(shoulder_roll)
                joints[f"shoulder_yaw_{side}_joint"] = 0.0
                joints[f"elbow_pitch_{side}_joint"] = float(np.clip(elbow_pitch, -2.3, -0.15))
                joints[f"elbow_yaw_{side}_joint"] = 0.0
                joints[f"wrist_pitch_{side}_joint"] = float(np.clip(-0.35 * (shoulder_pitch + elbow_pitch), -0.6, 0.8))
                joints[f"wrist_roll_{side}_joint"] = 0.0
            else:
                # Counterbalance Reflex: swing opposite arm backward proportionally to forward reach!
                counter_pitch = np.clip(0.55 * max(0.0, tx - 0.25), 0.0, 0.42)
                joints[f"shoulder_pitch_{side}_joint"] = float(counter_pitch)
                joints[f"shoulder_roll_{side}_joint"] = float(0.12 * sign_y)
                joints[f"shoulder_yaw_{side}_joint"] = 0.0
                joints[f"elbow_pitch_{side}_joint"] = -0.35
                joints[f"elbow_yaw_{side}_joint"] = 0.0
                joints[f"wrist_pitch_{side}_joint"] = 0.0
                joints[f"wrist_roll_{side}_joint"] = 0.0

        return joints

    @staticmethod
    def get_tucked_arm_joints() -> Dict[str, float]:
        """Compact arm-tuck posture that narrows shoulder profile to 0.48 m for tight corridors."""
        return {
            "shoulder_pitch_l_joint": -0.25,
            "shoulder_roll_l_joint": 0.02,
            "shoulder_yaw_l_joint": -0.35,
            "elbow_pitch_l_joint": -1.45,
            "elbow_yaw_l_joint": 0.0,
            "wrist_pitch_l_joint": 0.0,
            "wrist_roll_l_joint": 0.0,
            "shoulder_pitch_r_joint": -0.25,
            "shoulder_roll_r_joint": -0.02,
            "shoulder_yaw_r_joint": 0.35,
            "elbow_pitch_r_joint": -1.45,
            "elbow_yaw_r_joint": 0.0,
            "wrist_pitch_r_joint": 0.0,
            "wrist_roll_r_joint": 0.0,
        }


class WristForceAndFingerGuard:
    """
    Instinct #6: 6-DoF Finger Linkage Protection & SRI 6-Axis Wrist Force Reflex.
    Reads `/arm_6dof_left` and `/arm_6dof_right` (`geometry_msgs/WrenchStamped`) on Board 1.
    Prevents the delicate Inspire 6-DoF hand fingers from snapping under heavy loads!
    """

    def __init__(
        self,
        max_safe_finger_load_n: float = 18.0,  # ~1.8 kg max on delicate 6-DoF fingers!
        compliant_yield_threshold_n: float = 10.0,
        handover_tug_threshold_n: float = 2.5,
    ):
        self.max_safe_finger_load_n = max_safe_finger_load_n
        self.compliant_yield_threshold_n = compliant_yield_threshold_n
        self.handover_tug_threshold_n = handover_tug_threshold_n

    def evaluate_wrist_wrench(
        self,
        wrench_left_xyz_n: np.ndarray,
        wrench_right_xyz_n: np.ndarray,
        holding_object: bool = False,
        awaiting_human_handover: bool = False,
    ) -> Dict[str, object]:
        """
        Evaluates 3D force vectors `[Fx, Fy, Fz]` from the left & right SRI 6-DoF wrist sensors.
        Returns protective finger commands, compliant arm yield offsets, and voice feedback if overloaded.
        """
        f_l = float(np.linalg.norm(wrench_left_xyz_n))
        f_r = float(np.linalg.norm(wrench_right_xyz_n))
        peak_f = max(f_l, f_r)

        # 1. Check for gentle human pull during object handover (human taking cup/bottle/tool from robot)
        if awaiting_human_handover and peak_f >= self.handover_tug_threshold_n:
            return {
                "status": "HANDOVER_RELEASE",
                "peak_force_n": peak_f,
                "finger_open_ratio": 0.88,  # Open 6-DoF fingers smoothly
                "switch_to_forearm_cradle": False,
                "compliant_offset_l_m": np.zeros(3),
                "compliant_offset_r_m": np.zeros(3),
                "spoken_response": "There you go, I've got a safe release.",
            }

        # 2. Check for Finger-Snap Overload (> 18 N / ~1.8 kg on delicate 6-DoF fingers!)
        if peak_f > self.max_safe_finger_load_n:
            return {
                "status": "FINGER_OVERLOAD_PROTECTED",
                "peak_force_n": peak_f,
                "finger_open_ratio": 0.85,  # Open fingers immediately so linkages don't snap!
                "switch_to_forearm_cradle": True,
                "compliant_offset_l_m": np.array([0.0, 0.0, -0.04]),
                "compliant_offset_r_m": np.array([0.0, 0.0, -0.04]),
                "spoken_response": (
                    f"Careful—that load ({peak_f / 9.81:.1f} kg) is too heavy for my 6-DoF fingers. "
                    "Switching to forearm cradle support so my finger gears stay safe."
                ),
            }

        # 3. Compliant Spring Yielding (if bumped or pushed by > 10 N)
        offset_l = np.zeros(3)
        offset_r = np.zeros(3)
        if f_l > self.compliant_yield_threshold_n:
            offset_l = 0.004 * (wrench_left_xyz_n - (wrench_left_xyz_n / f_l) * self.compliant_yield_threshold_n)
        if f_r > self.compliant_yield_threshold_n:
            offset_r = 0.004 * (wrench_right_xyz_n - (wrench_right_xyz_n / f_r) * self.compliant_yield_threshold_n)

        return {
            "status": "COMPLIANT_YIELD" if (f_l > self.compliant_yield_threshold_n or f_r > self.compliant_yield_threshold_n) else "NOMINAL_SAFE",
            "peak_force_n": peak_f,
            "finger_open_ratio": 0.35 if holding_object else 0.75,
            "switch_to_forearm_cradle": False,
            "compliant_offset_l_m": offset_l,
            "compliant_offset_r_m": offset_r,
            "spoken_response": None,
        }


class SomaticSelfCareInstinct:
    """
    Instinct #11: Joint-Fatigue & Thermal Relief.
    Humans don't leave their arms raised in mid-air when a gesture finishes.
    Monitors arm elevation duration and motor currents (`MotorStatus.cur`) and relaxes idle arms.
    """

    def __init__(self, max_idle_raised_sec: float = 6.0):
        self.max_idle_raised_sec = max_idle_raised_sec
        self.arm_raised_timer_sec: float = 0.0

    def update(
        self,
        dt: float,
        shoulder_pitch_l: float,
        shoulder_pitch_r: float,
        task_active: bool,
        max_motor_current_a: float = 1.5,
    ) -> Tuple[bool, str]:
        is_raised = (abs(shoulder_pitch_l) > 0.35) or (abs(shoulder_pitch_r) > 0.35)
        if is_raised and not task_active:
            self.arm_raised_timer_sec += dt
        else:
            self.arm_raised_timer_sec = 0.0

        if self.arm_raised_timer_sec >= self.max_idle_raised_sec or max_motor_current_a > 12.0:
            return (
                True,
                f"Arm held idle for {self.arm_raised_timer_sec:.1f}s (peak current {max_motor_current_a:.1f}A) -> "
                "Relaxing shoulders to zero-strain neutral hang.",
            )
        return False, "Posture strain nominal."
