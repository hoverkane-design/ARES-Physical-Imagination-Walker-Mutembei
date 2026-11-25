#!/usr/bin/env python3
# Walker Mutembei (Model TK2301 / Serial 00TG20PD25330013) Unified ARES Daemon
# Grounded in Official UBTECH Walker TienKung SDK V2.0.7.x Specifications:
# - Board 1 (192.168.41.1, Intel i7-1355U): /bodyctrl (400 Hz Arm / 1 kHz Leg EtherCAT),
#   /rl_control, /hric/robot/float_base_rpyz_cmd (400 Hz RPYZ floating base attitude),
#   /arm_6dof_left & /arm_6dof_right (100 Hz SRI 6-axis force/torque),
#   /inspire_hand/state/{left,right}_hand & /inspire_hand/get_force/{left,right}_hand
# - Board 2 (192.168.41.2, Jetson AGX Orin 64GB, 275 TOPS): /ob_camera_head, Ollama LLM,
#   Lyre / Kokoro-82M CUDA TTS
# - Board 3 (192.168.41.3, Jetson AGX Orin 64GB, 275 TOPS): /livox/lidar (Mid-360),
#   /ob_camera_waist & /ob_camera_back

import dataclasses
import math
import sqlite3
import time
import uuid
from typing import Dict, List, Optional, Tuple
import numpy as np

from walker_brain.body_schema import (
    BodyDimensions,
    BodySchemaEnvelope,
    VisuomotorBodySchemaReacher,
    WristForceAndFingerGuard,
    SomaticSelfCareInstinct,
)
from walker_brain.active_perception import (
    ActiveGazeAndBlindSpotManager,
    SensorEnergyManager,
    GroundHazardInstinct,
)
from walker_brain.social_and_solver import (
    DetectedPerson,
    ProactiveCommonSenseEngine,
    CrowdSocialNavigator,
    AutonomousProblemSolver,
)


# Official TK2301 7-DoF Arm Torque Constants Kt (Nm/A) from SDK Section 3.5
# Joints 11..17 (Left) and 21..27 (Right):
# [ShoulderPitch, ShoulderRoll, ShoulderYaw, ElbowPitch, WristYaw, WristPitch, WristRoll]
TK2301_ARM_KT_NM_PER_A = np.array([3.207, 3.370, 2.430, 2.430, 1.710, 2.160, 2.160], dtype=float)
TK2301_ARM_RATED_TORQUE_NM = np.array([35.0, 20.0, 10.0, 10.0, 5.0, 3.0, 3.0], dtype=float)
TK2301_ARM_PEAK_TORQUE_NM = np.array([91.0, 60.0, 30.0, 30.0, 15.0, 9.0, 9.0], dtype=float)

# Official TK2301 MotorStatus.error Fault Code Dictionary from SDK Sections 8.5 to 8.8
TK2301_MOTOR_ERROR_CODES = {
    0: "NOMINAL_OK",
    1: "MOTOR_OVER_TEMPERATURE",
    2: "OVER_CURRENT",
    3: "UNDER_VOLTAGE",
    4: "JOINT_MOS_OVER_TEMPERATURE",
    5: "MECHANICAL_STALL",
    6: "OVER_VOLTAGE",
    7: "PHASE_LOSS",
    8: "ENCODER_ERROR",
    33072: "DEVICE_OFFLINE_OR_STO_UNZEROED",
    33073: "JOINT_POSITION_OUT_OF_RANGE",
}


# Output of Subsystem 4.1 (Egocentric RGB-D Perception & 3-Board Telemetry)
@dataclasses.dataclass
class WalkerSensorySnapshot:
    object_class: str
    target_pelvis_xyz: np.ndarray
    object_dims_lwh: np.ndarray
    object_rpy_rad: np.ndarray
    obstacle_detected: bool
    obstacle_xyz: Optional[np.ndarray]
    gap_width_m: float
    overhead_height_m: float
    wrist_wrench_left_n: np.ndarray
    wrist_wrench_right_n: np.ndarray
    rhand_encoder_ratios: np.ndarray
    rhand_force_ratios: np.ndarray
    arm_motor_currents_a: np.ndarray
    arm_motor_temps_c: np.ndarray
    arm_status_error_codes: List[int]
    float_base_rpyz_actual: np.ndarray
    battery_pct: float
    walk_mode: int
    is_console_control: bool


# Output of Subsystem 4.2 (400/500 Hz Execution Monitoring & Kt Torque-Residual Observer)
@dataclasses.dataclass
class ARESMonitorReport:
    status: str
    ee_tracking_error_m: float
    visual_object_shift_m: float
    finger_encoder_discrepancy: float
    measured_joint_torques_nm: np.ndarray
    tau_residual_norm_nm: float
    deduced_payload_mass_kg: float
    mass_discrepancy_kg: float
    decoded_motor_faults: List[str]
    rl_arm_lock_detected: bool
    finger_overload_detected: bool


# Output of Subsystem 4.3 (7-Category Failure Detection and Diagnosis)
@dataclasses.dataclass
class ARESDiagnosisReport:
    diag_id: str
    failure_category: str
    sub_cause: str
    confidence: float
    safe_to_recover: bool
    internal_monologue: str


# Output of Subsystem 4.4 (Counterfactual Physical Imagination Rollout)
@dataclasses.dataclass
class PhysicalImaginationRollout:
    strategy_name: str
    joint_targets: Dict[str, float]
    float_base_rpyz_cmd: np.ndarray
    hand_open_ratio: float
    use_forearm_cradle: bool
    predicted_cop_margin_m: float
    predicted_peak_torque_nm: float
    collision_free: bool
    rollout_score: float


# Subsystem 4.6: Persistent 3NF SQLite Episodic Experience and Memory System
class WalkerEpisodicMemoryDB:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS TASK_EXECUTIONS (
                execution_id TEXT PRIMARY KEY,
                task_instruction TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                final_outcome INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS FAILURE_DIAGNOSES (
                diag_id TEXT PRIMARY KEY,
                execution_id TEXT NOT NULL,
                failure_category TEXT NOT NULL,
                sub_cause TEXT NOT NULL,
                confidence REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS IMAGINATION_ROLLOUTS (
                rollout_id TEXT PRIMARY KEY,
                diag_id TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                orientation_bin_deg INTEGER NOT NULL,
                cop_margin_m REAL NOT NULL,
                rollout_score REAL NOT NULL,
                recovery_succeeded INTEGER NOT NULL
            );
            """
        )
        self.conn.commit()

    @staticmethod
    def orientation_bin(yaw_rad: float) -> int:
        deg = int(round(math.degrees(yaw_rad) / 30.0) * 30)
        return max(-90, min(90, deg))

    def get_inhibited_strategies(self, failure_category: str, yaw_rad: float) -> List[str]:
        obin = self.orientation_bin(yaw_rad)
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT r.strategy_name
            FROM IMAGINATION_ROLLOUTS r
            JOIN FAILURE_DIAGNOSES d ON r.diag_id = d.diag_id
            WHERE d.failure_category = ?
              AND r.orientation_bin_deg = ?
              AND r.recovery_succeeded = 0
            """,
            (failure_category, obin),
        )
        return [row[0] for row in cur.fetchall()]

    def record_episode(
        self,
        execution_id: str,
        task_instruction: str,
        scenario_id: str,
        diag: ARESDiagnosisReport,
        rollout: PhysicalImaginationRollout,
        yaw_rad: float,
        succeeded: bool,
    ) -> None:
        obin = self.orientation_bin(yaw_rad)
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO TASK_EXECUTIONS VALUES (?, ?, ?, ?, ?)",
            (execution_id, task_instruction, scenario_id, int(time.time() * 1000), int(succeeded)),
        )
        cur.execute(
            "INSERT INTO FAILURE_DIAGNOSES VALUES (?, ?, ?, ?, ?)",
            (diag.diag_id, execution_id, diag.failure_category, diag.sub_cause, diag.confidence),
        )
        cur.execute(
            "INSERT INTO IMAGINATION_ROLLOUTS VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                diag.diag_id,
                rollout.strategy_name,
                obin,
                rollout.predicted_cop_margin_m,
                rollout.rollout_score,
                int(succeeded),
            ),
        )
        self.conn.commit()


# Unified ARES Intelligent Execution Layer + 12 Embodied Instincts for TK2301 Walker Mutembei
class WalkerMutembeiARESBrain:
    def __init__(self, db_path: str = ":memory:"):
        self.dims = BodyDimensions()
        self.body_envelope = BodySchemaEnvelope(self.dims)
        self.visuomotor_reacher = VisuomotorBodySchemaReacher(self.dims)
        self.finger_guard = WristForceAndFingerGuard()
        self.self_care = SomaticSelfCareInstinct()
        self.gaze_manager = ActiveGazeAndBlindSpotManager()
        self.energy_manager = SensorEnergyManager()
        self.floor_instinct = GroundHazardInstinct()
        self.social_engine = ProactiveCommonSenseEngine()
        self.crowd_navigator = CrowdSocialNavigator()
        self.stuck_solver = AutonomousProblemSolver()
        self.memory_db = WalkerEpisodicMemoryDB(db_path=db_path)

        self.filtered_mass_kg: float = 0.0
        self.rl_q_table: Dict[Tuple[str, str], float] = {}
        self.internal_monologue_log: List[str] = []

    # Subsystem 4.2: 400/500 Hz Execution Monitoring using exact TK2301 Torque Constants Kt (Nm/A),
    # 100 Hz SRI 6-Axis Wrist Sensor (/arm_6dof_right), Inspire Hand Encoders & Force Readback
    # (/inspire_hand/get_angle & /inspire_hand/get_force), and Official SDK Error Codes.
    def monitor_execution_state(
        self,
        snap: WalkerSensorySnapshot,
        commanded_ee_xyz: np.ndarray,
        actual_ee_xyz: np.ndarray,
        expected_object_xyz: np.ndarray,
        commanded_hand_ratio: float,
        expected_mass_kg: float,
        gravity_bias_torques_nm: np.ndarray,
    ) -> ARESMonitorReport:
        ee_err = float(np.linalg.norm(actual_ee_xyz - commanded_ee_xyz))
        vis_shift = float(np.linalg.norm(snap.target_pelvis_xyz - expected_object_xyz))

        # Compute actual joint torques tau_act = Kt * current_A from SDK Section 3.5
        measured_torques_nm = TK2301_ARM_KT_NM_PER_A * snap.arm_motor_currents_a
        tau_residual_vec = measured_torques_nm - gravity_bias_torques_nm
        tau_residual_norm = float(np.linalg.norm(tau_residual_vec))

        # Compare commanded Inspire 6-DoF finger opening percentage against GetAngleAct readback
        mean_actual_finger = float(np.mean(snap.rhand_encoder_ratios[:5]))
        finger_disc = abs(mean_actual_finger - commanded_hand_ratio)

        # Deduce payload mass from 100 Hz SRI 6-Axis force sensor (/arm_6dof_right)
        fz_wrist = abs(float(snap.wrist_wrench_right_n[2]))
        instant_mass = fz_wrist / 9.81
        if fz_wrist < 1.0:
            self.filtered_mass_kg = instant_mass
        else:
            self.filtered_mass_kg = 0.70 * self.filtered_mass_kg + 0.30 * instant_mass
        mass_disc = abs(self.filtered_mass_kg - expected_mass_kg)

        # Decode official SDK Section 8.7 MotorStatus.error codes (0, 1..8, 33072, 33073)
        decoded_faults = [
            TK2301_MOTOR_ERROR_CODES.get(c, f"UNKNOWN_ERR_{c}")
            for c in snap.arm_status_error_codes
            if c != 0
        ]
        rl_locked = (snap.walk_mode != 3) or (not snap.is_console_control)
        wrist_eval = self.finger_guard.evaluate_wrist_wrench(
            snap.wrist_wrench_left_n, snap.wrist_wrench_right_n, holding_object=True
        )
        overload = wrist_eval["status"] == "FINGER_OVERLOAD_PROTECTED"

        status = "SUCCESSFUL"
        if (
            len(decoded_faults) > 0
            or rl_locked
            or overload
            or tau_residual_norm > 42.0
            or ee_err > 0.025
            or vis_shift > 0.035
            or mass_disc > 0.20
        ):
            status = "DISCREPANCY_FLAGGED"

        return ARESMonitorReport(
            status=status,
            ee_tracking_error_m=ee_err,
            visual_object_shift_m=vis_shift,
            finger_encoder_discrepancy=finger_disc,
            measured_joint_torques_nm=measured_torques_nm,
            tau_residual_norm_nm=tau_residual_norm,
            deduced_payload_mass_kg=self.filtered_mass_kg,
            mass_discrepancy_kg=mass_disc,
            decoded_motor_faults=decoded_faults,
            rl_arm_lock_detected=rl_locked,
            finger_overload_detected=overload,
        )

    # Subsystem 4.3: 7-Category Evidence-Based Failure Detection and Diagnosis
    def diagnose_failure(
        self, task_phase: str, snap: WalkerSensorySnapshot, mon: ARESMonitorReport, expected_mass_kg: float
    ) -> ARESDiagnosisReport:
        cat = "NOMINAL_SUCCESS"
        sub_cause = "Execution tracking within nominal bounds"
        conf = 0.97

        if len(mon.decoded_motor_faults) > 0 or mon.rl_arm_lock_detected:
            cat = "ACTUATOR_LIMITATION"
            sub_cause = f"Motor fault {mon.decoded_motor_faults} or rl_control impedance lock"
            conf = 0.96
        elif mon.finger_overload_detected:
            cat = "ACTUATOR_LIMITATION"
            sub_cause = f"Payload ({mon.deduced_payload_mass_kg:.2f} kg) exceeds 18N Inspire 6-DoF finger limit"
            conf = 0.95
        elif mon.tau_residual_norm_nm > 42.0 or snap.obstacle_detected:
            cat = "UNEXPECTED_COLLISION"
            sub_cause = "Obstacle intrusion in /ob_camera_head or high Kt*I torque residual spike"
            conf = 0.94
        elif task_phase == "LIFT" and mon.deduced_payload_mass_kg < 0.18 * expected_mass_kg:
            if mon.finger_encoder_discrepancy > 0.12 or float(np.mean(snap.rhand_encoder_ratios[:4])) < 0.15:
                cat = "GRASPING_FAILURE"
                sub_cause = "Inspire GetAngleAct closed to 0.05 with GetForceAct=0 -> object slipped"
                conf = 0.93
            else:
                cat = "PERCEPTION_ERROR"
                sub_cause = "Orbbec /ob_camera_head depth localization offset -> re-observation needed"
                conf = 0.86
        elif mon.visual_object_shift_m > 0.04:
            cat = "ENVIRONMENTAL_CHANGE"
            sub_cause = f"Target displaced by {mon.visual_object_shift_m*100:.1f} cm in /ob_camera_head"
            conf = 0.91
        elif mon.ee_tracking_error_m > 0.025:
            cat = "POSITIONING_ERROR"
            sub_cause = "Target requires /hric/robot/float_base_rpyz_cmd waist yaw + crouch extension"
            conf = 0.89

        monologue = (
            f"[ARES MONOLOGUE] Phase={task_phase} | Diagnosed={cat} ({sub_cause}) | "
            f"Conf={conf:.2f} | DeducedMass={mon.deduced_payload_mass_kg:.2f}kg"
        )
        self.internal_monologue_log.append(monologue)

        return ARESDiagnosisReport(
            diag_id=str(uuid.uuid4()),
            failure_category=cat,
            sub_cause=sub_cause,
            confidence=conf,
            safe_to_recover=(conf >= 0.75),
            internal_monologue=monologue,
        )

    # Subsystem 4.4 & 4.5: Counterfactual Physical Imagination ('Sim-in-the-Head')
    # Utilizing SDK Section 8.18.5 /hric/robot/float_base_rpyz_cmd (400 Hz [roll, pitch, yaw, z]),
    # 7-DoF Arm /arm/cmd_ctrl, and 6-DoF Inspire Hand /inspire_hand/set_angle_flexible
    def imagine_and_select_recovery(
        self,
        snap: WalkerSensorySnapshot,
        diag: ARESDiagnosisReport,
    ) -> PhysicalImaginationRollout:
        yaw_rad = float(snap.object_rpy_rad[2])
        inhibited = set(self.memory_db.get_inhibited_strategies(diag.failure_category, yaw_rad))

        reach_joints = self.visuomotor_reacher.compute_intuitive_reach(
            snap.target_pelvis_xyz, object_width_m=float(snap.object_dims_lwh[1])
        )
        waist_yaw = float(reach_joints.get("body_yaw_joint", 0.0))
        tz = float(snap.target_pelvis_xyz[2])
        # Compute optimal floating-base pelvis height z in [0.60, 0.89] m for low targets
        optimal_base_z = float(np.clip(0.89 - max(0.0, 0.08 - tz), 0.68, 0.89))

        # Candidate 1: Naive Open-Loop Retry (fixed base [0,0,0,0.89], no waist yaw, no counterbalance)
        naive_joints = dict(reach_joints)
        naive_joints["body_yaw_joint"] = 0.0
        naive_joints["shoulder_pitch_l_joint"] = 0.0
        rpyz_naive = np.array([0.0, 0.0, 0.0, 0.89], dtype=float)

        # Candidate 2: 400 Hz FloatBaseRPYZ Waist-Yaw/Crouch + Opposite-Arm Counterbalance + Adapted Wrist
        adapted_joints = dict(reach_joints)
        adapted_joints["wrist_pitch_r_joint"] = float(snap.object_rpy_rad[1])
        rpyz_adapted = np.array([0.0, 0.04, waist_yaw, optimal_base_z], dtype=float)

        # Candidate 3: Obstacle-Clearing Arc + 3-Tier Gaze Re-Observation (/ob_camera_head + /ob_camera_waist)
        arc_joints = dict(adapted_joints)
        arc_joints["shoulder_roll_r_joint"] = float(adapted_joints.get("shoulder_roll_r_joint", -0.25) - 0.22)
        rpyz_arc = np.array([0.0, 0.02, waist_yaw * 0.8, optimal_base_z], dtype=float)

        # Candidate 4: Clear Error Latch (/inspire_hand/set_clear_error) + Bimanual Forearm Cradle + Backward Pitch Lean
        cradle_joints = dict(adapted_joints)
        cradle_joints["elbow_pitch_r_joint"] = -1.35
        cradle_joints["elbow_pitch_l_joint"] = -1.35
        rpyz_cradle = np.array([0.0, -0.05, 0.0, 0.84], dtype=float)

        candidates = [
            ("NAIVE_OPEN_LOOP_RETRY", naive_joints, rpyz_naive, 0.35, False),
            ("FLOAT_BASE_RPYZ_WAIST_COUNTERBALANCE_ADAPT", adapted_joints, rpyz_adapted, 0.28, False),
            ("OBSTACLE_ARC_AND_ORBBEC_REOBSERVE", arc_joints, rpyz_arc, 0.30, False),
            ("CLEAR_ERROR_AND_BIMANUAL_FOREARM_CRADLE", cradle_joints, rpyz_cradle, 0.85, True),
        ]

        best_rollout = PhysicalImaginationRollout(
            strategy_name="REQUEST_OPERATOR_ASSISTANCE",
            joint_targets=naive_joints,
            float_base_rpyz_cmd=rpyz_naive,
            hand_open_ratio=0.85,
            use_forearm_cradle=False,
            predicted_cop_margin_m=0.0,
            predicted_peak_torque_nm=0.0,
            collision_free=False,
            rollout_score=-1e9,
        )

        for name, j_targets, rpyz_cmd, h_ratio, use_cradle in candidates:
            if name in inhibited:
                continue

            w_yaw = abs(float(rpyz_cmd[2]))
            base_pitch = float(rpyz_cmd[1])
            counter_pitch = float(j_targets.get("shoulder_pitch_l_joint", 0.0))
            tx = float(snap.target_pelvis_xyz[0])
            load_n = float(np.linalg.norm(snap.wrist_wrench_right_n))

            uncompensated_moment = max(0.0, tx - 0.28) * (1.0 + load_n / 25.0)
            cop_margin = (
                0.26
                - 0.35 * uncompensated_moment
                + 0.18 * counter_pitch
                + 0.12 * w_yaw
                - 0.10 * base_pitch * (load_n / 15.0)
            )
            peak_torque = 48.0 + 3.2 * load_n * (0.45 if use_cradle else 1.0)
            dexterity_bonus = 25.0 if (not use_cradle and load_n <= 18.0) else 0.0
            collision_free = True

            if name == "NAIVE_OPEN_LOOP_RETRY":
                collision_free = False
            if snap.obstacle_detected and name != "OBSTACLE_ARC_AND_ORBBEC_REOBSERVE":
                collision_free = False
            if not snap.obstacle_detected and name == "OBSTACLE_ARC_AND_ORBBEC_REOBSERVE":
                dexterity_bonus -= 12.0
            if load_n > 18.0 and not use_cradle:
                collision_free = False
            if (diag.failure_category == "ACTUATOR_LIMITATION") and (name != "CLEAR_ERROR_AND_BIMANUAL_FOREARM_CRADLE"):
                collision_free = False

            rl_bonus = 30.0 * self.rl_q_table.get((diag.failure_category, name), 0.0)
            score = (
                250.0 * cop_margin
                - 0.15 * peak_torque
                + dexterity_bonus
                + (80.0 if collision_free else -450.0)
                + rl_bonus
            )

            if score > best_rollout.rollout_score:
                best_rollout = PhysicalImaginationRollout(
                    strategy_name=name,
                    joint_targets=j_targets,
                    float_base_rpyz_cmd=rpyz_cmd,
                    hand_open_ratio=h_ratio,
                    use_forearm_cradle=use_cradle,
                    predicted_cop_margin_m=float(cop_margin),
                    predicted_peak_torque_nm=float(peak_torque),
                    collision_free=collision_free,
                    rollout_score=float(score),
                )

        return best_rollout


def run_self_test() -> None:
    brain = WalkerMutembeiARESBrain()
    gravity_bias = np.array([12.0, 8.0, 3.0, 6.0, 1.0, 1.5, 1.0], dtype=float)
    nominal_currents = gravity_bias / TK2301_ARM_KT_NM_PER_A
    collision_currents = (gravity_bias + np.array([35.0, 25.0, 10.0, 18.0, 0.0, 0.0, 0.0])) / TK2301_ARM_KT_NM_PER_A

    scenarios = [
        ("S1_DISPLACED_FAR", np.array([0.48, -0.18, 0.04]), np.array([0.0, 0.0, 0.0]), False, 5.0, nominal_currents, [0]*7),
        ("S2_ROTATED_75DEG", np.array([0.36, -0.15, 0.10]), np.array([0.0, 0.65, 1.30]), False, 5.0, nominal_currents, [0]*7),
        ("S3_GRASP_SLIP", np.array([0.34, -0.14, 0.10]), np.array([0.0, 0.0, 0.0]), False, 0.2, nominal_currents, [0]*7),
        ("S4_OBSTACLE_BLOCK", np.array([0.38, -0.16, 0.12]), np.array([0.0, 0.0, 0.0]), True, 5.0, collision_currents, [0]*7),
        ("S5_HEAVY_2.4KG_LOAD", np.array([0.38, -0.15, 0.10]), np.array([0.0, 0.0, 0.0]), False, 23.5, nominal_currents, [0]*7),
    ]
    print("Running TK2301 Walker Mutembei (V2.0.7.x SDK) ARES + 12 Instincts Self-Test:")
    for sc_id, pos, rpy, obs, fz_n, cur_a, err_codes in scenarios:
        snap = WalkerSensorySnapshot(
            object_class="bottle_or_crate",
            target_pelvis_xyz=pos,
            object_dims_lwh=np.array([0.08, 0.08, 0.22]),
            object_rpy_rad=rpy,
            obstacle_detected=obs,
            obstacle_xyz=np.array([0.28, -0.15, 0.12]) if obs else None,
            gap_width_m=0.44,
            overhead_height_m=2.20,
            wrist_wrench_left_n=np.zeros(3),
            wrist_wrench_right_n=np.array([0.0, 0.0, -fz_n]),
            rhand_encoder_ratios=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.45]) if fz_n < 1.0 else np.array([0.35]*6),
            rhand_force_ratios=np.array([0.0]*6) if fz_n < 1.0 else np.array([0.45]*6),
            arm_motor_currents_a=cur_a,
            arm_motor_temps_c=np.array([36.0]*7),
            arm_status_error_codes=err_codes,
            float_base_rpyz_actual=np.array([0.0, 0.02, 0.0, 0.88]),
            battery_pct=72.0,
            walk_mode=3,
            is_console_control=True,
        )
        mon = brain.monitor_execution_state(
            snap=snap,
            commanded_ee_xyz=np.array([0.32, -0.14, 0.10]),
            actual_ee_xyz=pos,
            expected_object_xyz=np.array([0.32, -0.14, 0.10]),
            commanded_hand_ratio=0.35,
            expected_mass_kg=0.50,
            gravity_bias_torques_nm=gravity_bias,
        )
        diag = brain.diagnose_failure("LIFT", snap, mon, expected_mass_kg=0.50)
        rollout = brain.imagine_and_select_recovery(snap, diag)
        brain.memory_db.record_episode(
            str(uuid.uuid4()), "Pick and place object", sc_id, diag, rollout, float(rpy[2]), True
        )
        print(
            f"  [{sc_id}] Diagnosed: {diag.failure_category} (conf={diag.confidence:.2f}) "
            f"-> Selected: {rollout.strategy_name} | FloatBaseRPYZ={np.round(rollout.float_base_rpyz_cmd, 2).tolist()} "
            f"(CoP={rollout.predicted_cop_margin_m*1000:.0f}mm, Score={rollout.rollout_score:.1f})"
        )


if __name__ == "__main__":
    run_self_test()
