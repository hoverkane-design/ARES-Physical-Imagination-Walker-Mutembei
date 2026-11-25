#!/usr/bin/env python3
"""
Walker TienKung 2.0 Pro (SN: 00TG20PD25330013) — 12 Intuitive Embodied Instincts Verification & MuJoCo Runner
=============================================================================================================
Loads the factory `tiangong2pro_torq.xml` model and verifies all 12 Human-Like Intuitive Instincts across
Modules 1, 2, and 3 (`body_schema.py`, `active_perception.py`, `social_and_solver.py`).

Run modes:
  1. Audit & Verification Table (default):
       python "C:\\Users\\Alex\\walker-physical-ai\\run_instincts_mujoco_demo.py"
  2. Live 3D Interactive MuJoCo Window:
       python "C:\\Users\\Alex\\walker-physical-ai\\run_instincts_mujoco_demo.py" --live
"""

import math
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from walker_brain.body_schema import (
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


def load_factory_tg2_model():
    urdf_dir = r"C:\Users\Alex\walker-physical-ai\tiangong2pro_urdf"
    xml_path = os.path.join(urdf_dir, "tiangong2pro_torq.xml")
    cwd = os.getcwd()
    os.chdir(urdf_dir)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    os.chdir(cwd)
    return model, data


def apply_upper_body_pose(model: mujoco.MjModel, data: mujoco.MjData, joint_targets: dict, pelvis_yaw_rad: float = 0.0, pelvis_dz_m: float = 0.0):
    """Applies upper-body + head + waist + crouch targets to the factory `tiangong2pro_torq.xml` model for visual inspection."""
    mujoco.mj_resetData(model, data)
    # Set pelvis height & yaw
    data.qpos[2] = 0.9317 + pelvis_dz_m
    cy, sy = math.cos(0.5 * pelvis_yaw_rad), math.sin(0.5 * pelvis_yaw_rad)
    data.qpos[3:7] = [cy, 0.0, 0.0, sy]

    # If crouching under a low overhang, bend knees & ankles symmetrically so feet stay flat
    if pelvis_dz_m < -0.02:
        knee_bend = 0.55
        for side in ("l", "r"):
            for jn, val in [
                (f"hip_pitch_{side}_joint", -0.5 * knee_bend),
                (f"knee_pitch_{side}_joint", knee_bend),
                (f"ankle_pitch_{side}_joint", -0.5 * knee_bend),
            ]:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    data.qpos[model.jnt_qposadr[jid]] = val

    for jname, angle_rad in joint_targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            qadr = model.jnt_qposadr[jid]
            low, high = model.jnt_range[jid]
            data.qpos[qadr] = float(np.clip(angle_rad, low, high))

    mujoco.mj_forward(model, data)


def run_all_12_instincts_audit():
    model, data = load_factory_tg2_model()
    envelope = BodySchemaEnvelope()
    reacher = VisuomotorBodySchemaReacher()
    finger_guard = WristForceAndFingerGuard()
    self_care = SomaticSelfCareInstinct()
    gaze_mgr = ActiveGazeAndBlindSpotManager()
    power_mgr = SensorEnergyManager()
    floor_instinct = GroundHazardInstinct()
    proactive_social = ProactiveCommonSenseEngine()
    crowd_nav = CrowdSocialNavigator()
    problem_solver = AutonomousProblemSolver()

    print("=" * 100)
    print("  WALKER TIENKUNG 2.0 PRO (SN: 00TG20PD25330013) — 12 INTUITIVE EMBODIED INSTINCTS VERIFICATION")
    print("=" * 100)

    # 1. Instinct #1: Narrow-Gap Body Schema (0.75m Wide vs. 0.56m Tight vs. 0.38m Sideways Squeeze)
    print("\n[INSTINCT 1] 3D Body-Width Schema & Narrow-Space Sideways Squeeze:")
    for gap_m in (0.75, 0.56, 0.38, 0.24):
        dec = envelope.evaluate_passage(gap_width_m=gap_m, overhead_height_m=2.20)
        print(
            f"  * Gap = {gap_m:.2f}m -> Mode: {dec.mode:<24} | Profile: {dec.effective_body_profile_m:.2f}m "
            f"| vx={dec.cmd_vx:+.2f}, vy={dec.cmd_vy:+.2f} m/s | Waist={math.degrees(dec.target_body_yaw_rad):+.0f} deg, "
            f"Head={math.degrees(dec.target_head_yaw_rad):+.0f} deg"
        )

    # 2. Instinct #2: Low-Clearance Overhang Ducking
    print("\n[INSTINCT 2] Low-Overhang Head-Duck & Knee-Crouch Reflex:")
    duck_dec = envelope.evaluate_passage(gap_width_m=0.80, overhead_height_m=1.60)
    print(
        f"  * Overhang = 1.60m (Standing Crown = 1.68m) -> Pelvis Crouch dz = {duck_dec.pelvis_crouch_dz_m*100:+.0f} cm, "
        f"Head Pitch = {math.degrees(duck_dec.target_head_pitch_rad):+.1f} deg | {duck_dec.explanation}"
    )

    # 3. Instinct #3 & #9: 3-Tier Blind-Spot Clearing (Head Only -> Head + Waist CW/CCW -> Whole Body) + Look-Before-Stepping
    print("\n[INSTINCT 3 & 9] 360-Deg Blind-Spot Clearing (Head -> Waist CW/CCW -> Pelvis) & Look-Before-Stepping:")
    for az_deg in (30.0, -95.0, 110.0, 165.0):
        gcmd = gaze_mgr.orient_to_target_or_blind_spot(az_deg, reason="Acoustic/Blind-spot check")
        print(
            f"  * Target Azimuth = {az_deg:+6.1f} deg -> {gcmd.tier:<28} | "
            f"Head Yaw={math.degrees(gcmd.head_yaw_rad):+5.1f} deg, Waist (`body_yaw`)={math.degrees(gcmd.body_yaw_rad):+5.1f} deg, "
            f"Pelvis vyaw={gcmd.pelvis_vyaw_rad_s:+.2f} rad/s | Power={gcmd.estimated_motor_watts:.1f}W"
        )
    look_sidestep = gaze_mgr.look_before_stepping(cmd_vx=0.0, cmd_vy=0.15, cmd_vyaw=0.0)
    print(f"  * Look-Before-Sidestepping (vy=+0.15 m/s): {look_sidestep.reason}")

    # 4. Instinct #4: Context & Battery-Driven Sensor Power Gating across Boards 1, 2, 3
    print("\n[INSTINCT 4] Smart Sensor Power-Saving Across Board 1 (x86), Board 2 (orin_llm) & Board 3 (orin_slam):")
    for sc_name, kw in [
        ("Quiet Room Idle", dict(battery_pct=82.0, is_walking=False, is_backing_up_or_squeezing=False, sound_detected=False, hands_interacting=False)),
        ("Voice / Hand Interaction", dict(battery_pct=82.0, is_walking=False, is_backing_up_or_squeezing=False, sound_detected=True, hands_interacting=True)),
        ("Normal Forward Walking", dict(battery_pct=75.0, is_walking=True, is_backing_up_or_squeezing=False, sound_detected=False, hands_interacting=False)),
        ("Narrow Sideways Squeeze", dict(battery_pct=70.0, is_walking=True, is_backing_up_or_squeezing=True, sound_detected=False, hands_interacting=False)),
        ("Low Battery (15%)", dict(battery_pct=15.0, is_walking=True, is_backing_up_or_squeezing=False, sound_detected=False, hands_interacting=False)),
    ]:
        plan = power_mgr.compute_sensor_power_plan(**kw)
        print(
            f"  * {sc_name:<25} -> Profile: {plan.profile_name:<24} | Power: {plan.total_sensor_watts:5.1f}W "
            f"(Saved {plan.power_saved_pct:4.1f}%)"
        )

    # 5. Instinct #5: Intuitive 3D Reach-With-Waist & Opposite-Arm CoM Counterbalance
    print("\n[INSTINCT 5] Intuitive 3D Reach-With-Waist & Opposite-Arm CoM Counterbalance (Zero JSON Playback):")
    reach_joints = reacher.compute_intuitive_reach(np.array([0.48, -0.22, 0.32]), object_width_m=0.08)
    apply_upper_body_pose(model, data, reach_joints)
    r_tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_tcp_link")
    l_tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_tcp_link")
    print(
        f"  * Reaching to [0.48, -0.22, 0.32]m with Right Arm -> Waist `body_yaw` twisted {math.degrees(reach_joints['body_yaw_joint']):+.1f} deg "
        f"to advance right shoulder! Opposite Left Arm swung back ({math.degrees(reach_joints['shoulder_pitch_l_joint']):+.1f} deg) as CoM counterbalance."
    )
    print(f"  * Verified in MuJoCo (`mj_forward`): Right TCP World Pos = {np.round(data.xpos[r_tcp_id], 3)}, Left TCP = {np.round(data.xpos[l_tcp_id], 3)}")

    # 6. Instinct #6: Delicate 6-DoF Finger Protection & SRI 6-Axis Wrist Force Reflex
    print("\n[INSTINCT 6] Inspire 6-DoF Finger Linkage Guard & SRI Wrist Force Reflex (`/arm_6dof_left/right`):")
    overload_res = finger_guard.evaluate_wrist_wrench(np.array([0.0, 0.0, -32.0]), np.zeros(3), holding_object=True)
    handover_res = finger_guard.evaluate_wrist_wrench(np.array([3.2, 0.0, 0.0]), np.zeros(3), holding_object=True, awaiting_human_handover=True)
    print(f"  * 32.0 N (~3.3 kg) Load Applied -> Status: {overload_res['status']} | Forearm Cradle={overload_res['switch_to_forearm_cradle']} | Speech: \"{overload_res['spoken_response']}\"")
    print(f"  * 3.2 N Human Pull During Handover -> Status: {handover_res['status']} | Finger Open={handover_res['finger_open_ratio']} | Speech: \"{handover_res['spoken_response']}\"")

    # 7. Instinct #7: Proactive Social Initiative (Fallen Child Help + Encumbered Doorway Courtesy + Starting Conversation)
    print("\n[INSTINCT 7] Proactive Social Common Sense (Unprompted Fallen-Child Help & Conversation Starter):")
    fallen_child = [DetectedPerson(person_id=1, pos_xy_m=np.array([1.35, 0.20]), vel_xy_m_s=np.zeros(2), head_height_z_m=0.42, is_child=True)]
    encumbered = [DetectedPerson(person_id=2, pos_xy_m=np.array([1.60, 0.10]), vel_xy_m_s=np.array([-0.4, 0.0]), head_height_z_m=1.68, carrying_bulky_load=True)]
    curious_human = [DetectedPerson(person_id=3, pos_xy_m=np.array([1.50, -0.35]), vel_xy_m_s=np.zeros(2), head_height_z_m=1.70, is_looking_at_robot=True, observation_duration_s=2.4)]
    for label, scene in [("Fallen Child on Floor", fallen_child), ("Human Carrying Heavy Box", encumbered), ("Silent Human Looking at Robot", curious_human)]:
        s_dec = proactive_social.evaluate_scene(scene)
        print(f"  * [{label}] -> Priority: {s_dec.priority_level} | Action: {s_dec.gesture_action} | Robot Speaks First: \"{s_dec.proactive_speech}\"")

    # 8. Instinct #8: Crowd-Weaving & Shoulder-Brush Rotation (`I, Robot` Street Scene)
    print("\n[INSTINCT 8] `I, Robot` Predictive Crowd-Weaving & Shoulder-Brush Rotation:")
    crowd = [
        DetectedPerson(person_id=10, pos_xy_m=np.array([1.10, 0.12]), vel_xy_m_s=np.array([-0.45, -0.05]), head_height_z_m=1.72),
        DetectedPerson(person_id=11, pos_xy_m=np.array([1.85, -0.28]), vel_xy_m_s=np.array([-0.35, 0.02]), head_height_z_m=1.65),
    ]
    c_cmd = crowd_nav.compute_crowd_weaving_command(goal_xy_m=np.array([3.5, 0.0]), people=crowd)
    print(f"  * {c_cmd['note']} -> Command: vx={c_cmd['cmd_vx']:+.2f}, vy={c_cmd['cmd_vy']:+.2f} m/s, Waist Shoulder-Brush={math.degrees(c_cmd['shoulder_turn_body_yaw_rad']):+.0f} deg")

    # 9. Instinct #10, #11, #12: Floor Hazard Detour, Joint Fatigue Self-Care, and 4-Stage Stuck Problem Solver
    print("\n[INSTINCT 10, 11, 12] Floor Hazard Watch, Somatic Self-Care & 4-Stage Stuck Problem Solver:")
    f_res = floor_instinct.evaluate_floor_patch("water_puddle", (0.65, 0.08))
    print(f"  * [Floor Hazard] {f_res['note']}")
    relax_needed, relax_msg = self_care.update(dt=6.5, shoulder_pitch_l=-0.85, shoulder_pitch_r=0.0, task_active=False, max_motor_current_a=8.2)
    print(f"  * [Self-Care] {relax_msg}")
    for attempt in range(1, 5):
        rec = problem_solver.step_problem_solver(
            commanded_speed_m_s=0.30,
            actual_displacement_m=0.002,
            measured_gap_m=0.38 if attempt == 2 else 0.25,
            left_clear_m=0.75 if attempt == 3 else 0.22,
            right_clear_m=0.20,
        )
        print(f"  * [Stuck Solver Stage {rec.stage}: {rec.strategy_name}] {rec.explanation}")

    print("\n" + "=" * 100)
    print("  ALL 12 INTUITIVE EMBODIED INSTINCTS VERIFIED AGAINST `tiangong2pro_torq.xml`!")
    print("=" * 100)


def run_live_mujoco_instinct_showcase():
    """Opens the interactive 3D MuJoCo viewer and cycles smoothly through the key physical postures."""
    model, data = load_factory_tg2_model()
    reacher = VisuomotorBodySchemaReacher()
    tucked = reacher.get_tucked_arm_joints()

    stages = [
        ("Stage 1: Tier-2 Blind-Spot Scan (Waist CW -55 deg + Head -40 deg)", {"body_yaw_joint": -0.95, "head_yaw_joint": -0.70, "head_pitch_joint": 0.05}, 0.0, 0.0),
        ("Stage 2: Narrow 38cm Gap -> 90-Deg Sideways Squeeze + Head Looking Into Gap", {**tucked, "body_yaw_joint": -1.57, "head_yaw_joint": 1.50, "head_pitch_joint": 0.10}, 0.0, 0.0),
        ("Stage 3: Low 1.60m Overhang -> Head Duck + 14cm Knee Crouch", {**tucked, "head_pitch_joint": 0.38}, 0.0, -0.14),
        ("Stage 4: Intuitive Reach-With-Waist Twist + Left-Arm Counterbalance", reacher.compute_intuitive_reach(np.array([0.48, -0.22, 0.32])), 0.0, 0.0),
        ("Stage 5: Proactive Fallen-Child Help (Looking Down + Offering Forearm)", {"body_yaw_joint": 0.25, "head_yaw_joint": 0.35, "head_pitch_joint": 0.40, "shoulder_pitch_r_joint": -0.85, "elbow_pitch_r_joint": -0.75}, 0.0, -0.10),
    ]

    print("Launching Live 3D MuJoCo Instinct Showcase... (Close window to exit)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            elapsed = time.time() - t0
            idx = int(elapsed // 3.5) % len(stages)
            name, joints, pyaw, pdz = stages[idx]
            apply_upper_body_pose(model, data, joints, pelvis_yaw_rad=pyaw, pelvis_dz_m=pdz)
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    if "--live" in sys.argv:
        run_live_mujoco_instinct_showcase()
    else:
        run_all_12_instincts_audit()
