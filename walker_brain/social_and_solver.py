#!/usr/bin/env python3
"""
Module 3: `social_and_solver.py` — Proactive Social Initiative, Fallen-Child Help, Crowd Weaving & Stuck Solver
===============================================================================================================
Implements the 3 high-level common-sense instincts you described:
  1. Instinct #7 (`ProactiveCommonSenseEngine`):
     Breaks the "wait passively for a voice command" limitation!
     The robot initiates conversation and physical assistance ON ITS OWN when it sees:
       - A fallen child or adult on the floor (`person_head_z < 0.65 m`) -> Interrupts walk, kneels/approaches,
         offers forearm support, and speaks first!
       - A person carrying a heavy load toward a doorway -> Steps aside (`vy`) and turns sideways to yield right-of-way!
       - A person standing nearby looking at the robot or hesitating -> Tilts head curiously, waves, and starts a conversation!
  2. Instinct #8 (`CrowdSocialNavigator` — The *I, Robot* Crowded-Street Instinct):
     Uses Predictive Velocity-Obstacles + a Social Personal-Space Field to weave smoothly through moving crowds
     while rotating `body_yaw_joint` (shoulder turn) when passing close so it never clips a pedestrian!
  3. Instinct #12 (`AutonomousProblemSolver`):
     When left alone in a difficult or blocked situation (`commanded velocity > 0` but `actual progress == 0`),
     does NOT freeze and wait for a human command—systematically executes a 4-stage physical recovery escalation!
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DetectedPerson:
    """Tracked human from Board 2/3 Orbbec RGB-D + Livox LiDAR."""
    person_id: int
    pos_xy_m: np.ndarray          # [x, y] in robot pelvis frame (m)
    vel_xy_m_s: np.ndarray        # [vx, vy] relative velocity (m/s)
    head_height_z_m: float        # Z height of person's head above floor (m)
    is_looking_at_robot: bool = False
    observation_duration_s: float = 0.0
    carrying_bulky_load: bool = False
    is_child: bool = False


@dataclass
class ProactiveDecision:
    """Output of Instinct #7 (`ProactiveCommonSenseEngine`)."""
    triggered: bool
    priority_level: str          # "EMERGENCY_HELP", "COURTESY_YIELD", "SOCIAL_INITIATE", "NONE"
    interrupt_navigation: bool
    cmd_vx: float
    cmd_vy: float
    cmd_vyaw: float
    target_body_yaw_rad: float
    target_head_yaw_rad: float
    target_head_pitch_rad: float
    target_head_roll_rad: float
    gesture_action: Optional[str]  # e.g. "extend_forearm_help", "friendly_wave", "yield_bow"
    proactive_speech: Optional[str]
    reasoning: str


class ProactiveCommonSenseEngine:
    """
    Instinct #7: Proactive Social Common Sense & Unprompted Conversation Starter.
    Allows the robot to pause and help a fallen child, yield a doorway, or start a conversation
    WITHOUT waiting for a voice command!
    """

    def evaluate_scene(self, people: List[DetectedPerson]) -> ProactiveDecision:
        if not people:
            return ProactiveDecision(
                triggered=False,
                priority_level="NONE",
                interrupt_navigation=False,
                cmd_vx=0.0,
                cmd_vy=0.0,
                cmd_vyaw=0.0,
                target_body_yaw_rad=0.0,
                target_head_yaw_rad=0.0,
                target_head_pitch_rad=0.0,
                target_head_roll_rad=0.0,
                gesture_action=None,
                proactive_speech=None,
                reasoning="No urgent social triggers in view.",
            )

        # 1. HIGHEST PRIORITY: Fallen Child or Adult on the Floor!
        # A standing child's head is ~1.0-1.3m; if head_height_z_m < 0.62m, they have fallen or collapsed!
        for p in people:
            if p.head_height_z_m < 0.62:
                dist = float(np.linalg.norm(p.pos_xy_m))
                az_rad = math.atan2(float(p.pos_xy_m[1]), max(0.1, float(p.pos_xy_m[0])))
                # Approach to a safe 0.85 m helping distance, then stop and crouch/extend forearm
                approach_vx = 0.25 if dist > 0.90 else 0.0
                who = "child" if p.is_child else "person"
                return ProactiveDecision(
                    triggered=True,
                    priority_level="EMERGENCY_HELP",
                    interrupt_navigation=True,
                    cmd_vx=approach_vx,
                    cmd_vy=0.0,
                    cmd_vyaw=float(np.clip(0.8 * az_rad, -0.5, 0.5)),
                    target_body_yaw_rad=float(np.clip(0.35 * az_rad, -0.6, 0.6)),
                    target_head_yaw_rad=float(np.clip(0.65 * az_rad, -1.2, 1.2)),
                    target_head_pitch_rad=0.38,  # Look down at the fallen child/person
                    target_head_roll_rad=0.08,
                    gesture_action="extend_forearm_help",
                    proactive_speech=(
                        "Hey buddy, I saw you fall! Are you okay? Hold onto my forearm if you want help standing up."
                        if p.is_child
                        else "Excuse me, I saw you fall—are you hurt? Please stay still if you're injured, or hold my forearm to steady yourself."
                    ),
                    reasoning=(
                        f"Common-Sense Override: Spotted fallen {who} (head Z={p.head_height_z_m:.2f}m at dist={dist:.2f}m). "
                        "Interrupting current walk without waiting for a command to pause and help!"
                    ),
                )

        # 2. SECOND PRIORITY: Encumbered Human Courtesy (Human carrying heavy box/baby approaching within 2.5m)
        for p in people:
            dist = float(np.linalg.norm(p.pos_xy_m))
            if p.carrying_bulky_load and dist < 2.5:
                # Step aside away from the human's path and turn sideways (`body_yaw_joint = 90 deg`) to free corridor width!
                yield_side = -1.0 if p.pos_xy_m[1] >= 0.0 else 1.0
                return ProactiveDecision(
                    triggered=True,
                    priority_level="COURTESY_YIELD",
                    interrupt_navigation=True,
                    cmd_vx=0.0,
                    cmd_vy=0.18 * yield_side,
                    cmd_vyaw=0.0,
                    target_body_yaw_rad=yield_side * 1.35,  # Turn torso sideways against wall to maximize corridor space
                    target_head_yaw_rad=-yield_side * 1.10,
                    target_head_pitch_rad=0.08,
                    target_head_roll_rad=0.0,
                    gesture_action="tuck_arms_courtesy",
                    proactive_speech="Looks like your hands are full—go right ahead, I'll step aside and make room for you.",
                    reasoning=(
                        f"Person #{p.person_id} is carrying a bulky load at {dist:.2f}m -> "
                        "Stepping aside and rotating torso sideways so they have full corridor clearance."
                    ),
                )

        # 3. THIRD PRIORITY: Proactive Conversation Initiation (Human looking at robot / hesitating nearby > 1.8s)
        for p in people:
            dist = float(np.linalg.norm(p.pos_xy_m))
            if p.is_looking_at_robot and dist < 2.2 and p.observation_duration_s >= 1.8:
                az_rad = math.atan2(float(p.pos_xy_m[1]), max(0.1, float(p.pos_xy_m[0])))
                return ProactiveDecision(
                    triggered=True,
                    priority_level="SOCIAL_INITIATE",
                    interrupt_navigation=False,
                    cmd_vx=0.0,
                    cmd_vy=0.0,
                    cmd_vyaw=0.0,
                    target_body_yaw_rad=float(np.clip(0.45 * az_rad, -0.8, 0.8)),
                    target_head_yaw_rad=float(np.clip(0.55 * az_rad, -1.2, 1.2)),
                    target_head_pitch_rad=0.05,
                    target_head_roll_rad=0.14,  # Friendly head tilt!
                    gesture_action="wave_and_nod",
                    proactive_speech="Hi there! I noticed you looking over—is there anything I can help you find or do today?",
                    reasoning=(
                        f"Person #{p.person_id} has been looking at us at {dist:.2f}m for {p.observation_duration_s:.1f}s -> "
                        "Proactively tilting head, waving, and starting the conversation first!"
                    ),
                )

        return ProactiveDecision(
            triggered=False,
            priority_level="NONE",
            interrupt_navigation=False,
            cmd_vx=0.0,
            cmd_vy=0.0,
            cmd_vyaw=0.0,
            target_body_yaw_rad=0.0,
            target_head_yaw_rad=0.0,
            target_head_pitch_rad=0.0,
            target_head_roll_rad=0.0,
            gesture_action=None,
            proactive_speech=None,
            reasoning="People present are walking normally; no interruption needed.",
        )


class CrowdSocialNavigator:
    """
    Instinct #8: The *I, Robot* Crowded-Street Weaving Instinct.
    Computes collision-free velocity `(vx, vy, vyaw)` + shoulder-brush rotation (`body_yaw_joint`)
    by predicting where each moving pedestrian will be 1.0 second into the future!
    """

    def __init__(self, personal_space_radius_m: float = 0.72):
        self.personal_space_radius_m = personal_space_radius_m

    def compute_crowd_weaving_command(
        self,
        goal_xy_m: np.ndarray,
        people: List[DetectedPerson],
        desired_speed_m_s: float = 0.55,
    ) -> Dict[str, object]:
        to_goal = np.array(goal_xy_m, dtype=float)
        dist_goal = float(np.linalg.norm(to_goal))
        if dist_goal < 0.10:
            return {
                "cmd_vx": 0.0,
                "cmd_vy": 0.0,
                "cmd_vyaw": 0.0,
                "shoulder_turn_body_yaw_rad": 0.0,
                "min_predicted_clearance_m": 9.9,
                "note": "Arrived at goal.",
            }

        v_attr = (to_goal / dist_goal) * desired_speed_m_s
        v_rep = np.zeros(2, dtype=float)
        min_pred_dist = 9.9
        closest_person_y = 0.0

        for p in people:
            # Predict pedestrian position 1.0 s ahead using their velocity vector!
            pred_pos = p.pos_xy_m + 1.0 * p.vel_xy_m_s
            d_now = float(np.linalg.norm(p.pos_xy_m))
            d_pred = float(np.linalg.norm(pred_pos))
            d_eff = min(d_now, d_pred)

            if d_eff < min_pred_dist:
                min_pred_dist = d_eff
                closest_person_y = float(p.pos_xy_m[1])

            if d_eff < self.personal_space_radius_m * 1.8:
                # Repulsive vector away from predicted future pedestrian position, biased to pass BEHIND them!
                away = -pred_pos / max(0.15, d_pred)
                # Tangential slip vector to weave smoothly left/right instead of stopping dead
                tangent = np.array([-away[1], away[0]]) if pred_pos[1] >= 0.0 else np.array([away[1], -away[0]])
                strength = 0.48 * ((self.personal_space_radius_m * 1.8 - d_eff) / (self.personal_space_radius_m * 1.8)) ** 2
                v_rep += strength * (0.55 * away + 0.85 * tangent)

        v_total = v_attr + v_rep
        speed = float(np.linalg.norm(v_total))
        if speed > desired_speed_m_s:
            v_total = (v_total / speed) * desired_speed_m_s

        # Instinctive Shoulder-Brush Rotation:
        # When passing within 0.85 m of someone on our left/right, rotate `body_yaw_joint` slightly so our shoulder
        # blades rotate away from them, just like a human weaving through a busy street!
        shoulder_yaw = 0.0
        if min_pred_dist < 0.85:
            shoulder_yaw = -0.55 if closest_person_y >= 0.0 else 0.55

        return {
            "cmd_vx": float(np.clip(v_total[0], -0.20, 0.60)),
            "cmd_vy": float(np.clip(v_total[1], -0.25, 0.25)),
            "cmd_vyaw": float(np.clip(math.atan2(v_total[1], max(0.15, v_total[0])) * 0.45, -0.50, 0.50)),
            "shoulder_turn_body_yaw_rad": float(shoulder_yaw),
            "min_predicted_clearance_m": float(min_pred_dist),
            "note": (
                f"Weaving around {len(people)} moving pedestrians (min clearance={min_pred_dist:.2f}m, "
                f"shoulder rotation={math.degrees(shoulder_yaw):+.0f} deg)."
            ),
        }


@dataclass
class RecoveryPlan:
    """Output of Instinct #12 (`AutonomousProblemSolver`)."""
    stage: int
    strategy_name: str
    cmd_vx: float
    cmd_vy: float
    cmd_vyaw: float
    body_yaw_rad: float
    head_scan_deg: float
    tuck_arms: bool
    spoken_request: Optional[str]
    explanation: str


class AutonomousProblemSolver:
    """
    Instinct #12: Autonomous Difficult-Situation & Stuck Recovery Engine.
    When the robot is left alone and encounters a blockage (`commanded motion > 0` but `actual displacement ~ 0`),
    it does NOT wait passively for a command—it escalates through 4 physical reasoning strategies!
    """

    def __init__(self):
        self.stuck_counter: int = 0

    def step_problem_solver(
        self,
        commanded_speed_m_s: float,
        actual_displacement_m: float,
        measured_gap_m: float = 0.42,
        left_clear_m: float = 0.85,
        right_clear_m: float = 0.30,
    ) -> RecoveryPlan:
        if commanded_speed_m_s > 0.10 and actual_displacement_m < 0.015:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            return RecoveryPlan(
                stage=0,
                strategy_name="NOMINAL_PROGRESS",
                cmd_vx=commanded_speed_m_s,
                cmd_vy=0.0,
                cmd_vyaw=0.0,
                body_yaw_rad=0.0,
                head_scan_deg=0.0,
                tuck_arms=False,
                spoken_request=None,
                explanation="Making normal forward progress.",
            )

        # Stage 1: Back up slightly & scan blind spots left/right with Head + Waist
        if self.stuck_counter == 1:
            return RecoveryPlan(
                stage=1,
                strategy_name="BACK_UP_AND_SCAN_BLIND_SPOTS",
                cmd_vx=-0.12,
                cmd_vy=0.0,
                cmd_vyaw=0.0,
                body_yaw_rad=0.45,
                head_scan_deg=65.0,
                tuck_arms=True,
                spoken_request=None,
                explanation="Attempt 1: Forward progress blocked -> Backing up 12 cm and scanning left/right blind spots with head + waist.",
            )

        # Stage 2: Try 90-Degree Sideways Body-Schema Squeeze if gap >= 0.29 m
        if self.stuck_counter == 2 and measured_gap_m >= 0.29:
            return RecoveryPlan(
                stage=2,
                strategy_name="SIDEWAYS_TORSION_SQUEEZE",
                cmd_vx=0.0,
                cmd_vy=0.14,
                cmd_vyaw=0.0,
                body_yaw_rad=-1.57,
                head_scan_deg=90.0,
                tuck_arms=True,
                spoken_request=None,
                explanation=(
                    f"Attempt 2: Gap ({measured_gap_m:.2f}m) blocked our 0.58m shoulders, "
                    "so rotating 90 deg sideways (0.23m depth) and sidestepping through!"
                ),
            )

        # Stage 3: Detour Toward Wider Open Side (Left vs. Right)
        if self.stuck_counter <= 3 and max(left_clear_m, right_clear_m) >= 0.55:
            go_left = left_clear_m >= right_clear_m
            sign = 1.0 if go_left else -1.0
            return RecoveryPlan(
                stage=3,
                strategy_name="AUTONOMOUS_FLANK_DETOUR",
                cmd_vx=0.15,
                cmd_vy=0.18 * sign,
                cmd_vyaw=0.35 * sign,
                body_yaw_rad=0.30 * sign,
                head_scan_deg=45.0 * sign,
                tuck_arms=True,
                spoken_request=None,
                explanation=(
                    f"Attempt 3: Center blocked -> Rerouting toward wider {'left' if go_left else 'right'} "
                    f"opening ({max(left_clear_m, right_clear_m):.2f}m clear)."
                ),
            )

        # Stage 4: Completely Boxed In -> Safe Halt & Proactive Human Collaboration Request
        return RecoveryPlan(
            stage=4,
            strategy_name="REQUEST_HUMAN_ASSISTANCE",
            cmd_vx=0.0,
            cmd_vy=0.0,
            cmd_vyaw=0.0,
            body_yaw_rad=0.0,
            head_scan_deg=35.0,
            tuck_arms=True,
            spoken_request="Excuse me—my path is completely blocked on all sides and the gap is under 28 centimeters. Could someone move this obstacle slightly?",
            explanation="Attempt 4: All physical bypass geometries exhausted (<0.29m everywhere) -> Safely halting and asking nearby human for assistance.",
        )
