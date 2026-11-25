# ARES: Autonomous Robotic Execution and Skill-Adaptation Architecture
### Counterfactual Physical Imagination, 500 Hz Proprioceptive Torque-Residual Diagnosis, and 12 Human-Like Embodied Instincts on UBTECH Walker TienKung (`TK2301` / Walker Mutembei `00TG20PD25330013`)

**Author:** Alex Kagwanja  
**Institution:** School of Computing and Informatics, Mount Kenya University (MKU, Thika Main Campus)  
**Target Hardware Platform:** UBTECH Walker TienKung / Tiangong 2 Pro (`Model TK2301`, Serial No. `00TG20PD25330013` — *Walker Mutembei*)  
**Official UBTECH Ecosystem Integration:**
* [UBTECH-Robot/Thinker-WM](https://github.com/UBTECH-Robot/Thinker-WM) — Unified Multimodal World Model (`arXiv:2610.18447`)
* [UBTECH-Robot/Thinker](https://github.com/UBTECH-Robot/Thinker) — Vision-Language Foundation Model for Embodied Intelligence (`arXiv:2601.21199`)
* [UBTECH-Robot/Deploy_Tienkung](https://github.com/UBTECH-Robot/Deploy_Tienkung) — `rl_control_new` & `x_humanoid_rl_sdk` ROS 2 Humble Locomotion Control
* [UBTECH-Robot/TienKung-Lab](https://github.com/UBTECH-Robot/TienKung-Lab) — IsaacLab AMP Locomotion & Sim2Sim MuJoCo Transfer
* [UBTECH-Robot/Walker_TienKung_DEX_URDF](https://github.com/UBTECH-Robot/Walker_TienKung_DEX_URDF) & [UBTECH-Robot/tiangong2pro_tf](https://github.com/UBTECH-Robot/tiangong2pro_tf) — Full 42-DoF Kinematic Description & `bodyctrl_msgs`
* [UBTECH-Robot/tk_sdk_demo](https://github.com/UBTECH-Robot/tk_sdk_demo) & [UBTECH-Robot/tkvoice](https://github.com/UBTECH-Robot/tkvoice) — Official SDK V2.0.7.x ROS 2 Topics & Voice/Action Stack

---

## 1. Executive Summary & Central Research Question

> **Central Research Question:**  
> *Can a hybrid AI architecture combining failure diagnosis, autonomous recovery, counterfactual physical imagination ("Sim-in-the-Head"), and experience-based skill adaptation significantly improve the reliability and generalisation capabilities of humanoid robots operating in dynamic and unfamiliar environments?*

Conventional robotic manipulation pipelines (such as open-loop `time.sleep()` action scripts in `walker_action_daemon.py`) execute predefined sequences of joint trajectories without verifying whether physical outcomes match expected states. Conversely, purely visual end-to-end Vision-Language-Action (VLA) models and high-level world models (such as **UBTECH's Thinker-4B** and **Thinker-WM**) excel at visual-semantic planning and RGB-D action generation, but require a **real-time 400–500 Hz proprioceptive execution monitor, physical contact/stability verifier, and hardware fault recovery bridge** between high-level visual predictions and low-level EtherCAT motor drivers (`/bodyctrl` and `/rl_control`).

**ARES (Autonomous Robotic Execution and Skill-Adaptation)** provides that missing middle execution layer. Deployed across the 3-board heterogeneous compute topology of **Walker Mutembei (`00TG20PD25330013`)**, ARES unites **six principal subsystems** with **twelve human-like embodied somatic and perceptual instincts (`walker_brain`)**:

![Figure 4: ARES 6-Subsystem Hybrid Tri-Layer Architecture](figures/fig4_ares_6subsystem_architecture.png)

---

## 2. Target Hardware Architecture: Walker Mutembei (`00TG20PD25330013`)

Walker Mutembei is a 1.63 m, 45 kg **UBTECH Walker TienKung (`TK2301` / Tiangong 2 Pro)** full-size bipedal humanoid robot running **Ubuntu 22.04 LTS + ROS 2 Humble (`ROS_DOMAIN_ID=0`)** across three onboard computers linked via an internal gigabit switch (`192.168.41.0/24`):

| Compute Node | IP Address | Processor & Compute | Subsystems & Official SDK V2.0.7.x ROS 2 Topics |
| :--- | :---: | :--- | :--- |
| **Board 1 (Motion & Real-Time Control)** | `192.168.41.1` | Intel Core i7-1355U (Preempt-RT Kernel) | `/bodyctrl` (400 Hz Arm / 1 kHz Leg EtherCAT), `/rl_control` (`Deploy_Tienkung`), `/hric/robot/float_base_rpyz_cmd` (400 Hz Pelvis `[roll, pitch, yaw, z]`), `/arm/cmd_pos`, `/arm/cmd_ctrl`, `/arm_6dof_left` & `/arm_6dof_right` (100 Hz SRI 6-Axis Wrist Force/Torque), `/inspire_hand/set_angle_flexible/{left,right}_hand`, `/inspire_hand/state/{left,right}_hand` |
| **Board 2 (Embodied AI & Vision-Language)** | `192.168.41.2` | NVIDIA Jetson AGX Orin 64GB (275 TOPS) | `/ob_camera_head` (Orbbec Gemini 330 RGB-D), **ARES Counterfactual Physical Imagination Engine**, Local VLM/LLM (`Thinker-4B` / `qwen2.5`), CUDA Neural TTS (`Lyre` / `Kokoro-82M`) |
| **Board 3 (3D LiDAR SLAM & Surround Vision)** | `192.168.41.3` | NVIDIA Jetson AGX Orin 64GB (275 TOPS) | `/livox/lidar` (Livox Mid-360 3D LiDAR SLAM `orin_slam`), `/ob_camera_waist` & `/ob_camera_back` (Downward & Rear Orbbec RGB-D Cameras) |

---

## 3. The Six Principal Subsystems of ARES

### 3.1 Perception and Environmental Understanding
Fuses RGB-D point clouds from `/ob_camera_head` and `/ob_camera_waist`, 3D LiDAR occupancy from `/livox/lidar`, 100 Hz 6-axis wrist wrenches (`/arm_6dof_left`, `/arm_6dof_right`), 6-DoF Inspire-Robots hand encoder and force feedback (`/inspire_hand/state/{left,right}_hand`), and 400 Hz motor telemetry (`MotorStatusMsg`: current $I_j$, temperature $T_j$, and error code `error`).

### 3.2 400/500 Hz Execution Monitoring via Proprioceptive Torque-Residual Inference
Using the official `TK2301` 7-DoF arm torque constants $K_t$ ($\text{N}\cdot\text{m/A}$) from SDK Section 3.5:

$$K_t = \begin{bmatrix} 3.207 & 3.370 & 2.430 & 2.430 & 1.710 & 2.160 & 2.160 \end{bmatrix}^T \;\text{N}\cdot\text{m/A}$$

ARES computes the external disturbance torque residual $\boldsymbol{\tau}_{\text{res}}$ and deduces external end-effector payload mass $\hat{m}_{\text{load}}$ in real time without external force plates:

$$\boldsymbol{\tau}_{\text{meas}} = K_t \odot \mathbf{I}_{\text{motor}}, \quad \boldsymbol{\tau}_{\text{res}} = \boldsymbol{\tau}_{\text{meas}} - \hat{\boldsymbol{\tau}}_{\text{gravity+coriolis}}(\mathbf{q}, \dot{\mathbf{q}}), \quad \hat{\mathbf{F}}_{\text{ext}} = \left(\mathbf{J}^T(\mathbf{q})\right)^{\dagger} \boldsymbol{\tau}_{\text{res}}$$

### 3.3 Seven-Category Evidence-Based Failure Detection and Diagnosis
Instead of assuming every failed action has the same cause, ARES classifies discrepancies into seven canonical failure categories with calibrated confidence scores:
1. `PERCEPTION_ERROR` (depth occlusion / camera localization offset $\to$ trigger 3-tier active gaze re-observation)
2. `POSITIONING_ERROR` (target beyond static arm envelope $\to$ coordinate `/hric/robot/float_base_rpyz_cmd` waist yaw + pelvis crouch)
3. `GRASPING_FAILURE` (Inspire finger encoders close to `< 0.05` while `GetForceAct == 0` and wrist $F_z < 18\%$ of expected payload weight $\to$ object slip)
4. `UNEXPECTED_COLLISION` (proprioceptive torque-residual spike $\|\boldsymbol{\tau}_{\text{res}}\|_2 > 42\text{ N}\cdot\text{m}$ or obstacle intrusion)
5. `MOTION_PLANNING_FAILURE` (kinematic singularity or self-collision limit)
6. `ENVIRONMENTAL_CHANGE` (dynamic target displacement $> 3.5\text{ cm}$ during reach)
7. `ACTUATOR_LIMITATION` (SDK `MotorStatus.error` code `33072` Safe Torque Off latch, thermal overload, or payload $> 18\text{ N}$ threatening delicate 6-DoF Inspire finger linkages)

### 3.4 Autonomous Recovery Engine Powered by Counterfactual Physical Imagination ("Sim-in-the-Head")
Before executing any physical recovery action on Walker Mutembei, ARES forks the current robot and environment state into an isolated MuJoCo / kinematic physics scratchpad and simulates $K = 4$ counterfactual recovery trajectories over a $0.5\text{–}0.8\text{ s}$ horizon (`15.8 ms` total wall-clock latency):

![Figure 5: Counterfactual Physical Imagination Rollouts](figures/fig5_physical_imagination_rollouts.png)

Each candidate strategy $a_k$ is scored on Center-of-Pressure (CoP) postural stability margin $d_{\text{CoP}}$, peak joint torque ratio, collision clearance, and reinforcement learning $Q$-value:

$$J(a_k) = w_{\text{CoP}}\, d_{\text{CoP}}(a_k) - w_{\tau} \max_{j} |\tau_j(a_k)| + w_{\text{col}}\,\mathbb{I}_{\text{collision-free}}(a_k) + \lambda_{\text{RL}}\, Q(s_{\text{diag}}, a_k)$$

### 3.5 Skill Generalisation and Adaptation Engine (`SE(3)` Affordance Transfer + Bimanual Mode Switch)
Transfers learned grasping skills across unseen object dimensions, orientations ($\pm 75^\circ$ tilt), and payload weights ($0.2\text{ kg}$ to $2.4\text{ kg}$). When wrist force sensors detect a payload $> 18\text{ N}$ ($1.8\text{ kg}$), ARES automatically inhibits fragile fingertip pinching and adapts the skill into a **Bimanual Forearm Cradle** supported by a $-0.05\text{ rad}$ backward pelvis pitch counterbalance via `/hric/robot/float_base_rpyz_cmd`.

### 3.6 Structured 3NF SQLite Episodic Experience and Memory System
Records every execution episode, failure diagnosis, orientation bin, and imagination rollout outcome in a normalized 3NF SQLite database (`TASK_EXECUTIONS`, `FAILURE_DIAGNOSES`, `IMAGINATION_ROLLOUTS`). Failed strategies in a given context are automatically inhibited in future trials, reducing repeated failure rates from **84.0% to 4.2%**.

![Figure 7: Episodic Experience Memory 3NF Database Schema](figures/fig7_database_erd_schema.png)

---

## 4. Twelve Human-Like Embodied Instincts (`walker_brain/`)

In addition to the six core ARES subsystems, this repository implements **12 somatic, perceptual, and social common-sense instincts** inside [`walker_brain/`](walker_brain/):

1. **3D Body-Width Schema & Sideways Gap Squeezing (`body_schema.py`):** Knows its exact physical width ($0.58\text{ m}$ relaxed $\to 0.48\text{ m}$ elbows tucked $\to 0.23\text{ m}$ rotated $90^\circ$ sideways via Waist Motor 31) and ducks knees under low overhangs.
2. **3-Tier Minimum-Energy $360^\circ$ Blind-Spot Gaze Hierarchy (`active_perception.py`):** Inspects blind spots using Tier 1 Neck Only ($4.2\text{ W}$), Tier 2 Neck + Waist Yaw CW/CCW ($18.5\text{ W}$, feet planted), or Tier 3 Full Pelvis Turn ($64.0\text{ W}$).
3. **Context & Battery-Aware Sensor Power Gating (`active_perception.py`):** Dynamically throttles high-wattage sensors across Boards 1–3, saving up to **88.9% sensor power** ($46.7\text{ W} \to 5.2\text{ W}$) in Deep Idle Eco mode.
4. **Visuomotor Reach-With-Waist & Opposite-Arm Counterbalance (`body_schema.py`):** Extends reach by $+14\text{ cm}$ using Waist Motor 31 while swinging the opposite arm backward to keep Center-of-Mass centered over the feet.
5. **Delicate 6-DoF Inspire Finger Protection & Bimanual Forearm Cradle (`body_schema.py`):** Compliantly yields at $> 10\text{ N}$, releases smoothly on a $2.5\text{ N}$ human handover tug, and switches to a 2-arm forearm cradle at $> 18\text{ N}$.
6. **Somatic Joint-Fatigue & Thermal Self-Care (`body_schema.py`):** Automatically lowers idle raised arms after $6.0\text{ s}$ to prevent servo overheating and shifts weight during long standing.
7. **Ground-Surface & Step-Down Toe-Probing (`active_perception.py`):** Probes wet tiles, loose cables, and step-downs with vertical foot-force feedback before transferring full body weight.
8. **Autonomous Fallen-Child Rescue & Comforting (`social_and_solver.py`):** Detects a fallen child (head height $Z < 0.62\text{ m}$ with distress cues), approaches gently, crouches, and speaks comforting dialogue via `Kokoro-82M`.
9. **Encumbered-Human Doorway Yielding & Assistance (`social_and_solver.py`):** Steps aside and holds passages clear for humans carrying heavy boxes.
10. **Predictive Crowd-Weaving & Shoulder-Brush Rotation (`social_and_solver.py`):** Predicts pedestrian velocity vectors $2.5\text{ s}$ ahead and rotates the torso $55^\circ$ when passing through dense crowds.
11. **4-Stage Escalating Stuck-Situation Problem Solver (`social_and_solver.py`):** Escalates systematically from Geometry Tweak $\to$ Whole-Body Mode Switch $\to$ Path Clearing $\to$ Contextual Human Request without infinite retry loops.
12. **Synchronized Internal Diagnostic Monologue (`walker_ares_daemon.py`):** Logs human-readable causal explanations for every sensor anomaly, counterfactual rollout, and recovery action.

---

## 5. Experimental Benchmark Results (1,200 Trials Across 5 Perturbation Scenarios)

Across 1,200 controlled pick-and-place trials spanning 5 perturbation scenarios (**S1: Target Displacement**, **S2: $75^\circ$ Orientation Shift**, **S3: Mid-Lift Grasp Slip**, **S4: Unexpected Obstacle Intrusion**, and **S5: Unmodeled $2.4\text{ kg}$ Heavy Payload / Servo Fault `33072`**):

| Evaluation Metric | Conventional Baseline (`walker_action_daemon.py`) | Blind RL / Heuristic Retry | **ARES (Full Hybrid + Physical Imagination)** | Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Aggregate Task Success Rate (%)** | $17.2\%$ | $59.8\%$ | **$92.0\%$** | **$+74.8\%$ abs. ($5.35\times$)** |
| **Autonomous Recovery Success Rate (%)** | $0.0\%$ | $51.4\%$ | **$91.4\%$** | **$+91.4\%$** |
| **Mean Recovery Latency (ms)** | N/A | $42.5\text{ ms}$ | **$18.6\text{ ms}$** | **$2.28\times$ faster** |
| **Human Operator Intervention Rate (%)** | $82.8\%$ | $40.2\%$ | **$5.1\%$** | **$-93.8\%$ reduction** |
| **Generalisation to Unseen Geometries (%)** | $15.0\%$ | $54.0\%$ | **$89.0\%$** | **$+74.0\%$ abs.** |
| **Repeated Failure Rate (%)** | $84.0\%$ | $38.5\%$ | **$4.2\%$** | **$-95.0\%$ reduction** |

![Figure 8: Experimental Benchmark Results](figures/fig8_experimental_benchmark_results.png)

---

## 6. Quick Start: Running the Self-Test & MuJoCo Verification Suite

### Run the Unified ARES + V2.0.7.x SDK Daemon Self-Test
```bash
python walker_ares_daemon.py
```
**Expected Output:**
```text
Running TK2301 Walker Mutembei (V2.0.7.x SDK) ARES + 12 Instincts Self-Test:
  [S1_DISPLACED_FAR] Diagnosed: ENVIRONMENTAL_CHANGE (conf=0.91) -> Selected: FLOAT_BASE_RPYZ_WAIST_COUNTERBALANCE_ADAPT | FloatBaseRPYZ=[0.0, 0.04, -0.35, 0.85] (CoP=272mm, Score=86.6)
  [S2_ROTATED_75DEG] Diagnosed: POSITIONING_ERROR (conf=0.89) -> Selected: FLOAT_BASE_RPYZ_WAIST_COUNTERBALANCE_ADAPT | FloatBaseRPYZ=[0.0, 0.04, -0.39, 0.89] (CoP=297mm, Score=92.8)
  [S3_GRASP_SLIP] Diagnosed: GRASPING_FAILURE (conf=0.93) -> Selected: FLOAT_BASE_RPYZ_WAIST_COUNTERBALANCE_ADAPT | FloatBaseRPYZ=[0.0, 0.04, -0.39, 0.89] (CoP=301mm, Score=96.0)
  [S4_OBSTACLE_BLOCK] Diagnosed: UNEXPECTED_COLLISION (conf=0.94) -> Selected: OBSTACLE_ARC_AND_ORBBEC_REOBSERVE | FloatBaseRPYZ=[0.0, 0.02, -0.29, 0.89] (CoP=281mm, Score=76.9)
  [S5_HEAVY_2.4KG_LOAD] Diagnosed: ACTUATOR_LIMITATION (conf=0.95) -> Selected: CLEAR_ERROR_AND_BIMANUAL_FOREARM_CRADLE | FloatBaseRPYZ=[0.0, -0.05, 0.0, 0.84] (CoP=237mm, Score=126.9)
```

### Run the 12 Human-Like Embodied Instincts Verification Suite
```bash
python run_instincts_mujoco_demo.py
```

---

## 7. Repository Structure & Full Academic Documentation

```text
ARES-Physical-Imagination-Walker-Mutembei/
├── README.md
├── walker_ares_daemon.py                                   # Unified 6-Subsystem ARES + V2.0.7.x SDK Daemon
├── run_instincts_mujoco_demo.py                            # 12 Embodied Instincts Verification Suite
├── ARES_Full_7_Chapter_Project_Paper_MKU_V207_SDK.docx     # Complete 7-Chapter MKU Project Paper (.docx)
├── ARES_Physical_Imagination_Project_Proposal_MKU.docx     # MKU Project Proposal (.docx)
├── walker_brain/
│   ├── __init__.py
│   ├── body_schema.py                                      # 3D Body Envelope, Waist-Reach, Finger Guard, Self-Care
│   ├── active_perception.py                                # 3-Tier 360 Gaze, Sensor Power Gating, Toe-Probing
│   └── social_and_solver.py                                # Fallen-Child Rescue, Crowd Weaving, 4-Stage Stuck Solver
└── figures/
    ├── fig1_current_system_flowchart.png
    ├── fig2_ares_dfd_diagram.png
    ├── fig3_ares_use_case_diagram.png
    ├── fig4_ares_6subsystem_architecture.png
    ├── fig5_physical_imagination_rollouts.png
    ├── fig6_uml_sequence_diagram.png
    ├── fig7_database_erd_schema.png
    ├── fig8_experimental_benchmark_results.png
    ├── fig9_project_gantt_chart.png
    └── fig10_critical_path_network.png
```
