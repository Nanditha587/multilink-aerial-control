 Adaptive Motion Planning for Multilink Underactuated Systems
### Comparative Analysis of LQR, MPC, and RL for Stabilization Under Morphological Change

> *A simulation study exploring the control fundamentals of articulated aerial robots, with focus on controller performance during mid-flight reconfiguration.*

---

## Abstract

This project investigates the stability and recovery performance of three control paradigms — Linear Quadratic Regulator (LQR), Model Predictive Control (MPC), and Reinforcement Learning (PPO) — applied to a triple-link underactuated system undergoing sudden morphological change. Motivated by the quasi-static assumption limitation acknowledged in recent articulated aerial robot literature, we simulate what happens when link mass and length parameters change instantaneously mid-simulation — violating the assumption that joint velocities can be treated as zero. Results show that MPC, by re-solving the Discrete Algebraic Riccati Equation (DARE) online, recovers in 3.01 seconds with 639 J energy expenditure, while fixed-gain LQR fails to recover and expends 54,128 J. We extend the study to a simplified 2D aerial robot model with gimbal-vectored rotor thrust, directly based on linearised equations from published articulated aerial robot literature. The aerial model achieves full controllability (rank 10/10) with 6-dimensional control (thrust + gimbal angle per link), and demonstrates comparable LQR/MPC performance under hover stabilisation. These findings support the case for online model-adaptive controllers in articulated aerial systems operating beyond quasi-static regimes.

---

## Motivation

Multi-link articulated aerial robots represent a frontier in aerial manipulation — capable of reconfiguring their shape in flight to navigate constrained environments and exert wrenches on objects. A central challenge acknowledged in published work is the reliance on a **quasi-static assumption**: joints are actuated slowly enough that joint velocity is treated as zero, simplifying the multi-body dynamics to a single rigid body with varying inertial parameters.

> *"Developing a new control methodology capable of handling non-approximated, multi-rigid-body dynamics in the air is a major goal for future work."* — Recent articulated aerial robot literature

This project builds a simulation framework to study exactly this: **what happens to controller performance when the quasi-static assumption breaks down?**

---

## Project Structure

```
thesis_multilink_control/
├── src/
│   ├── dynamics.py              # Triple-link pendulum: M(q), C(q,dq), G(q), linearisation
│   ├── lqr_controller.py        # LQR via Continuous ARE (CARE)
│   ├── mpc_controller.py        # MPC via Discrete ARE (DARE), online model update
│   ├── rl_controller.py         # PPO agent (stable-baselines3)
│   ├── pendulum_env.py          # Gymnasium environment for RL training
│   ├── simulate.py              # LQR simulation loop
│   ├── simulate_comparison.py   # LQR vs MPC morphology change comparison
│   ├── aerial_robot_dynamics.py # Aerial robot model with gimbal vectoring
│   └── simulate_aerial_robot.py # Aerial model LQR vs MPC comparison
├── plots/                       # Generated figures
├── animations/                  # Pendulum stabilisation GIF
├── results/                     # Trained RL model
├── main.py                      # Single entry point — runs all three controllers
└── README.md
```

---

## Mathematical Foundation

### Triple-Link Pendulum Dynamics

Derived via Lagrangian mechanics. The equation of motion is:

```
M(q)*ddq + C(q,dq) + G(q) = tau
```

Where:
- **M(q)** — 3x3 configuration-dependent mass/inertia matrix
- **C(q,dq)** — Coriolis and centripetal vector (from Christoffel symbols of M)
- **G(q)** — gravity torque vector
- **tau** — control torques [tau1, tau2, tau3]

State vector: `x = [q1, q2, q3, dq1, dq2, dq3]`

Linearised around upright equilibrium (small angle approximation):

```
dx/dt = A*x + B*u

A = [ 0    I  ]      B = [ 0     ]
    [ M0^-1 Kg  0]        [ M0^-1 ]
```

### Aerial Robot Model

Based on the simplified single rigid body model from articulated aerial robot literature:

```
State (10x1):  [x, z, theta1, theta2, theta3, dx, dz, dtheta1, dtheta2, dtheta3]
Control (6x1): [F1, F2, F3, alpha1, alpha2, alpha3]  <- thrust + gimbal angle per link
```

Equations of motion:

```
Translational:  m_total * ddR = F_total - G
Rotational:     Ii * ddtheta_i = (Fi - mi*g)*(li/2)*cos(thetai+alphai) + Fi*sin(alphai)*(li/2)
```

Hover equilibrium: `Fi* = mi*g`, `alphai* = 0`. Fully controllable: rank = 10/10.

---

## Controllers

### LQR — Linear Quadratic Regulator
Minimises: `J = integral(x'Qx + u'Ru)dt`

Solved via Continuous Algebraic Riccati Equation (CARE). Gain K computed **once** at initial configuration. Does not adapt to morphology changes.

### MPC — Model Predictive Control
Same cost function, solved via Discrete Algebraic Riccati Equation (DARE) at each configuration change. **Re-solves online** — adapts immediately when system parameters change.

### RL — Proximal Policy Optimisation (PPO)
Reward: `r = -(sum(qi^2) + 0.1*sum(dqi^2) + 0.01*sum(taui^2))`

Trained for 200k timesteps. Finding: insufficient for full stabilisation — triple-link instability requires significantly more sample complexity than model-based approaches.

---

## Results

### Pendulum — Morphology Change at t=5s

| Metric | LQR | MPC |
|---|---|---|
| Settling Time (s) | 1.63 | 1.65 |
| Recovery Time after change (s) | **FAILED (>15s)** | **3.01** |
| Max Overshoot (deg) | 91.36 | 20.66 |
| Total Energy (J) | 54,128 | 639.5 |

**LQR fails — MPC recovers in 3 seconds.** Fixed-gain K becomes wrong after morphology change. MPC re-solves DARE with updated A, B matrices and adapts immediately.

### Aerial Robot Model — Morphology Change at t=5s

| Metric | LQR | MPC |
|---|---|---|
| Settling Time (s) | 1.23 | 1.35 |
| Recovery Time (s) | 2.42 | 2.65 |
| Max Tilt (deg) | 11.46 | 11.46 |
| Max Z Drift (m) | 0.153 | 0.153 |
| Total Energy (J) | 0.050 | 0.052 |

Both controllers perform comparably on the aerial model — rotor thrust provides more robust hovering than gravity compensation. The decisive difference emerges under aggressive morphology change (pendulum study).

---

## Key Findings

1. **Fixed-gain LQR is fragile under morphology change** — energy expenditure increases 85x and system fails to recover
2. **Online model updating (MPC/DARE) is necessary** for fast reconfiguration beyond quasi-static regimes
3. **Gimbal vectoring is essential for full controllability** — vertical-thrust-only model has rank 8/10; adding gimbal angles achieves rank 10/10
4. **RL requires model priors for sample efficiency** on unstable underactuated systems

---

## Installation

```bash
git clone https://github.com/Nanditha587/multilink-aerial-control.git
cd multilink-aerial-control
pip install numpy scipy matplotlib stable-baselines3 gymnasium casadi do-mpc
```

---

## Usage

```bash
# Run all three controllers and generate comparison plots
python main.py

# Run pendulum morphology change comparison (LQR vs MPC)
python src/simulate_comparison.py

# Run aerial robot model comparison
python src/simulate_aerial_robot.py

# Train RL agent
python src/rl_controller.py
```

---

## Generated Outputs

| File | Description |
|---|---|
| `plots/state_trajectories.png` | Joint angles and velocities under LQR |
| `plots/control_effort.png` | Control torques under LQR |
| `plots/lqr_vs_mpc_comparison.png` | Side-by-side morphology change comparison |
| `plots/aerial_robot_lqr_vs_mpc.png` | Aerial robot model comparison |
| `plots/all_three_comparison.png` | LQR vs MPC vs RL on same axes |
| `animations/lqr_pendulum.gif` | Pendulum stabilisation animation |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | >=1.24 | Linear algebra, simulation |
| scipy | >=1.10 | CARE/DARE solvers, ODE integration |
| matplotlib | >=3.7 | Plots and animation |
| stable-baselines3 | >=2.0 | PPO training |
| gymnasium | >=0.29 | RL environment |
| casadi | >=3.6 | Symbolic math (do-mpc dependency) |
| do-mpc | >=5.1 | MPC framework (optional) |

---

## References

1. Zhao, M., et al. "Dragon: A Composable Aerial Vehicle System." IEEE Robotics and Automation Letters, 2018.
2. Zhao, M., et al. "Versatile Multilinked Aerial Robot with Tilted Rotors." IEEE Transactions on Robotics, 2022.
3. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
4. Anderson, B.D.O., Moore, J.B. "Optimal Control: Linear Quadratic Methods." Prentice Hall, 1990.

