"""
G1 interactive walking with pretrained RL policy.

Controls (MuJoCo window must be focused):
  UP    — forward
  DOWN  — backward
  LEFT  — turn left
  RIGHT — turn right
  Space — stop

Usage:
  python3 deploy/g1_walk.py
"""
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
import torch

POLICY_PATH   = "pre_train/g1/motion.pt"
XML_PATH      = "g1_rl_gym_assets/scene.xml"

# Config from g1.yaml
simulation_dt      = 0.002
control_decimation = 10
kps = np.array([100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40], dtype=np.float32)
kds = np.array([2, 2, 2, 4, 2, 2,
                2, 2, 2, 4, 2, 2], dtype=np.float32)
default_angles = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)
ang_vel_scale  = 0.25
dof_pos_scale  = 1.0
dof_vel_scale  = 0.05
action_scale   = 0.25
cmd_scale      = np.array([2.0, 2.0, 0.25], dtype=np.float32)
num_actions    = 12
num_obs        = 47

# Fixed speeds assigned to each arrow key
VX_FWD  =  0.5   # m/s forward
VX_BCK  = -0.3   # m/s backward
YAW_L   =  0.6   # rad/s turn left
YAW_R   = -0.6   # rad/s turn right

# Shared command vector [vx, vy, yaw_rate]
cmd      = np.array([0.0, 0.0, 0.0], dtype=np.float32)
cmd_lock = threading.Lock()


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz)
    ])


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def key_callback(key):
    """Arrow keys set velocity directly; Space stops."""
    import mujoco.glfw as glfw_mod
    glfw = glfw_mod.glfw

    with cmd_lock:
        if key == glfw.KEY_SPACE:
            cmd[:] = 0.0
        elif key == glfw.KEY_UP:
            cmd[0] = VX_FWD;  cmd[2] = 0.0
        elif key == glfw.KEY_DOWN:
            cmd[0] = VX_BCK;  cmd[2] = 0.0
        elif key == glfw.KEY_LEFT:
            cmd[2] = YAW_L;   cmd[0] = 0.0
        elif key == glfw.KEY_RIGHT:
            cmd[2] = YAW_R;   cmd[0] = 0.0


def print_status(cmd):
    dirs = []
    if cmd[0] >  0.1: dirs.append("↑ fwd")
    if cmd[0] < -0.1: dirs.append("↓ bck")
    if cmd[2] >  0.1: dirs.append("← left")
    if cmd[2] < -0.1: dirs.append("→ right")
    state = " + ".join(dirs) if dirs else "■ stop"
    print(f"\r  {state:20s}  vx={cmd[0]:+.2f}  yaw={cmd[2]:+.2f}  ", end="", flush=True)


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.abspath(__file__))

    m = mujoco.MjModel.from_xml_path(os.path.join(base, XML_PATH))
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy = torch.jit.load(os.path.join(base, POLICY_PATH))
    policy.eval()

    action         = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs            = np.zeros(num_obs, dtype=np.float32)
    counter        = 0

    print("G1 Interactive Walk")
    print("  ↑ forward  |  ↓ backward  |  ← turn left  |  → turn right  |  Space = stop")
    print()

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()

            with cmd_lock:
                current_cmd = cmd.copy()

            # PD torque
            tau = pd_control(target_dof_pos, d.qpos[7:], kps,
                             np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                qj      = (d.qpos[7:] - default_angles) * dof_pos_scale
                dqj     =  d.qvel[6:] * dof_vel_scale
                quat    =  d.qpos[3:7]
                omega   =  d.qvel[3:6] * ang_vel_scale
                gravity =  get_gravity_orientation(quat)

                period    = 0.8
                phase     = (counter * simulation_dt % period) / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3]                              = omega
                obs[3:6]                             = gravity
                obs[6:9]                             = current_cmd * cmd_scale
                obs[9:9+num_actions]                 = qj
                obs[9+num_actions:9+2*num_actions]   = dqj
                obs[9+2*num_actions:9+3*num_actions] = action
                obs[9+3*num_actions:9+3*num_actions+2] = [sin_phase, cos_phase]

                with torch.no_grad():
                    action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

                print_status(current_cmd)

            viewer.sync()
            elapsed = time.perf_counter() - step_start
            if simulation_dt - elapsed > 0:
                time.sleep(simulation_dt - elapsed)
