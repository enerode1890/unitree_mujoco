"""
G1 wave hello — elastic band support required.
Press 9 (band), then 8 (lift) in MuJoCo after running this script.

Terminal 1: cd ~/unitree_mujoco/simulate_python && python3 unitree_mujoco.py
Terminal 2: cd ~/unitree_mujoco/example/python && python3 g1_wave.py
"""
import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

NUM_MOTORS = 29

# Same gains as working g1_stand.py
Kp = [60, 40, 40, 80, 20, 20,
      60, 40, 40, 80, 20, 20,
      40, 20, 20,
      20, 20, 20, 20, 10, 10, 10,
      20, 20, 20, 20, 10, 10, 10]

Kd = [15, 12, 12, 18,  8,  8,
      15, 12, 12, 18,  8,  8,
      12,  8,  8,
       8,  8,  8,  8,  5,  5,  5,
       8,  8,  8,  8,  5,  5,  5]

def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def interp(a, b, t):
    return a + smooth(t) * (b - a)

low_state = None
def state_handler(msg: LowState_):
    global low_state
    low_state = msg

ChannelFactoryInitialize(1, "lo")
pub = ChannelPublisher("rt/lowcmd", LowCmd_)
pub.Init()
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(state_handler, 10)

print("Waiting for robot state...")
while low_state is None:
    time.sleep(0.1)

init_pose = np.array([low_state.motor_state[i].q for i in range(NUM_MOTORS)])
print("Ready. Press 9 (band), then 8 (lift). Wave starts in 10s.")

crc = CRC()
cmd = unitree_hg_msg_dds__LowCmd_()
cmd.mode_pr = 0
cmd.mode_machine = low_state.mode_machine
for i in range(NUM_MOTORS):
    cmd.motor_cmd[i].mode = 1
    cmd.motor_cmd[i].kp   = Kp[i]
    cmd.motor_cmd[i].kd   = Kd[i]
    cmd.motor_cmd[i].dq   = 0.0
    cmd.motor_cmd[i].tau  = 0.0
    cmd.motor_cmd[i].q    = init_pose[i]

# Wave pose — built on top of init_pose, only move right arm
WAVE_UP = init_pose.copy()
WAVE_UP[22] = -0.60   # R_SHOULDER_PITCH — arm raised (~35°)
WAVE_UP[23] =  0.20   # R_SHOULDER_ROLL  — arm out
WAVE_UP[25] =  1.00   # R_ELBOW bent
WAVE_UP[15] = init_pose[15] - 0.20  # L_SHOULDER_PITCH — slight counterweight

WAVE_A = WAVE_UP.copy(); WAVE_A[28] =  0.4  # R_WRIST_YAW
WAVE_B = WAVE_UP.copy(); WAVE_B[28] = -0.4

dt = 0.002
t  = 0.0
T_HOLD  = 10.0  # hold init_pose — time to press 9+8 and stabilize
T_RAISE =  4.0  # raise arm
T_WAVE  =  8.0  # wave
T_LOWER =  4.0  # lower arm

while True:
    step_start = time.perf_counter()
    t += dt
    cmd.mode_machine = low_state.mode_machine

    if t < T_HOLD:
        target = init_pose
    elif t < T_HOLD + T_RAISE:
        target = interp(init_pose, WAVE_UP, (t - T_HOLD) / T_RAISE)
    elif t < T_HOLD + T_RAISE + T_WAVE:
        wt = t - T_HOLD - T_RAISE
        phase = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * wt)
        target = interp(WAVE_A, WAVE_B, phase)
    elif t < T_HOLD + T_RAISE + T_WAVE + T_LOWER:
        target = interp(WAVE_UP, init_pose, (t - T_HOLD - T_RAISE - T_WAVE) / T_LOWER)
    else:
        t = T_HOLD  # loop

    for i in range(NUM_MOTORS):
        cmd.motor_cmd[i].q = float(target[i])

    cmd.crc = crc.Crc(cmd)
    pub.Write(cmd)

    elapsed = time.perf_counter() - step_start
    if dt - elapsed > 0:
        time.sleep(dt - elapsed)
