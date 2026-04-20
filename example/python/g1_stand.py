"""
G1 standing controller — no elastic band.

The robot starts on the ground with joints near zero (upright pose).
PD control holds the standing pose. Ankle Kp is high for balance.

Steps:
  1. Run simulator:   python3 unitree_mujoco.py
  2. Run this script: python3 g1_stand.py
  Robot should stand still within ~3 seconds.
"""
import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

NUM_MOTORS = 29

# Higher ankle/knee gains for free-standing balance
Kp = [60, 40, 40, 80, 20, 20,   # L leg
      60, 40, 40, 80, 20, 20,   # R leg
      40, 20, 20,                # waist
      20, 20, 20, 20, 10, 10, 10,
      20, 20, 20, 20, 10, 10, 10]

Kd = [15, 12, 12, 18,  8,  8,   # L leg
      15, 12, 12, 18,  8,  8,   # R leg
      12,  8,  8,
       8,  8,  8,  8,  5,  5,  5,
       8,  8,  8,  8,  5,  5,  5]

# Hold the initial pose the robot was in when the script started
STAND = None  # set after reading init_pose

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
STAND = init_pose.copy()
print("Ready. Holding current pose.")

crc = CRC()
cmd = unitree_hg_msg_dds__LowCmd_()
cmd.mode_pr = 0
cmd.mode_machine = low_state.mode_machine

for i in range(NUM_MOTORS):
    cmd.motor_cmd[i].mode = 1
    cmd.motor_cmd[i].q   = init_pose[i]
    cmd.motor_cmd[i].dq  = 0.0
    cmd.motor_cmd[i].tau = 0.0
    cmd.motor_cmd[i].kp  = Kp[i]
    cmd.motor_cmd[i].kd  = Kd[i]

dt = 0.002

while True:
    step_start = time.perf_counter()

    cmd.mode_machine = low_state.mode_machine
    for i in range(NUM_MOTORS):
        cmd.motor_cmd[i].q = float(STAND[i])

    cmd.crc = crc.Crc(cmd)
    pub.Write(cmd)

    elapsed = time.perf_counter() - step_start
    if dt - elapsed > 0:
        time.sleep(dt - elapsed)
