"""
G1 standalone controller using the MuJoCo Menagerie model.

- Position actuators (kp=500, critically damped) — ctrl[i] = desired angle (rad)
- Starts from 'stand' keyframe — no elastic band needed
- No DDS/bridge — direct mj_data.ctrl control in the viewer loop

Usage:
  python3 g1_menagerie.py
  python3 g1_menagerie.py wave

MuJoCo viewer controls:
  Space  — pause/resume
  Ctrl+R — reset to stand keyframe

Joint index map (29 DOF):
  0-5:   L leg  (hip_p, hip_r, hip_y, knee, ank_p, ank_r)
  6-11:  R leg
  12:    waist_yaw
  13:    waist_roll
  14:    waist_pitch
  15-21: L arm  (sh_p, sh_r, sh_y, elbow, wr_r, wr_p, wr_y)
  22-28: R arm
"""
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "../../unitree_robots/g1_menagerie/scene.xml"
NUM_MOTORS = 29


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def interp(a, b, t):
    return a + smooth(t) * (b - a)


# ── Stand pose (from keyframe) ──────────────────────────────────────────────
STAND = np.zeros(NUM_MOTORS)
STAND[15] =  0.20;  STAND[16] =  0.20   # L shoulder pitch, roll
STAND[18] =  1.28                        # L elbow
STAND[22] =  0.20;  STAND[23] = -0.20   # R shoulder pitch, roll
STAND[25] =  1.28                        # R elbow

# ── Wave pose (right arm raised, left arm counterweight) ────────────────────
WAVE_UP = STAND.copy()
WAVE_UP[22] = -0.60   # R shoulder pitch — arm up
WAVE_UP[23] =  0.20   # R shoulder roll  — arm out
WAVE_UP[25] =  1.00   # R elbow bent
WAVE_UP[15] =  0.00   # L shoulder pitch — slight counterweight

WAVE_A = WAVE_UP.copy(); WAVE_A[28] =  0.4   # R wrist yaw
WAVE_B = WAVE_UP.copy(); WAVE_B[28] = -0.4


def run(mode="stand"):
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_PATH)

    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    # Reset to stand keyframe
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

    dt = model.opt.timestep  # 0.002s in menagerie

    T_HOLD  = 2.0
    T_RAISE = 4.0
    T_WAVE  = 8.0
    T_LOWER = 4.0

    t = 0.0
    paused = False

    def key_cb(key):
        nonlocal paused
        import glfw
        if key == glfw.KEY_SPACE:
            paused = not paused

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        print(f"Mode: {mode}. Space=pause. Robot starts standing.")

        while viewer.is_running():
            if paused:
                viewer.sync()
                time.sleep(0.01)
                continue

            step_start = time.perf_counter()
            t += dt

            if mode == "wave":
                if t < T_HOLD:
                    target = STAND
                elif t < T_HOLD + T_RAISE:
                    target = interp(STAND, WAVE_UP, (t - T_HOLD) / T_RAISE)
                elif t < T_HOLD + T_RAISE + T_WAVE:
                    wt = t - T_HOLD - T_RAISE
                    phase = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * wt)
                    target = interp(WAVE_A, WAVE_B, phase)
                elif t < T_HOLD + T_RAISE + T_WAVE + T_LOWER:
                    target = interp(WAVE_UP, STAND, (t - T_HOLD - T_RAISE - T_WAVE) / T_LOWER)
                else:
                    t = T_HOLD
                    target = STAND
            else:
                target = STAND

            # Position actuators: ctrl[i] = desired angle directly
            data.ctrl[:NUM_MOTORS] = target

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - step_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stand"
    run(mode)
