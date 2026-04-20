"""
G1 standalone controller using the MuJoCo Menagerie model.

- Position actuators (kp=500, critically damped) — ctrl[i] = desired angle (rad)
- Starts from 'stand' keyframe — no elastic band needed
- No DDS/bridge — direct mj_data.ctrl control in the viewer loop

Usage:
  python3 g1_menagerie.py
  python3 g1_menagerie.py wave
  python3 g1_menagerie.py celebrate
  python3 g1_menagerie.py walk

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

# ── Walk poses ──────────────────────────────────────────────────────────────
# Base walk stance: slight forward lean + bent knees to lower CoM
WALK_BASE = STAND.copy()
WALK_BASE[0]  = -0.10   # L hip pitch — forward lean
WALK_BASE[6]  = -0.10   # R hip pitch — forward lean
WALK_BASE[3]  =  0.20   # L knee bent
WALK_BASE[9]  =  0.20   # R knee bent
WALK_BASE[4]  = -0.10   # L ankle compensate
WALK_BASE[10] = -0.10   # R ankle compensate

# Shift weight onto left leg
WEIGHT_LEFT = WALK_BASE.copy()
WEIGHT_LEFT[1]  =  0.10   # L hip roll — adduct
WEIGHT_LEFT[7]  = -0.05   # R hip roll — slight abduct

# Right leg lifted
R_LIFT = WEIGHT_LEFT.copy()
R_LIFT[6]  = -0.20   # R hip pitch — swing forward
R_LIFT[9]  =  0.35   # R knee — lift foot
R_LIFT[10] =  0.10   # R ankle — flex up

# Right foot planted forward
R_PLANT = WEIGHT_LEFT.copy()
R_PLANT[6]  = -0.20   # R hip pitch — step forward
R_PLANT[9]  =  0.20   # R knee — soft landing
R_PLANT[10] = -0.08   # R ankle — heel down

# Shift weight onto right leg
WEIGHT_RIGHT = WALK_BASE.copy()
WEIGHT_RIGHT[7]  = -0.10  # R hip roll — adduct
WEIGHT_RIGHT[1]  =  0.05  # L hip roll — slight abduct
WEIGHT_RIGHT[6]  = -0.20  # keep R leg forward

# Left leg lifted
L_LIFT = WEIGHT_RIGHT.copy()
L_LIFT[0]  = -0.20   # L hip pitch — swing forward
L_LIFT[3]  =  0.35   # L knee — lift foot
L_LIFT[4]  =  0.10   # L ankle — flex up

# Left foot planted
L_PLANT = WALK_BASE.copy()
L_PLANT[0]  = -0.20   # L hip pitch — step forward
L_PLANT[6]  = -0.20   # R hip pitch — stay forward
L_PLANT[3]  =  0.20   # L knee — soft landing
L_PLANT[4]  = -0.08   # L ankle — heel down

# ── Celebrate pose — both arms raised straight up ───────────────────────────
CELEBRATE = STAND.copy()
CELEBRATE[15] = -1.40   # L shoulder pitch — arm up
CELEBRATE[16] = -0.20   # L shoulder roll  — slightly out
CELEBRATE[18] =  0.30   # L elbow — nearly straight
CELEBRATE[22] = -1.40   # R shoulder pitch — arm up
CELEBRATE[23] =  0.20   # R shoulder roll  — slightly out
CELEBRATE[25] =  0.30   # R elbow — nearly straight


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

    T_HOLD      = 2.0
    T_RAISE     = 4.0
    T_WAVE      = 8.0
    T_LOWER     = 4.0
    T_CELEBRATE = 3.0   # hold arms up

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

            elif mode == "walk":
                # Each phase duration
                S = 2.0   # shift weight
                L = 2.0   # lift + swing
                P = 1.5   # plant
                # Cumulative phase boundaries
                p = [0,
                     S,           # weight left
                     S+L,         # R lift+swing
                     S+L+P,       # R plant
                     S+L+P+S,     # weight right
                     S+L+P+S+L,   # L lift+swing
                     S+L+P+S+L+P, # L plant
                     S+L+P+S+L+P+S]  # return to stand
                tc = t % p[-1]   # loop
                if   tc < p[1]: target = interp(WALK_BASE,    WEIGHT_LEFT,  (tc-p[0])/S)
                elif tc < p[2]: target = interp(WEIGHT_LEFT,  R_LIFT,       (tc-p[1])/L)
                elif tc < p[3]: target = interp(R_LIFT,       R_PLANT,      (tc-p[2])/P)
                elif tc < p[4]: target = interp(R_PLANT,      WEIGHT_RIGHT, (tc-p[3])/S)
                elif tc < p[5]: target = interp(WEIGHT_RIGHT, L_LIFT,       (tc-p[4])/L)
                elif tc < p[6]: target = interp(L_LIFT,       L_PLANT,      (tc-p[5])/P)
                else:           target = interp(L_PLANT,      WALK_BASE,    (tc-p[6])/S)

            elif mode == "celebrate":
                T1 = T_HOLD
                T2 = T1 + T_RAISE
                T3 = T2 + T_CELEBRATE
                T4 = T3 + T_LOWER
                if t < T1:
                    target = STAND
                elif t < T2:
                    target = interp(STAND, CELEBRATE, (t - T1) / T_RAISE)
                elif t < T3:
                    target = CELEBRATE
                elif t < T4:
                    target = interp(CELEBRATE, STAND, (t - T3) / T_LOWER)
                else:
                    t = T1
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
