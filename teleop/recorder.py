import argparse
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from pynput import keyboard
from teleop.teacher import register_learned_skill

def record_task(task_name, xml_path="sim/demo_arena.xml"):
    print(f"=== RECORDING: {task_name.upper()} ===")
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Get the ID of the 'hand' body for IK
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    if hand_id == -1:
        print("ERROR: 'hand' body not found in XML!")
        return

    # Initialize target position to current hand position
    target_pos = data.xpos[hand_id].copy()
    
    keys_pressed = set()
    is_recording = False
    recorded_qpos = []
    move_speed = 0.05 # How fast the hand moves in 3D space

    def on_press(key):
        try: keys_pressed.add(key.char.lower())
        except AttributeError:
            if key == keyboard.Key.space: keys_pressed.add('space')
            if key == keyboard.Key.enter: keys_pressed.add('enter')

    def on_release(key):
        try: keys_pressed.discard(key.char.lower())
        except AttributeError:
            if key == keyboard.Key.space: keys_pressed.discard('space')
            if key == keyboard.Key.enter: keys_pressed.discard('enter')

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if model.ncam > 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = 0

        print("\n" + "="*50)
        print(" CARTESIAN CONTROL (Like a real robot!)")
        print("="*50)
        print(" W / S : Hand UP / DOWN")
        print(" A / D : Hand LEFT / RIGHT")
        print(" Q / E : Hand FORWARD / BACKWARD")
        print(" SPACE : Start / Stop Recording")
        print(" ENTER : Save Task")
        print("="*50 + "\n")

        last_space = False
        last_enter = False

        while viewer.is_running():
            # 1. Handle Recording
            if 'space' in keys_pressed and not last_space:
                is_recording = not is_recording
                print(f"\n>>> {'RECORDING' if is_recording else 'PAUSED'} <<<")
            last_space = 'space' in keys_pressed

            if 'enter' in keys_pressed and not last_enter:
                if recorded_qpos:
                    save_path = f"data/skills/{task_name}_demo.npz"
                    os.makedirs("data/skills", exist_ok=True)
                    np.savez(save_path, qpos_seq=np.array(recorded_qpos))
                    register_learned_skill(task_name, save_path)
                    print(f"\n>>> SAVED '{task_name}' ({len(recorded_qpos)} steps) <<<")
                else:
                    print("\n>>> WARNING: Nothing to save! <<<")
            last_enter = 'enter' in keys_pressed

            # 2. Update Target Position based on keys
            if 'w' in keys_pressed: target_pos[2] += move_speed # Up
            if 's' in keys_pressed: target_pos[2] -= move_speed # Down
            if 'a' in keys_pressed: target_pos[1] += move_speed # Left (Y-axis)
            if 'd' in keys_pressed: target_pos[1] -= move_speed # Right (Y-axis)
            if 'q' in keys_pressed: target_pos[0] += move_speed # Forward (X-axis)
            if 'e' in keys_pressed: target_pos[0] -= move_speed # Backward (X-axis)

            # 3. Inverse Kinematics (Make the arm reach the target)
            current_hand_pos = data.xpos[hand_id].copy()
            error = target_pos - current_hand_pos
            
            # Calculate Jacobian
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, None, current_hand_pos, hand_id)
            
            # Only use the first 3 joints (the arm)
            jacp_arm = jacp[:, :3]
            
            # Jacobian Transpose IK step
            dq = jacp_arm.T @ error * 2.0 
            
            # Apply movement and clamp to joint limits
            for i in range(3):
                data.qpos[i] += dq[i] * model.opt.timestep * 10
                # Simple clamping to prevent breaking the arm
                if model.jnt_range[i, 0] != model.jnt_range[i, 1]:
                    data.qpos[i] = np.clip(data.qpos[i], model.jnt_range[i, 0], model.jnt_range[i, 1])

            mujoco.mj_forward(model, data)
            
            if is_recording:
                recorded_qpos.append(data.qpos.copy())
                
            viewer.sync()
            time.sleep(model.opt.timestep)

    listener.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    record_task(args.task)
