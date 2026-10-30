import cv2
import numpy as np
import mujoco
import time
import mediapipe as mp

# 1. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 2. Load MuJoCo Franka Arena
model = mujoco.MjModel.from_xml_path("sim/franka_arena.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
renderer = mujoco.Renderer(model, height=600, width=800)

# Gripper joints (7 and 8 in the Franka XML)
GRIPPER_JOINTS = [7, 8]

# 3. State Machine
MODE_IDLE = 0
MODE_TEACH = 1
MODE_EXECUTE = 2

current_mode = MODE_IDLE
memory_trajectory = []
playback_index = 0
target_pos = np.array([0.0, 0.4, 0.95])

print("="*60)
print(" PRIMUS HAND-SYNC TELEOPERATION")
print("="*60)
print(" GESTURES:")
print("  ✋ OPEN PALM   : TELEOP MODE (Robot follows hand & records in memory)")
print("  ✊ FIST         : EXECUTE MODE (Robot replays your last movement)")
print("  ☝️ POINTING    : IDLE MODE (Robot stops and resets)")
print("="*60)
print(" NO DATA SAVED TO DISK. Everything is in-memory.")
print(" Press 'q' in terminal or close window to quit.")
print("="*60)

cap = cv2.VideoCapture(0)

def get_gesture(landmarks):
    wrist = landmarks.landmark[0]
    tips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky tips
    
    extended_count = 0
    for tip_idx in tips:
        tip = landmarks.landmark[tip_idx]
        dist = np.sqrt((wrist.x - tip.x)**2 + (wrist.y - tip.y)**2)
        if dist > 0.2: # Threshold for extended finger
            extended_count += 1
            
    if extended_count == 4:
        return "OPEN_PALM"
    elif extended_count == 0:
        return "FIST"
    elif extended_count == 1:
        return "POINTING"
    else:
        return "NEUTRAL"

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        gesture = "NEUTRAL"
        hand_landmarks = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            gesture = get_gesture(hand_landmarks)
            # Optional: Uncomment next line to see hand skeleton on webcam for debugging
            # mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- STATE MACHINE LOGIC ---
        
        if gesture == "OPEN_PALM":
            if current_mode != MODE_TEACH:
                print(">>> TELEOP MODE: Recording in memory...")
                current_mode = MODE_TEACH
                memory_trajectory = []
                playback_index = 0
            
            # Update target from hand (smooth mapping)
            target_pos[0] = (hand_landmarks.landmark[9].x - 0.5) * 0.8
            target_pos[1] = 0.2 + hand_landmarks.landmark[9].y * 0.4
            target_pos[2] = 0.95
            
            # Record current robot state in memory
            memory_trajectory.append(data.qpos.copy())
            
        elif gesture == "FIST":
            if current_mode != MODE_EXECUTE:
                print(f">>> EXECUTE MODE: Replaying {len(memory_trajectory)} steps...")
                current_mode = MODE_EXECUTE
                playback_index = 0
                
        elif gesture == "POINTING":
            if current_mode != MODE_IDLE:
                print(">>> IDLE MODE: Resetting.")
                current_mode = MODE_IDLE
                memory_trajectory = []
                playback_index = 0
                mujoco.mj_resetData(model, data)
                target_pos = np.array([0.0, 0.4, 0.95])

        # --- ROBOT CONTROL ---
        
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
        
        if current_mode == MODE_TEACH:
            # IK to follow hand
            if hand_id != -1:
                current_pos = data.xpos[hand_id].copy()
                error = target_pos - current_pos
                jacp = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jacp, None, current_pos, hand_id)
                jacp_arm = jacp[:, :7]
                dq = jacp_arm.T @ error * 5.0
                for i in range(7):
                    data.qpos[i] += dq[i] * model.opt.timestep
                    if model.jnt_range[i, 0] != model.jnt_range[i, 1]:
                        data.qpos[i] = np.clip(data.qpos[i], model.jnt_range[i, 0], model.jnt_range[i, 1])
            # Gripper open during teach
            data.qpos[GRIPPER_JOINTS[0]] = 0.0
            data.qpos[GRIPPER_JOINTS[1]] = 0.0
            
        elif current_mode == MODE_EXECUTE:
            # Playback from memory
            if playback_index < len(memory_trajectory):
                data.qpos[:7] = memory_trajectory[playback_index][:7]
                # Close gripper at the end of the trajectory for visual effect
                if playback_index > len(memory_trajectory) * 0.8:
                    data.qpos[GRIPPER_JOINTS[0]] = 0.04
                    data.qpos[GRIPPER_JOINTS[1]] = 0.04
                playback_index += 1
                
        elif current_mode == MODE_IDLE:
            # Gripper open, robot stays at reset pose
            data.qpos[GRIPPER_JOINTS[0]] = 0.0
            data.qpos[GRIPPER_JOINTS[1]] = 0.0

        mujoco.mj_forward(model, data)

        # --- RENDER (Only Simulation) ---
        renderer.update_scene(data, camera="top_cam")
        sim_frame = renderer.render()
        sim_frame = cv2.cvtColor(sim_frame, cv2.COLOR_RGB2BGR)
        
        # Overlay status
        mode_text = {0: "IDLE (Point to Reset)", 1: "TELEOP (Open Palm)", 2: "EXECUTE (Fist)"}
        cv2.putText(sim_frame, f"MODE: {mode_text[current_mode]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(sim_frame, f"GESTURE: {gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("PRIMUS Hand-Sync Teleop", sim_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("PRIMUS Hand-Sync Teleop", cv2.WND_PROP_VISIBLE) < 1:
            break
            
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    renderer.close()
