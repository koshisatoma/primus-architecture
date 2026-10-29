import cv2
import numpy as np
import mujoco
import time
import os
import threading
from teleop.teacher import register_learned_skill

# MODERN MEDIAPIPE IMPORTS
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# Initialize Hand Landmarker
base_options = mp_tasks.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# Note: If 'hand_landmarker.task' is missing, MediaPipe usually downloads it automatically 
# or we can use the legacy bundle. For stability, let's use the legacy bundle if available, 
# but modern API is better. If this fails, we will switch to a simpler tracker.
# Actually, let's use the most robust method for a demo: Legacy API with correct import.

# FALLBACK TO LEGACY IF MODERN FAILS (Common in 0.10.x)
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    USE_LEGACY = True
    print("Using Legacy MediaPipe API")
except Exception as e:
    print("Legacy API failed, trying modern...")
    USE_LEGACY = False
    # Modern API setup would go here, but Legacy is more stable for quick demos.

# Load MuJoCo
model = mujoco.MjModel.from_xml_path("sim/franka_arena.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
renderer = mujoco.Renderer(model, height=600, width=800)

# Gripper joint indices (7 and 8 in the Franka XML)
GRIPPER_JOINTS = [7, 8] 

# State
is_recording = False
recorded_qpos = []
target_pos = np.array([0.0, 0.4, 0.95]) # Initial target (x, y, z)

print("="*60)
print(" PRIMUS MIND-CONTROL MODE")
print("="*60)
print(" OPEN PALM  : Move Robot Arm (Camera is hidden)")
print(" MAKE A FIST: Grasp Object & Start Recording")
print(" CLOSE WINDOW TO QUIT & SAVE")
print("="*60)

# Start Webcam in background
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Camera error!")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        is_fist = False
        hand_landmarks = None

        # PROCESS HAND
        if USE_LEGACY:
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # FIST DETECTION
                wrist = hand_landmarks.landmark[0]
                middle_tip = hand_landmarks.landmark[12]
                distance = np.sqrt((wrist.x - middle_tip.x)**2 + (wrist.y - middle_tip.y)**2)
                if distance < 0.15: 
                    is_fist = True
        else:
            # Modern API processing would go here
            pass

        # 1. Update Target Position from Hand (Only if palm is open)
        if hand_landmarks and not is_fist:
            # Map hand x (0-1) to robot x (-0.4 to 0.4)
            target_pos[0] = (hand_landmarks.landmark[9].x - 0.5) * 0.8 
            # Map hand y (0-1) to robot y (0.2 to 0.6) - depth
            target_pos[1] = 0.2 + hand_landmarks.landmark[9].y * 0.4
            target_pos[2] = 0.95 # Hover height

        # 2. Handle Gripper and Recording based on Fist
        if is_fist:
            data.qpos[GRIPPER_JOINTS[0]] = 0.04
            data.qpos[GRIPPER_JOINTS[1]] = 0.04
            if not is_recording:
                is_recording = True
                print(">>> FIST DETECTED: GRASPING & RECORDING <<<")
        else:
            data.qpos[GRIPPER_JOINTS[0]] = 0.0
            data.qpos[GRIPPER_JOINTS[1]] = 0.0
            if is_recording:
                is_recording = False
                print(">>> PALM OPEN: RELEASING & STOPPING RECORD <<<")

        # 3. Simple IK (Move arm towards target_pos)
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
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

        mujoco.mj_forward(model, data)

        # 4. Record if grasping
        if is_recording:
            recorded_qpos.append(data.qpos.copy())

        # 5. Render ONLY THE SIMULATION
        renderer.update_scene(data, camera="top_cam")
        sim_frame = renderer.render()
        sim_frame = cv2.cvtColor(sim_frame, cv2.COLOR_RGB2BGR)
        
        # Add status text to simulation
        status = "RECORDING & GRASPING" if is_recording else "MOVING (Open Palm)"
        cv2.putText(sim_frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show ONLY the simulation window
        cv2.imshow("PRIMUS Simulation (Hand tracking active in background)", sim_frame)

        # Break if window closed or 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("PRIMUS Simulation (Hand tracking active in background)", cv2.WND_PROP_VISIBLE) < 1:
            break
            
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    renderer.close()
    
    if recorded_qpos:
        save_path = "data/skills/gesture_demo.npz"
        os.makedirs("data/skills", exist_ok=True)
        np.savez(save_path, qpos_seq=np.array(recorded_qpos))
        register_learned_skill("gesture_task", save_path)
        print(f"\n>>> SAVED 'gesture_task' ({len(recorded_qpos)} steps) <<<")
