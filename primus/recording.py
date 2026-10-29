"""
Automatic video recording for task teaching sessions.
Records video + saves metadata (task name, date, time, duration) as JSON.
"""
import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import threading

RECORDINGS_DIR = Path("data/recordings")
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


class TaskRecorder:
    def __init__(self):
        self.is_recording = False
        self.writer = None
        self.current_task = None
        self.start_time = None
        self.frame_count = 0
        
    def start(self, task_name, camera_source=0):
        """
        Start recording a teaching session.
        Args:
            task_name: Name of task being taught (e.g., "reach", "grasp")
            camera_source: Camera index (0) or RTSP stream URL
        """
        if self.is_recording:
            print("Already recording!")
            return
        
        self.current_task = task_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_name}_{timestamp}"
        
        self.video_path = RECORDINGS_DIR / f"{filename}.mp4"
        self.metadata_path = RECORDINGS_DIR / f"{filename}.json"
        
        # Open camera
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            print("Failed to open camera")
            return
        
        # Get video properties
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(self.video_path), fourcc, fps, (width, height))
        
        self.is_recording = True
        self.start_time = datetime.now()
        self.frame_count = 0
        
        print(f"🎬 Recording started: {self.video_path}")
        print(f"   Task: {task_name}")
        print(f"   Press 'q' to stop recording")
        
        # Start recording thread
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.start()
    
    def _record_loop(self):
        """Background recording loop."""
        while self.is_recording:
            ret, frame = self.cap.read()
            if ret:
                self.writer.write(frame)
                self.frame_count += 1
            else:
                print("Frame capture failed")
    
    def stop(self):
        """Stop recording and save metadata."""
        if not self.is_recording:
            print("Not recording!")
            return
        
        self.is_recording = False
        self.thread.join()
        
        # Release resources
        self.cap.release()
        self.writer.release()
        
        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Save metadata JSON
        metadata = {
            "task": self.current_task,
            "date": self.start_time.strftime("%Y-%m-%d"),
            "time": self.start_time.strftime("%H:%M:%S"),
            "duration_seconds": duration,
            "frames": self.frame_count,
            "video_file": str(self.video_path),
            "timestamp": self.start_time.isoformat()
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Recording saved: {self.video_path}")
        print(f"   Metadata: {self.metadata_path}")
        print(f"   Duration: {duration:.1f}s, Frames: {self.frame_count}")
        
        return metadata


# Global recorder instance
recorder = TaskRecorder()


def start_recording(task_name):
    """Convenience function to start recording."""
    recorder.start(task_name)


def stop_recording():
    """Convenience function to stop recording."""
    recorder.stop()