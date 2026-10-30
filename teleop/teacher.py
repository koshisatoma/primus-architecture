import json
import os
from pathlib import Path

SKILLS_FILE = Path("data/skills_registry.json")

def load_registry():
    if not SKILLS_FILE.exists():
        return {}
    with open(SKILLS_FILE, 'r') as f:
        return json.load(f)

def save_registry(registry):
    SKILLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SKILLS_FILE, 'w') as f:
        json.dump(registry, f, indent=2)

def check_skill(task_name):
    """Check if skill exists. Returns (exists: bool, message: str)"""
    registry = load_registry()
    task_key = task_name.lower().strip()
    
    if task_key not in registry:
        return False, f"I haven't been taught about '{task_name}'. Please navigate and perform it for me once."
    
    skill_data = registry[task_key]
    if not os.path.exists(skill_data['file_path']):
        return False, f"Skill '{task_name}' data file is missing. Please re-teach me."
    
    return True, f"Executing learned skill: '{task_name}' ({skill_data['steps']} steps recorded)."

def save_skill(task_name, trajectory_file, steps):
    """Save a newly taught skill to registry"""
    registry = load_registry()
    task_key = task_name.lower().strip()
    
    registry[task_key] = {
        "file_path": str(trajectory_file),
        "steps": steps,
        "taught_date": str(Path(trajectory_file).stat().st_mtime)
    }
    save_registry(registry)
    print(f"✅ Skill '{task_name}' saved to memory ({steps} steps)")
