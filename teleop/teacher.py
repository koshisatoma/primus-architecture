"""Persistent registry for skills demonstrated in the local simulator."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS_FILE = PROJECT_ROOT / "data" / "skills.json"
SKILLS_DIRECTORY = PROJECT_ROOT / "data" / "skills"


def normalize_task(task_name: str) -> str:
    """Return a stable registry key for a task name."""
    task_key = task_name.casefold().strip()
    if not task_key:
        raise ValueError("Task name must not be empty")
    return task_key


def task_filename(task_name: str) -> str:
    """Create a filesystem-safe filename stem from a task name."""
    task_key = normalize_task(task_name)
    filename = re.sub(r"[^a-z0-9]+", "-", task_key).strip("-") or "task"
    suffix = hashlib.sha256(task_key.encode("utf-8")).hexdigest()[:10]
    return f"{filename[:72]}-{suffix}"


def load_skills() -> dict[str, Any]:
    """Load the skill registry, returning an empty registry on first use."""
    if not SKILLS_FILE.exists():
        return {}
    with SKILLS_FILE.open(encoding="utf-8") as skills_file:
        skills = json.load(skills_file)
    if not isinstance(skills, dict):
        raise ValueError(f"Skill registry must contain a JSON object: {SKILLS_FILE}")
    return skills


def save_skills(skills: dict[str, Any]) -> None:
    """Atomically persist the skill registry."""
    SKILLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=SKILLS_FILE.parent,
            prefix=f".{SKILLS_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(skills, temporary_file, indent=2, allow_nan=False)
            temporary_file.write("\n")
        os.replace(temporary_path, SKILLS_FILE)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def learned_skill(task_name: str) -> dict[str, Any] | None:
    """Return a skill only when a human demonstration has been registered."""
    skill = load_skills().get(normalize_task(task_name))
    if isinstance(skill, dict) and skill.get("status") == "learned":
        return skill
    return None


def check_and_respond(task_name: str) -> str:
    """Refuse untaught tasks and acknowledge recorded demonstrations."""
    if learned_skill(task_name) is None:
        return (
            f"I haven't been taught about {task_name}. "
            "Please navigate me to perform it."
        )
    return f"Executing learned skill: {task_name}."


def register_learned_skill(task_name: str, file_path: str | Path) -> None:
    """Register a saved human demonstration without discarding other skills."""
    task_key = normalize_task(task_name)
    trajectory_path = Path(file_path)
    if not trajectory_path.is_absolute():
        trajectory_path = PROJECT_ROOT / trajectory_path
    trajectory_path = trajectory_path.resolve()
    try:
        stored_path = trajectory_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        stored_path = str(trajectory_path)

    skills = load_skills()
    skills[task_key] = {
        "status": "learned",
        "trajectory_file": stored_path,
        "taught_by": "human_demo",
    }
    save_skills(skills)
    print(f"[TEACHER] Skill '{task_name}' successfully learned and saved.")


def resolve_trajectory_path(skill: dict[str, Any]) -> Path:
    """Resolve a registered trajectory path relative to the project root."""
    trajectory_file = skill.get("trajectory_file")
    if not isinstance(trajectory_file, str) or not trajectory_file:
        raise ValueError("Learned skill has no valid trajectory_file")
    path = Path(trajectory_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path
