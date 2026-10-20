import json
import re
import threading
from pathlib import Path

from primus.social import chat


SKILLS_FILE = Path(__file__).resolve().parent.parent / 'data' / 'skills.json'
_ROS_COMMAND_LOCK = threading.Lock()
_ROS_COMMAND_PUBLISHER = None
TASK_INTENT = re.compile(
    r'\b(?:how\s+(?:do\s+i|to)|make|cook|chop|perform|prepare)\b',
    re.IGNORECASE,
)
INTRODUCTION = re.compile(
    r"\b(?:i am|i'm|im|my name is|call me|this is|these are)\s+"
    r"(?P<first>[a-z][a-z'-]{0,31})"
    r"(?P<rest>(?:(?:,\s*|\s+and\s+)"
    r"(?:(?:i am|i'm|im|this is|that is|that's)\s+)?"
    r"[a-z][a-z'-]{0,31})*)",
    re.IGNORECASE,
)
INTRODUCTION_TAIL = re.compile(
    r"(?:,\s*|\s+and\s+)"
    r"(?:(?:i am|i'm|im|this is|that is|that's)\s+)?"
    r"([a-z][a-z'-]{0,31})",
    re.IGNORECASE,
)
OBJECT_INTRODUCTION = re.compile(
    r"\b(?:this is|that is|that's|call this|this)\s+"
    r"(?:a\s+|an\s+)?([a-z][a-z0-9_-]{0,31})\b",
    re.IGNORECASE,
)
OBJECT_QUESTION = re.compile(
    r"\b(?:what(?:'s| is)|tell me about)\s+(?:a\s+|an\s+)?"
    r"([a-z][a-z0-9_-]{0,31})\b",
    re.IGNORECASE,
)
ACTION_COMMAND = re.compile(
    r'\b(?:find|locate|pick(?:\s+up)?|grab|place|put|sit|stand)\b',
    re.IGNORECASE,
)


def _send_ros_command(command):
    global _ROS_COMMAND_PUBLISHER
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init(args=None)
        with _ROS_COMMAND_LOCK:
            if _ROS_COMMAND_PUBLISHER is None:
                node = Node('primus_brain_bridge')
                publisher = node.create_publisher(String, '/primus/command', 10)
                _ROS_COMMAND_PUBLISHER = (node, publisher)
            node, publisher = _ROS_COMMAND_PUBLISHER
            message = String()
            message.data = command
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0.1)
        return f"Okay, {command}. I'm on it."
    except (ImportError, RuntimeError, OSError) as error:
        return f'I could not send that command to ROS 2: {error}'


def _load():
    if not SKILLS_FILE.exists():
        return {}
    with SKILLS_FILE.open(encoding='utf-8') as skills_file:
        skills = json.load(skills_file)
    if not isinstance(skills, dict):
        raise ValueError(f'{SKILLS_FILE} must contain a JSON object')
    return skills


def _save(skills):
    SKILLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SKILLS_FILE.open('w', encoding='utf-8') as skills_file:
        json.dump(skills, skills_file, indent=2)


def think(text):
    from primus.integration import cognition

    cognitive_state = cognition()
    context = cognitive_state.think_hook(text)
    reply = _think(text, context)
    cognitive_state.respond_hook(reply)
    return reply


def _think(text, cognition_context):
    normalized = text.casefold().strip()
    skills = _load()

    introduction = INTRODUCTION.search(text)
    introduced_names = []
    if introduction and introduction.group('first').casefold() not in {'a', 'an'}:
        names = [introduction.group('first').capitalize()]
        names.extend(
            match.group(1).capitalize()
            for match in INTRODUCTION_TAIL.finditer(introduction.group('rest'))
        )
        match_text = introduction.group(0).casefold()
        if not match_text.startswith(('this is', 'these are')) or len(names) > 1:
            introduced_names = names

    if introduced_names:
        from primus.camera_state import latest_teacher_frame
        from primus.people import start_introduction

        return start_introduction(introduced_names, latest_teacher_frame())

    from primus.people import (
        cancel_introduction,
        confirm_spelling,
        has_pending_introduction,
    )

    if has_pending_introduction():
        try:
            return confirm_spelling(text)
        except (OSError, RuntimeError, ValueError):
            cancel_introduction()
            return "I couldn't save that face right now. Please introduce yourself again."

    if ACTION_COMMAND.search(text):
        return _send_ros_command(text.strip())

    object_question = OBJECT_QUESTION.search(text)
    if object_question:
        from primus.objects import get_object

        name = object_question.group(1)
        record = get_object(name)
        if record is not None:
            teacher = (
                f" {record['taught_by']} showed it to me."
                if record.get('taught_by')
                else ''
            )
            return (
                f"I remember {name}: it looked like detector class "
                f"{record['class_id']} when you showed it to me.{teacher}"
            )

    for command in ('learn', 'teach'):
        prefix = f'{command} '
        if normalized == command or normalized.startswith(prefix):
            raw_command, separator, raw_steps = text.strip()[len(prefix):].partition(':')
            name = raw_command.strip()
            steps = [step.strip() for step in raw_steps.split(';') if step.strip()]
            if separator and name and steps:
                skill_id = '_'.join(name.casefold().split())
                skills[skill_id] = {
                    'name': name,
                    'steps': steps,
                    'confidence': 0.5,
                }
                _save(skills)
                return f'Okay, {name} is in my memory now. {len(steps)} steps—got it.'
            return 'Try saying: learn <task>: step one; step two.'

    if TASK_INTENT.search(normalized):
        for skill_id, skill in skills.items():
            if (
                isinstance(skill, dict)
                and isinstance(skill.get('name'), str)
                and isinstance(skill.get('steps'), list)
                and all(isinstance(step, str) for step in skill['steps'])
                and (skill_id in normalized or skill['name'].casefold() in normalized)
            ):
                return (
                    f"Oh, you showed me this: "
                    + ', then '.join(skill['steps'])
                )
        return 'Honestly, I have no clue; nobody has shown me how. Want to teach me?'

    object_introduction = OBJECT_INTRODUCTION.search(text)
    if object_introduction:
        from primus.camera_state import latest_teacher_frame
        from primus.objects import ground_object
        from primus.people import most_recent_person

        name = object_introduction.group(1)
        try:
            grounded = ground_object(
                name,
                latest_teacher_frame(),
                taught_by=most_recent_person(),
            )
        except (OSError, RuntimeError, ValueError):
            return f"I couldn't save what I saw for {name}. Could you try again?"
        if grounded:
            return (
                f"Got it, that's a {name}. I found an object very close to "
                "the person in view and saved that association."
            )
        return (
            "I couldn't confidently match an object near the person in view. "
            "Hold it close to the camera and try again."
        )

    if 'who are you' in normalized or 'your name' in normalized:
        return 'I’m Primus, your little robot pal. I learn things from what you teach me.'
    return chat(text, system_prompt=cognition_context['system_prompt'])
