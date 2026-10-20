import json
import os
import re
from pathlib import Path

VERBS = {
    'pick': ['reach', 'grasp', 'lift'],
    'grab': ['reach', 'grasp'],
    'take': ['reach', 'grasp', 'lift'],
    'lift': ['lift'],
    'place': ['walk_to', 'place'],
    'put': ['walk_to', 'place'],
    'carry': ['grasp', 'lift', 'walk_to'],
    'bring': ['grasp', 'lift', 'walk_to'],
    'sit': ['sit'],
    'stand': ['stand'],
    'walk': ['walk_to'],
    'reach': ['reach'],
    'touch': ['reach'],
}
PREP = {'on': 'place', 'onto': 'place', 'in': 'place', 'into': 'place',
        'to': 'walk_to', 'near': 'walk_to'}
FALLBACK_OBJECTS = {
    'cup', 'bottle', 'chair', 'table', 'shelf', 'counter', 'sofa', 'bowl',
    'plate', 'mug', 'can', 'apple', 'orange', 'book', 'box', 'pan', 'spoon',
    'knife', 'fork', 'glass', 'jar', 'lemon', 'onion', 'pot', 'kettle',
    'remote', 'phone', 'pen', 'towel'
}


def _objects():
    objects = {name: name for name in FALLBACK_OBJECTS}
    path = Path(__file__).resolve().parent.parent / 'data' / 'objects.json'
    if path.exists():
        try:
            with path.open(encoding='utf-8') as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                for key in payload.keys():
                    clean = str(key).strip()
                    if clean:
                        objects[clean.casefold()] = clean
        except (OSError, ValueError):
            pass
    return objects


def parse_action(text):
    """Parse a natural-language action into a primitive DAG.

    Common object nouns are accepted even when they are not yet grounded in
    data/objects.json, so the parser still compiles a useful action plan.
    """
    if not text or not isinstance(text, str):
        return []

    obj_map = _objects()
    normalized = text.lower()
    tokens = re.findall(r'[a-z]+', normalized)
    mentions = []
    for name in sorted(obj_map, key=lambda item: len(item), reverse=True):
        for match in re.finditer(rf'\b{re.escape(name)}\b', normalized):
            mentions.append((name, match.start()))
    mentions.sort(key=lambda item: item[1])
    objects = [name for name, _ in mentions]
    if not objects:
        for token in tokens:
            if token in FALLBACK_OBJECTS:
                objects.append(token)
    operand = objects[0] if objects else None
    target = objects[1] if len(objects) > 1 else None
    plan = []
    for i, word in enumerate(tokens):
        if word in VERBS:
            if word in {'pick', 'grab', 'take'}:
                if operand is not None:
                    plan.extend([
                        {'prim': 'reach', 'obj': operand, 'target': target},
                        {'prim': 'grasp', 'obj': operand, 'target': target},
                        {'prim': 'lift', 'obj': operand, 'target': target},
                    ])
            elif word in {'place', 'put'}:
                if operand is not None:
                    if target is not None:
                        plan.append({'prim': 'walk_to', 'obj': operand, 'target': target})
                    plan.append({'prim': 'place', 'obj': operand, 'target': target})
            elif word in {'sit', 'stand', 'walk'}:
                if word == 'sit':
                    plan.append({'prim': 'sit', 'obj': operand, 'target': target})
                elif word == 'stand':
                    plan.append({'prim': 'stand', 'obj': operand, 'target': target})
                else:
                    plan.append({'prim': 'walk_to', 'obj': operand, 'target': target})
            else:
                for prim in VERBS[word]:
                    plan.append({'prim': prim, 'obj': operand, 'target': target})
        elif word in PREP and i + 1 < len(tokens):
            next_word = tokens[i + 1]
            if next_word in obj_map:
                target = obj_map[next_word]
                if operand is not None:
                    plan.append({'prim': PREP[word], 'obj': operand, 'target': target})
            elif next_word in FALLBACK_OBJECTS:
                target = next_word
                if operand is not None:
                    plan.append({'prim': PREP[word], 'obj': operand, 'target': target})
    if not plan and operand is not None:
        plan.append({'prim': 'reach', 'obj': operand, 'target': target})
    return plan


def teach_action(text):
    plan = parse_action(text)
    if not plan:
        return None
    name = re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()[:40].replace(' ', '_')
    skills_path = Path(__file__).resolve().parent.parent / 'data' / 'skills.json'
    skills = {}
    if skills_path.exists():
        try:
            with skills_path.open(encoding='utf-8') as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                skills = data
        except (OSError, ValueError):
            skills = {}
    skills[name] = {'kind': 'action', 'plan': plan, 'confidence': 0.5}
    with skills_path.open('w', encoding='utf-8') as handle:
        json.dump(skills, handle, indent=2)
    return name, plan
