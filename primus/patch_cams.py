import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / 'sim' / 'g1' / 'g1.xml'
EYE = '<camera name="g1_eye" pos="0.06 0 0.42" mode="targetbody" target="focus" fovy="70"/>'
WRIST = '<camera name="g1_wrist" pos="0.06 0 0.06" mode="targetbody" target="focus" fovy="85"/>'


def attach_camera(xml, name, camera, candidates):
    if f'name="{name}"' in xml:
        body = next(
            (body for body in candidates if re.search(
                rf'<body\b[^>]*\bname="{re.escape(body)}"', xml
            )),
            None,
        )
        return xml, body

    for body in candidates:
        pattern = rf'(<body\b[^>]*\bname="{re.escape(body)}"[^>]*>)'
        xml, count = re.subn(pattern, rf'\1{camera}', xml, count=1)
        if count:
            return xml, body
    raise ValueError(f'None of the camera attachment bodies exist: {candidates}')


def main():
    xml = PATH.read_text()
    xml, eye_body = attach_camera(
        xml, 'g1_eye', EYE, ('head_link', 'torso_link', 'waist_pitch_link')
    )
    xml, wrist_body = attach_camera(
        xml, 'g1_wrist', WRIST,
        ('left_wrist_roll_link', 'left_wrist_yaw_link', 'left_elbow_link'),
    )
    PATH.write_text(xml)
    print('eye ->', eye_body, '| wrist ->', wrist_body)


if __name__ == '__main__':
    main()
