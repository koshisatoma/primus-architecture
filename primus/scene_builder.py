import mujoco
import os
import tempfile

SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(SCENE_DIR, '..', 'sim', 'g1', 'scene.xml')


ADD_TO_ASSET = '''
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.22 0.22 0.26" rgb2="0.32 0.32 0.36"/>
    <material name="floor" texture="grid" texrepeat="4 4"/>
'''

ADD_TO_WORLDBODY = '''
    <light pos="0 0 3" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor"/>
    <geom name="wall_n" type="plane" pos="0 3 1.5" zaxis="0 -1 0" size="3 1.5 0.1" rgba="0.85 0.85 0.90 1"/>
    <geom name="wall_w" type="plane" pos="-3 0 1.5" zaxis="1 0 0" size="3 1.5 0.1" rgba="0.85 0.85 0.90 1"/>

    <body name="counter_w" pos="-2.35 0.4 0">
      <geom type="box" size="0.3 1.6 0.45" pos="0 0 0.45" rgba="0.45 0.33 0.22 1"/>
      <geom type="box" size="0.33 1.65 0.02" pos="0 0 0.92" rgba="0.15 0.15 0.17 1"/>
      <geom type="box" size="0.22 0.35 0.08" pos="0 0.6 0.86" rgba="0.6 0.6 0.65 1"/>
    </body>
    <body name="stove" pos="-2.35 -1.5 0">
      <geom type="box" size="0.32 0.35 0.45" pos="0 0 0.45" rgba="0.1 0.1 0.12 1"/>
      <geom type="box" size="0.34 0.37 0.02" pos="0 0 0.92" rgba="0.2 0.2 0.22 1"/>
    </body>
    <body name="fridge" pos="1.9 2.7 0">
      <geom type="box" size="0.35 0.35 0.9" pos="0 0 0.9" rgba="0.75 0.78 0.8 1"/>
    </body>
    <body name="table" pos="1.5 -1.2 0">
      <geom type="cylinder" size="0.45 0.02" pos="0 0 0.72" rgba="0.55 0.4 0.25 1"/>
      <geom type="cylinder" size="0.05 0.36" pos="0 0 0.36" rgba="0.3 0.2 0.15 1"/>
    </body>
    <body name="chair1" pos="1.5 -0.5 0">
      <geom type="box" size="0.18 0.18 0.02" pos="0 0 0.45" rgba="0.2 0.4 0.6 1"/>
    </body>
    <geom name="rug" type="box" size="0.9 0.7 0.005" pos="0.3 0.2 0.006" rgba="0.35 0.15 0.2 1"/>

    <body name="mug"    pos="-2.35 0.9 0.94"><geom type="cylinder" size="0.04 0.05" rgba="0.85 0.2 0.2 1"/></body>
    <body name="bowl"   pos="-2.35 0.2 0.94"><geom type="sphere" size="0.07" rgba="0.9 0.9 0.95 1"/></body>
    <body name="bottle" pos="-2.35 -0.4 0.94"><geom type="capsule" size="0.035 0.12" rgba="0.2 0.6 0.9 1"/></body>
    <body name="apple"  pos="-2.3 1.4 0.94"><geom type="sphere" size="0.035" rgba="0.9 0.1 0.1 1"/></body>
    <body name="banana" pos="-2.4 1.55 0.94"><geom type="capsule" size="0.02 0.09" rgba="0.95 0.85 0.2 1"/></body>
    <body name="pan"    pos="-2.35 -1.5 0.95"><geom type="cylinder" size="0.11 0.02" rgba="0.15 0.15 0.15 1"/></body>
    <body name="knife"  pos="-2.3 0.6 0.97"><geom type="box" size="0.012 0.09 0.002" rgba="0.8 0.8 0.85 1"/></body>
    <body name="cup2"   pos="1.4 -1.05 0.75"><geom type="cylinder" size="0.035 0.045" rgba="0.2 0.7 0.5 1"/></body>

    <body name="focus" pos="0 0.5 1.0">
      <geom type="sphere" size="0.02" rgba="1 0 0 0" contype="0" conaffinity="0"/>
    </body>

    <camera name="kitchen" pos="2.2 2.2 1.7" mode="targetbody" target="focus" fovy="60"/>
    <camera name="room"    pos="-2.5 -2.5 2.3" mode="targetbody" target="focus" fovy="65"/>
    <camera name="front"   pos="0 -2.7 1.4" mode="targetbody" target="focus" fovy="60"/>
    <camera name="front3m" pos="0 -3 1.4" xyaxes="1 0 0 0 0.34 0.94" fovy="60"/>
    <camera name="corner"  pos="2.6 -2.6 2.2" xyaxes="0.7 0.7 0 -0.23 0.23 0.95" fovy="70"/>
    <camera name="g1_body" pos="2.4 0 1.15" mode="targetbody" target="pelvis" fovy="45"/>
'''


def build_xml():
    with open(SCENE) as f:
        xml = f.read()

    base_floor = '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>'
    if base_floor not in xml:
        raise ValueError(f'Could not find the G1 scene floor in {SCENE}')
    xml = xml.replace(base_floor, '', 1)

    scene_dir = os.path.dirname(os.path.abspath(SCENE))
    xml = xml.replace(
        '<mujoco model="g1_29dof_rev_1_0 scene">',
        f'<mujoco model="g1_29dof_rev_1_0 scene">\n'
        f'  <compiler meshdir="{os.path.join(scene_dir, "assets")}"/>',
        1,
    )
    xml = xml.replace(
        '<global azimuth="140" elevation="-20"/>',
        '<global azimuth="140" elevation="-20" offwidth="960" offheight="540"/>',
        1,
    )
    xml = xml.replace('</asset>', ADD_TO_ASSET + '\n  </asset>')
    from primus.room import room_xml
    chunk, n_items = room_xml()
    xml = xml.replace('</worldbody>', chunk + '\n</worldbody>')
    print(f'[room] injected {n_items} objects')
    xml = xml.replace('</worldbody>', ADD_TO_WORLDBODY + '\n  </worldbody>')
    xml = xml.replace('file="g1.xml"', f'file="{os.path.join(scene_dir, "g1.xml")}"')
    return xml


def load_model():
    scene_dir = os.path.dirname(os.path.abspath(SCENE))
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', dir=scene_dir) as xml_file:
        xml_file.write(build_xml())
        xml_file.flush()
        return mujoco.MjModel.from_xml_path(xml_file.name)