from pathlib import Path

from primus.scene_builder import SCENE, build_xml

xml = build_xml()
workspace = Path(__file__).resolve().parent
scene_dir = Path(SCENE).resolve().parent
g1_xml = scene_dir / 'g1.xml'
g1_dump = workspace / 'sim' / 'kitchen_g1.xml'
g1_source = g1_xml.read_text()
g1_source = g1_source.replace(
    'meshdir="assets"',
    f'meshdir="{(scene_dir / "assets").resolve()}"',
    1,
)
g1_dump.write_text(g1_source)

xml = xml.replace(f'file="{g1_xml}"', f'file="{g1_dump.resolve()}"', 1)
output = workspace / 'sim' / 'kitchen_compiled.xml'
output.write_text(xml)
print('Dumped to sim/kitchen_compiled.xml')
