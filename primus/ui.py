import gradio as gr

from primus.cams import Cams
from primus import viewops as V


CSS = """
html, body, .gradio-container {
    background: #0b1220 !important;
    color: #e9eff8 !important;
}
.gradio-container {
    margin: 0 auto !important;
    max-width: 1500px !important;
    min-height: 100vh !important;
    padding: 18px !important;
}
footer { display: none !important; }
.block, .form, .container, .wrap, [data-testid="block-info"] {
    background-color: #101826 !important;
    border-color: #1d2a3d !important;
    color: #e9eff8 !important;
}
label, .label-wrap, .prose, .markdown-body, input, textarea, select {
    color: #e9eff8 !important;
}
input, textarea, select, [role="combobox"] {
    background-color: #101826 !important;
}
.primus-subtitle { color: #8ea3bf; margin-top: 2px; }
.primus-section { color: #dfe8f4; font-weight: 600; }
.gr-box, .gr-panel, .gr-group {
    background: #101826 !important;
    border-color: #1d2a3d !important;
}
@media (max-width: 800px) {
    .gradio-container { padding: 10px !important; }
}
"""

SIM_CAMERA_CHOICES = ['kitchen', 'room', 'g1_eye', 'g1_body']
C = Cams()


def tick(mode):
    return V.tick_all(mode, C.dashboard_views(), C.status())


def focus(name):
    C.focus = name


def set_orbit(azimuth, elevation, distance):
    C.orbit = (azimuth, elevation, distance)


def build():
    with gr.Blocks(title='PRIMUS') as demo:
        gr.HTML(
            '<h1 style="color:#e9eff8;margin:0;font-weight:700">'
            'Robotics Vision &amp; Simulation</h1>'
            '<div class="primus-subtitle">'
            'MuJoCo Simulation&nbsp; | &nbsp;Live Camera Input</div>'
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.HTML('<div class="primus-section">MuJoCo Simulation</div>')
                sim_main = gr.Image(
                    label='MuJoCo Simulation',
                    height=430,
                    interactive=False,
                )
                with gr.Row():
                    sim_front = gr.Image(
                        label='Front of Robot Cam (3 m)',
                        height=200,
                        interactive=False,
                    )
                    sim_corner = gr.Image(
                        label='Corner of Room Cam',
                        height=200,
                        interactive=False,
                    )
                details = gr.Markdown('### Simulation Details')
                with gr.Accordion('Simulation controls', open=False):
                    sim_camera = gr.Dropdown(
                        SIM_CAMERA_CHOICES,
                        value='kitchen',
                        label='Main simulation camera',
                    )
                    sim_camera.change(focus, inputs=sim_camera, outputs=None)
                    with gr.Row():
                        azimuth = gr.Slider(
                            0, 360, value=200, step=1, label='Azimuth'
                        )
                        elevation = gr.Slider(
                            -80, 20, value=-15, step=1, label='Elevation'
                        )
                        distance = gr.Slider(
                            1.0, 8.0, value=3.5, step=0.1, label='Distance'
                        )
                    for slider in (azimuth, elevation, distance):
                        slider.change(
                            set_orbit,
                            inputs=[azimuth, elevation, distance],
                            outputs=None,
                        )
            with gr.Column(scale=1):
                gr.HTML('<div class="primus-section">Camera Input</div>')
                mode = gr.Dropdown(
                    V.CHOICES,
                    value=V.CHOICES[0],
                    label='Select Input Type',
                )
                live = gr.Image(
                    label='Live Camera (RGB)',
                    height=520,
                    interactive=False,
                )

        gr.Timer(0.15).tick(
            fn=tick,
            inputs=[mode],
            outputs=[sim_main, sim_front, sim_corner, live, details],
        )
        demo.load(
            fn=tick,
            inputs=[mode],
            outputs=[sim_main, sim_front, sim_corner, live, details],
        )
    return demo
