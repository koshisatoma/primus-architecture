import gradio as gr

from primus.cams import Cams


C = Cams()
CSS = """
html, body {
    background: #000 !important;
    height: 100%;
    margin: 0 !important;
    overflow: hidden;
    padding: 0 !important;
    width: 100%;
}
.gradio-container {
    background: #000 !important;
    margin: 0 !important;
    max-width: 100vw !important;
    min-height: 100vh !important;
    padding: 0 !important;
    width: 100vw !important;
}
footer { display: none !important; }
#viewport {
    background: #000 !important;
    border: 0 !important;
    height: 100vh !important;
    inset: 0;
    margin: 0 !important;
    max-width: none !important;
    overflow: hidden !important;
    padding: 0 !important;
    position: fixed !important;
    width: 100vw !important;
    z-index: 0;
}
#viewport > div, #viewport .image-container, #viewport .wrap {
    background: #000 !important;
    height: 100vh !important;
    max-height: none !important;
    padding: 0 !important;
    width: 100vw !important;
}
#viewport img {
    background: #000 !important;
    height: 90vh !important;
    max-height: none !important;
    object-fit: contain !important;
    width: 100vw !important;
}
#swap-row {
    align-items: center;
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 4px;
    justify-content: center;
    width: 100%;
}
#swap-row button {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 999px !important;
    color: #d8e4e2 !important;
    flex: 0 0 auto !important;
    font: 600 11px ui-monospace, monospace !important;
    letter-spacing: .04em;
    min-height: 32px !important;
    padding: 4px 10px !important;
    white-space: nowrap;
}
#swap-row button:hover {
    background: #71e0c022 !important;
    border-color: #71e0c088 !important;
    color: #8af0ce !important;
}
#controls {
    align-items: center;
    backdrop-filter: blur(14px);
    background: #080d12e8;
    border: 1px solid #ffffff26;
    border-radius: 14px;
    bottom: 12px;
    left: 50%;
    max-width: 96vw;
    padding: 8px 12px;
    position: fixed !important;
    transform: translateX(-50%);
    width: min(920px, 96vw);
    z-index: 20;
}
#orbit-row {
    align-items: center;
    gap: 14px;
}
#orbit-row > div { flex: 1 1 0; min-width: 120px; }
#telemetry {
    background: #080d12bb !important;
    border: 1px solid #ffffff24 !important;
    border-radius: 8px !important;
    left: 12px;
    max-width: calc(100vw - 24px);
    position: fixed !important;
    top: 12px;
    width: max-content;
    z-index: 20;
}
#telemetry textarea {
    background: transparent !important;
    border: 0 !important;
    color: #8af0ce !important;
    font: 600 10px ui-monospace, monospace !important;
    min-height: 28px !important;
    padding: 6px 10px !important;
}
@media (max-width: 620px) {
    #swap-row button {
        font-size: 9px !important;
        padding: 3px 7px !important;
    }
    #orbit-row { gap: 4px; }
    #orbit-row > div { min-width: 90px; }
    #controls { bottom: 8px; padding: 6px; }
}
"""


def tick():
    return C.view(), C.status()


def focus(name):
    C.focus = name
    return tick()


def set_orbit(azimuth, elevation, distance):
    C.orbit = (azimuth, elevation, distance)


def build():
    with gr.Blocks() as demo:
        viewport = gr.Image(
            elem_id='viewport',
            interactive=False,
            show_label=False,
        )
        telemetry = gr.Textbox(
            elem_id='telemetry',
            interactive=False,
            lines=1,
            show_label=False,
        )
        with gr.Column(elem_id='controls'):
            with gr.Row(elem_id='swap-row'):
                for name in C.names():
                    label = name.replace('_', ' ').upper()
                    gr.Button(label, size='sm').click(
                        focus,
                        inputs=gr.State(name),
                        outputs=[viewport, telemetry],
                    )
            with gr.Row(elem_id='orbit-row'):
                azimuth = gr.Slider(
                    0, 360, value=200, label='azimuth', step=1
                )
                elevation = gr.Slider(
                    -80, 20, value=-15, label='elevation', step=1
                )
                distance = gr.Slider(
                    1.0, 8.0, value=3.5, label='distance', step=0.1
                )
                for slider in (azimuth, elevation, distance):
                    slider.change(
                        set_orbit,
                        inputs=[azimuth, elevation, distance],
                        outputs=None,
                    )
        gr.Timer(0.05).tick(
            tick, inputs=None, outputs=[viewport, telemetry]
        )
        demo.load(tick, inputs=None, outputs=[viewport, telemetry])
    return demo
