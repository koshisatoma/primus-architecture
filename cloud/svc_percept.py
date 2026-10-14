import os
import threading

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool
import uvicorn


app = FastAPI()
DEVICE = os.environ.get('PRIMUS_DEVICE', 'cuda')
MODELS_DIR = os.environ.get('MODELS', os.path.join(os.path.expanduser('~'), 'models'))
_model_lock = threading.Lock()
_whisper = _depth = _dinov2 = _dinov2_processor = _yolo = None


def _model_path(name):
    return os.path.join(MODELS_DIR, name)


def _get_whisper():
    global _whisper
    if _whisper is None:
        with _model_lock:
            if _whisper is None:
                from faster_whisper import WhisperModel

                _whisper = WhisperModel(
                    _model_path('whisper-large-v3'),
                    device=DEVICE,
                    compute_type='float16' if DEVICE.startswith('cuda') else 'int8',
                )
    return _whisper


def _get_depth_model():
    global _depth
    if _depth is None:
        with _model_lock:
            if _depth is None:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                path = _model_path('depth-anything')
                _depth = (
                    AutoImageProcessor.from_pretrained(path),
                    AutoModelForDepthEstimation.from_pretrained(path)
                    .to(DEVICE)
                    .eval(),
                )
    return _depth


def _get_dinov2():
    global _dinov2, _dinov2_processor
    if _dinov2 is None:
        with _model_lock:
            if _dinov2 is None:
                from transformers import AutoImageProcessor, AutoModel

                path = _model_path('dinov2')
                _dinov2_processor = AutoImageProcessor.from_pretrained(path)
                _dinov2 = AutoModel.from_pretrained(path).to(DEVICE).eval()
    return _dinov2_processor, _dinov2


def _get_yolo():
    global _yolo
    if _yolo is None:
        with _model_lock:
            if _yolo is None:
                from ultralytics import YOLO

                _yolo = YOLO(os.environ.get('PRIMUS_YOLO_MODEL', 'yolo11x.pt'))
    return _yolo


def _decode_image(contents):
    if not contents:
        raise HTTPException(status_code=400, detail='Request body must contain an image')
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail='Request body must be a valid image')
    return image


def _transcribe(pcm):
    audio = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
    segments, _ = _get_whisper().transcribe(audio, language='en', beam_size=5)
    return {'text': ' '.join(segment.text for segment in segments).strip()}


@app.post('/transcribe')
async def transcribe(req: Request):
    pcm = await req.body()
    if not pcm or len(pcm) % 2:
        raise HTTPException(
            status_code=400,
            detail='Request body must contain non-empty 16-bit PCM audio',
        )
    return await run_in_threadpool(_transcribe, pcm)


def _estimate_depth(image):
    processor, model = _get_depth_model()
    inputs = processor(
        images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        return_tensors='pt',
    ).to(DEVICE)
    with torch.no_grad():
        prediction = model(**inputs).predicted_depth
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode='bilinear',
            align_corners=False,
        )[0, 0]
        maximum = prediction.max()
        if maximum > 0:
            prediction = prediction / maximum
        depth = (prediction * 65535).round().cpu().numpy().astype(np.uint16)
    return depth


@app.post('/depth')
async def depth(req: Request):
    image = _decode_image(await req.body())
    result = await run_in_threadpool(_estimate_depth, image)
    return Response(
        content=result.tobytes(),
        media_type='application/octet-stream',
        headers={
            'X-Depth-Width': str(result.shape[1]),
            'X-Depth-Height': str(result.shape[0]),
            'X-Depth-Dtype': 'uint16-le',
        },
    )


def _person_embeddings(image):
    detections = _get_yolo()(image, verbose=False, classes=[0])
    processor, model = _get_dinov2()
    embeddings = []
    boxes = []
    height, width = image.shape[:2]

    for box in detections[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x1 >= x2 or y1 >= y2:
            continue

        crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        inputs = processor(images=crop, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state[:, 0]
            embedding = torch.nn.functional.normalize(embedding, dim=1)
        embeddings.append(embedding[0].cpu().tolist())
        boxes.append([x1, y1, x2, y2])

    return {'embeddings': embeddings, 'boxes': boxes}


@app.post('/reid')
async def reid(req: Request):
    image = _decode_image(await req.body())
    return await run_in_threadpool(_person_embeddings, image)


if __name__ == '__main__':
    uvicorn.run(
        app,
        host=os.environ.get('PRIMUS_PERCEPT_HOST', '0.0.0.0'),
        port=int(os.environ.get('PRIMUS_PERCEPT_PORT', '8002')),
    )
