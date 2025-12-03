import asyncio
import base64
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pynvml
import tritonclient.grpc as grpcclient
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sklearn.manifold import TSNE
from uvicorn import Config, Server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("orchestrator")


@dataclass
class CameraConfig:
    name: str
    url: str
    fps: int = 15
    width: int = 1280
    height: int = 720


@dataclass
class ModelConfig:
    ensemble_name: str
    detection_name: str
    segmentation_name: str
    depth_name: str
    flow_name: str
    embedding_name: str


class CameraStream:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.capture = cv2.VideoCapture(config.url)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.interval = 1.0 / max(config.fps, 1)
        self.last_frame: Optional[np.ndarray] = None
        self.prev_frame: Optional[np.ndarray] = None
        self.running = True

    async def frames(self):
        while self.running:
            start = time.time()
            ret, frame = self.capture.read()
            if not ret:
                LOGGER.warning("Camera %s dropped frame", self.config.name)
                await asyncio.sleep(self.interval)
                continue
            frame = cv2.resize(frame, (self.config.width, self.config.height))
            self.prev_frame = self.last_frame
            self.last_frame = frame
            yield frame, self.prev_frame
            elapsed = time.time() - start
            await asyncio.sleep(max(0, self.interval - elapsed))

    def stop(self) -> None:
        self.running = False
        self.capture.release()


class TritonMultiClient:
    def __init__(self, url: str, model_cfg: ModelConfig) -> None:
        self.client = grpcclient.InferenceServerClient(url=url, verbose=False)
        self.model_cfg = model_cfg

    def preprocess(self, frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        resized = cv2.resize(frame, size)
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def infer(self, current: np.ndarray, previous: Optional[np.ndarray]) -> Dict[str, Any]:
        curr_det = self.preprocess(current, (640, 640))
        curr_seg = self.preprocess(current, (512, 512))
        curr_depth = self.preprocess(current, (224, 224))
        curr_clip = self.preprocess(current, (224, 224))
        curr_flow = self.preprocess(current, (320, 320))
        prev_flow = self.preprocess(previous if previous is not None else current, (320, 320))
        inputs = [
            grpcclient.InferInput("image_det", curr_det.shape, "FP32"),
            grpcclient.InferInput("image_seg", curr_seg.shape, "FP32"),
            grpcclient.InferInput("image_depth", curr_depth.shape, "FP32"),
            grpcclient.InferInput("image_flow_prev", prev_flow.shape, "FP32"),
            grpcclient.InferInput("image_flow_curr", curr_flow.shape, "FP32"),
            grpcclient.InferInput("image_clip", curr_clip.shape, "FP32"),
        ]
        inputs[0].set_data_from_numpy(curr_det)
        inputs[1].set_data_from_numpy(curr_seg)
        inputs[2].set_data_from_numpy(curr_depth)
        inputs[3].set_data_from_numpy(prev_flow)
        inputs[4].set_data_from_numpy(curr_flow)
        inputs[5].set_data_from_numpy(curr_clip)
        outputs = [grpcclient.InferRequestedOutput(name) for name in [
            "det_boxes",
            "det_scores",
            "det_labels",
            "seg_masks",
            "depth",
            "flow",
            "embedding",
        ]]
        start = time.time()
        result = self.client.infer(self.model_cfg.ensemble_name, inputs=inputs, outputs=outputs)
        latency_ms = (time.time() - start) * 1000
        parsed = {
            "boxes": result.as_numpy("det_boxes"),
            "scores": result.as_numpy("det_scores"),
            "labels": result.as_numpy("det_labels"),
            "masks": result.as_numpy("seg_masks"),
            "depth": result.as_numpy("depth"),
            "flow": result.as_numpy("flow"),
            "embedding": result.as_numpy("embedding"),
            "latency_ms": latency_ms,
        }
        return parsed


class Metrics(BaseModel):
    fps: float
    triton_latency_ms: float
    end_to_end_ms: float
    postprocess_ms: float
    gpu_util: float
    vram_used_gb: float
    vram_total_gb: float


class EmbeddingProjector:
    def __init__(self, buffer_size: int = 512) -> None:
        self.buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.labels: Deque[str] = deque(maxlen=buffer_size)
        self.coords: List[Tuple[float, float]] = []

    def add(self, embedding: np.ndarray, label: str) -> None:
        self.buffer.append(embedding.flatten())
        self.labels.append(label)

    def compute(self) -> None:
        if len(self.buffer) < 5:
            return
        data = np.stack(list(self.buffer), axis=0)
        tsne = TSNE(n_components=2, perplexity=10, init="pca", learning_rate="auto")
        coords = tsne.fit_transform(data)
        self.coords = [(float(x), float(y)) for x, y in coords]

    def export(self) -> Dict[str, Any]:
        return {
            "points": self.coords,
            "labels": list(self.labels),
        }


class OverlayRenderer:
    def __init__(self) -> None:
        self.palette = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]

    def draw(self, frame: np.ndarray, detections: Dict[str, np.ndarray], masks: Optional[np.ndarray], depth: Optional[np.ndarray], flow: Optional[np.ndarray]) -> np.ndarray:
        output = frame.copy()
        boxes = detections.get("boxes") if detections.get("boxes") is not None else []
        scores = detections.get("scores") if detections.get("scores") is not None else []
        labels = detections.get("labels") if detections.get("labels") is not None else []
        for i, box in enumerate(boxes[:5]):
            color = self.palette[i % len(self.palette)]
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"ID {labels[i]}: {scores[i]:.2f}"
            cv2.putText(output, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if masks is not None:
            mask = np.argmax(masks, axis=0)
            colored = cv2.applyColorMap((mask % 255).astype(np.uint8), cv2.COLORMAP_JET)
            colored = cv2.resize(colored, (output.shape[1], output.shape[0]))
            output = cv2.addWeighted(output, 0.7, colored, 0.3, 0)
        if depth is not None:
            depth_map = depth.squeeze()
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
            depth_colored = cv2.resize(depth_colored, (output.shape[1] // 3, output.shape[0] // 3))
            output[0:depth_colored.shape[0], 0:depth_colored.shape[1]] = depth_colored
        if flow is not None:
            flow_show = self.visualize_flow(flow)
            flow_show = cv2.resize(flow_show, (output.shape[1] // 3, output.shape[0] // 3))
            output[0:flow_show.shape[0], output.shape[1] - flow_show.shape[1]: output.shape[1]] = flow_show
        return output

    def visualize_flow(self, flow: np.ndarray) -> np.ndarray:
        fx, fy = flow[0], flow[1]
        mag, ang = cv2.cartToPolar(fx, fy)
        hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def encode_frame(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("ascii")


def load_config(path: str) -> Tuple[List[CameraConfig], ModelConfig]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cameras = [CameraConfig(**c) for c in raw.get("cameras", [])]
    models = ModelConfig(**raw["models"])
    return cameras, models


def read_gpu_metrics() -> Tuple[float, float, float]:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = mem_info.used / (1024 ** 3)
    total = mem_info.total / (1024 ** 3)
    return util, used, total


class PipelineManager:
    def __init__(self, config_path: str) -> None:
        cameras, model_cfg = load_config(config_path)
        self.cameras = cameras
        self.model_cfg = model_cfg
        self.streams = {cfg.name: CameraStream(cfg) for cfg in cameras}
        self.triton = TritonMultiClient(os.getenv("TRITON_GRPC_URL", "localhost:8001"), model_cfg)
        self.renderer = OverlayRenderer()
        self.projector = EmbeddingProjector()
        self.metrics: Dict[str, Metrics] = {}
        self.latest_frames: Dict[str, str] = {}
        self.embedding_export: Dict[str, Any] = {}

    async def run_camera(self, name: str, stream: CameraStream) -> None:
        async for frame, prev in stream.frames():
            start = time.time()
            inference = self.triton.infer(frame, prev)
            post_start = time.time()
            rendered = self.renderer.draw(
                frame,
                {"boxes": inference["boxes"], "scores": inference["scores"], "labels": inference["labels"]},
                inference.get("masks"),
                inference.get("depth"),
                inference.get("flow"),
            )
            self.projector.add(inference["embedding"], label=f"{name}")
            post_ms = (time.time() - post_start) * 1000
            fps = 1.0 / max(time.time() - start, 1e-3)
            gpu_util, vram_used, vram_total = read_gpu_metrics()
            self.metrics[name] = Metrics(
                fps=fps,
                triton_latency_ms=inference["latency_ms"],
                end_to_end_ms=(time.time() - start) * 1000,
                postprocess_ms=post_ms,
                gpu_util=gpu_util,
                vram_used_gb=vram_used,
                vram_total_gb=vram_total,
            )
            self.latest_frames[name] = encode_frame(rendered)

    async def background_project(self) -> None:
        while True:
            await asyncio.sleep(5)
            self.projector.compute()
            self.embedding_export = self.projector.export()

    async def start(self) -> None:
        tasks = [asyncio.create_task(self.run_camera(name, stream)) for name, stream in self.streams.items()]
        tasks.append(asyncio.create_task(self.background_project()))
        await asyncio.gather(*tasks)


class WSMessage(BaseModel):
    type: str
    camera: Optional[str] = None
    frame: Optional[str] = None
    metrics: Optional[Metrics] = None
    embeddings: Optional[Dict[str, Any]] = None


app = FastAPI(title="Vision Orchestrator")
manager: Optional[PipelineManager] = None


@app.on_event("startup")
async def startup_event() -> None:
    global manager
    config_path = os.getenv("CONFIG_PATH", "/config/cameras.yaml")
    LOGGER.info("Loading config from %s", config_path)
    manager = PipelineManager(config_path)
    asyncio.create_task(manager.start())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    assert manager is not None
    try:
        while True:
            await asyncio.sleep(0.2)
            for name in manager.streams.keys():
                frame_b64 = manager.latest_frames.get(name)
                metrics = manager.metrics.get(name)
                if frame_b64 and metrics:
                    await websocket.send_text(WSMessage(
                        type="frame",
                        camera=name,
                        frame=frame_b64,
                        metrics=metrics,
                    ).model_dump_json())
            if manager.embedding_export:
                await websocket.send_text(WSMessage(
                    type="embeddings",
                    embeddings=manager.embedding_export,
                ).model_dump_json())
    except WebSocketDisconnect:
        LOGGER.info("WebSocket disconnected")


def main() -> None:
    config = Config(app=app, host="0.0.0.0", port=8080, log_level="info")
    server = Server(config)
    server.run()


if __name__ == "__main__":
    main()
