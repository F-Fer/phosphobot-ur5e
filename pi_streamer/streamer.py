"""ZMQ camera streamer for remote Phosphobot cameras.

This script is intended to run on an edge device (e.g., Raspberry Pi 4)
connected to USB cameras. It captures frames via OpenCV and publishes them
to the Phosphobot workstation over ZeroMQ as multipart PUSH messages.

Usage:
  uv run python streamer.py --config config.json

See `config.example.json` for a sample configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import zmq
import zmq.asyncio
from pydantic import BaseModel, Field, ValidationError
from base64 import b64encode


class CameraConfig(BaseModel):
    device: str = Field(..., description="Path or index of the video device")
    topic: str = Field(..., description="ZMQ topic name")
    width: Optional[int] = Field(
        default=None, ge=16, le=7680, description="Capture width in pixels"
    )
    height: Optional[int] = Field(
        default=None, ge=16, le=4320, description="Capture height in pixels"
    )
    fps: Optional[int] = Field(
        default=None, ge=1, le=120, description="Requested frames per second"
    )
    fourcc: Optional[str] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Optional fourcc (e.g., MJPG, YUYV). Auto if omitted.",
    )


class StreamerConfig(BaseModel):
    endpoint: str = Field(
        "tcp://0.0.0.0:5555", description="ZMQ PUSH bind endpoint"
    )
    cameras: list[CameraConfig]
    jpeg_quality: int = Field(
        default=80, ge=10, le=100, description="JPEG quality for encoding"
    )
    reconnect_interval: float = Field(
        default=3.0, ge=0.5, le=30.0, description="Seconds between reconnects"
    )


@dataclass
class VideoCapture:
    config: CameraConfig
    capture: cv2.VideoCapture


class CameraStreamer:
    def __init__(self, config: StreamerConfig, loop: asyncio.AbstractEventLoop):
        self.config = config
        self.loop = loop
        self.context = zmq.asyncio.Context.instance()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDHWM, 3)
        self.socket.bind(self.config.endpoint)
        self.captures: list[VideoCapture] = []
        self._shutdown = asyncio.Event()

    async def initialize(self) -> None:
        for camera_cfg in self.config.cameras:
            capture = self._open_capture(camera_cfg)
            if capture is None:
                continue
            self.captures.append(VideoCapture(camera_cfg, capture))

        if not self.captures:
            raise RuntimeError("No cameras could be opened; check configuration")

    def _open_capture(self, camera_cfg: CameraConfig) -> Optional[cv2.VideoCapture]:
        device = camera_cfg.device
        # Allow numeric strings to be treated as indices
        if device.isdigit():
            index = int(device)
            capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        else:
            capture = cv2.VideoCapture(device, cv2.CAP_V4L2)

        if not capture.isOpened():
            print(f"[WARN] Failed to open camera {device}")
            return None

        if camera_cfg.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.width)
        if camera_cfg.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.height)
        if camera_cfg.fps:
            capture.set(cv2.CAP_PROP_FPS, camera_cfg.fps)
        if camera_cfg.fourcc:
            code = cv2.VideoWriter.fourcc(*camera_cfg.fourcc)
            capture.set(cv2.CAP_PROP_FOURCC, code)

        # Run one read to confirm we get frames
        ok, frame = capture.read()
        if not ok or frame is None:
            print(
                f"[WARN] Camera {device} produced no frames after initialization"
            )
            capture.release()
            return None

        return capture

    async def start(self) -> None:
        print(
            f"[INFO] ZMQ streamer bound to {self.config.endpoint} for"
            f" {len(self.captures)} camera(s)"
        )
        tasks = [self._spawn_camera_task(vc) for vc in self.captures]
        await asyncio.gather(*tasks)

    async def _spawn_camera_task(self, video_capture: VideoCapture) -> None:
        cfg = video_capture.config
        capture = video_capture.capture
        topic_bytes = cfg.topic.encode()
        interval = 1.0 / (cfg.fps or capture.get(cv2.CAP_PROP_FPS) or 30.0)
        last_log = time.time()
        frames_sent = 0

        try:
            while not self._shutdown.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    print(
                        f"[WARN] Camera {cfg.device} read failed; retrying in"
                        f" {self.config.reconnect_interval}s"
                    )
                    await asyncio.sleep(self.config.reconnect_interval)
                    continue

                # Convert from BGR to RGB for consistency with Phosphobot
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                payload = self._encode_frame(frame_rgb)
                message = {
                    "shape": frame_rgb.shape,
                    "dtype": str(frame_rgb.dtype),
                    "timestamp": time.time(),
                    "frame_bytes": payload,
                }

                await self.socket.send_multipart(
                    [topic_bytes, json.dumps(message).encode("utf-8")]
                )

                frames_sent += 1
                now = time.time()
                if now - last_log > 5:
                    print(
                        f"[INFO] Camera {cfg.topic}: {frames_sent / (now - last_log):.1f} fps"
                    )
                    frames_sent = 0
                    last_log = now

                await asyncio.sleep(interval)
        finally:
            capture.release()
            print(f"[INFO] Camera {cfg.topic} capture closed")

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)]
        ok, buffer = cv2.imencode(".jpg", frame, params)
        if not ok:
            raise RuntimeError("Failed to encode frame")
        return b64encode(buffer).decode("ascii")

    async def shutdown(self) -> None:
        self._shutdown.set()
        await asyncio.sleep(0.1)
        self.socket.close(linger=0)
        self.context.term()


def load_config(path: Path) -> StreamerConfig:
    raw = json.loads(path.read_text())
    try:
        return StreamerConfig.model_validate(raw)
    except ValidationError as exc:
        print("Invalid configuration:\n", exc)
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phosphobot ZMQ camera streamer")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON configuration file",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    loop = asyncio.get_running_loop()
    streamer = CameraStreamer(config, loop)
    await streamer.initialize()

    stop_event = asyncio.Event()

    def _signal_handler(*_: Any) -> None:
        print("[INFO] Shutdown signal received")
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _signal_handler)
    loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    await asyncio.gather(streamer.start(), stop_event.wait())
    await streamer.shutdown()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()


