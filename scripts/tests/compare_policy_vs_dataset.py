#!/usr/bin/env python3

"""Compare remote Pi0 policy actions vs dataset actions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download an observation from a phosphobot dataset, query the remote "
            "Pi0 policy server with the observation, and compare predicted vs "
            "ground-truth actions."
        )
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repository ID, e.g. F-Fer/ur-1.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to sample. If unset, a random episode is chosen.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Step index inside the episode. If unset, a random step is chosen.",
    )
    parser.add_argument(
        "--server-host",
        required=True,
        help="Remote policy server hostname or IP.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        required=True,
        help="Remote policy server port.",
    )
    parser.add_argument(
        "--image-keys",
        nargs="*",
        default=(
            "observation.images.main",
            "observation.images.secondary_0",
        ),
        help="Image keys to retrieve and forward to the policy.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Prompt string to include in the observation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the generated plot (PNG).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot interactively.",
    )
    return parser.parse_args()


def _list_available_episodes(repo_id: str) -> Sequence[str]:
    files = list_repo_files(repo_id, repo_type="dataset")
    return [f for f in files if f.startswith("data/") and f.endswith(".parquet")]


def _episode_file_for(repo_id: str, episode_index: int) -> Tuple[str, str, str, Optional[str]]:
    files = _list_available_episodes(repo_id)
    if not files:
        raise RuntimeError(f"No episodes found in dataset {repo_id}.")

    by_episode = sorted(files)
    if episode_index < 0 or episode_index >= len(by_episode):
        raise IndexError(
            f"Episode index {episode_index} out of range (0 <= idx < {len(by_episode)})"
        )
    parquet_relpath = by_episode[episode_index]

    # Derive associated video paths (if present)
    inner_name = Path(parquet_relpath).stem
    chunk = Path(parquet_relpath).parts[1]
    main_relpath = f"videos/{chunk}/observation.images.main/{inner_name}.mp4"
    wrist_relpath = f"videos/{chunk}/observation.images.secondary_0/{inner_name}.mp4"
    task_prompt = _load_episode_prompt(repo_id, episode_index)
    return parquet_relpath, main_relpath, wrist_relpath, task_prompt


def _load_episode_prompt(repo_id: str, episode_index: int) -> Optional[str]:
    try:
        meta_path = hf_hub_download(
            repo_id,
            filename="meta/episodes.jsonl",
            repo_type="dataset",
        )
    except Exception:  # noqa: BLE001
        return None

    with open(meta_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("episode_index") == episode_index:
                tasks = record.get("tasks") or []
                if isinstance(tasks, list) and tasks:
                    return str(tasks[0])
                break
    return None


def _download_dataset_artifacts(
    repo_id: str,
    episode_index: int,
    image_keys: Sequence[str],
) -> Tuple[Path, dict, Optional[str]]:
    parquet_relpath, main_relpath, wrist_relpath, task_prompt = _episode_file_for(
        repo_id, episode_index
    )
    parquet_path = Path(
        hf_hub_download(repo_id, filename=parquet_relpath, repo_type="dataset")
    )

    videos: dict[str, Optional[Path]] = {}
    key_to_relpath = {
        "observation.images.main": main_relpath,
        "observation.images.secondary_0": wrist_relpath,
    }
    for key in image_keys:
        rel = key_to_relpath.get(key)
        if rel is None:
            videos[key] = None
            continue
        try:
            videos[key] = Path(
                hf_hub_download(repo_id, filename=rel, repo_type="dataset")
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to download video for {key}: {exc}")
            videos[key] = None

    return parquet_path, videos, task_prompt


def _load_episode_table(parquet_path: Path) -> pq.Table:
    return pq.read_table(parquet_path)


def _pick_step(table: pq.Table, step: Optional[int]) -> Tuple[int, dict]:
    num_steps = table.num_rows
    if num_steps == 0:
        raise RuntimeError("Episode table is empty.")
    if step is None:
        step_idx = random.randrange(num_steps)
    else:
        if step < 0 or step >= num_steps:
            raise IndexError(f"Step index {step} out of range (0 <= idx < {num_steps}).")
        step_idx = step

    row = table.slice(step_idx, 1).to_pydict()
    return step_idx, row


def _load_rgb_from_video(video_path: Optional[Path], frame_index: int) -> Optional[np.ndarray]:
    if video_path is None:
        return None
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "OpenCV (cv2) is required for loading video frames. Install with `uv pip install opencv-python`."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"Failed to read frame {frame_index} from video {video_path}"
        )
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def _prepare_observation(
    step_row: dict,
    videos: dict,
    image_keys: Sequence[str],
    prompt: str,
) -> tuple[dict, np.ndarray]:
    state = np.array(step_row["observation.state"][0], dtype=np.float32)
    action = np.array(step_row["action"][0], dtype=np.float32)
    frame_index = int(step_row["frame_index"][0])

    images: list[np.ndarray] = []
    for key in image_keys:
        video_path = videos.get(key)
        if video_path is None:
            continue
        frame = _load_rgb_from_video(video_path, frame_index)
        images.append(frame)

    observation = {
        "images": images,
        "state": state,
        "prompt": prompt,
    }
    return observation, action


def _collect_ground_truth_actions(
    table: pq.Table,
    start_step: int,
    horizon: int,
) -> np.ndarray:
    available = min(horizon, table.num_rows - start_step)
    if available <= 0:
        raise RuntimeError(
            f"Requested horizon extends beyond episode length (start={start_step}, len={table.num_rows})."
        )
    window = table.slice(start_step, available).to_pydict()["action"]
    gt_actions = np.asarray(window, dtype=np.float32)
    return gt_actions


def _query_policy(
    host: str,
    port: int,
    observation: dict,
) -> np.ndarray:
    try:
        from openpi_client import image_tools, websocket_client_policy  # type: ignore
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "openpi_client package is required. Install from https://github.com/phospho-app/openpi.git"
        ) from exc

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port, api_key=None)

    pi_input = {
        "observation/joint_position": observation["state"][:7],
        "observation/gripper_position": observation["state"][-1] if observation["state"].shape[0] > 7 else 0.0,
        "prompt": observation["prompt"],
    }

    image_keys = [
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
    ]

    for idx, image in enumerate(observation["images"]):
        key = image_keys[idx if idx < len(image_keys) else -1]
        processed = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
        pi_input[key] = processed

    response = client.infer(pi_input)
    actions = response.get("actions")
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    return actions


def _plot_actions(
    gt_actions: np.ndarray,
    predicted_actions: np.ndarray,
    output_path: Optional[str],
    metadata: dict,
    show: bool,
) -> None:
    joints = gt_actions.shape[1]
    pred_horizon = predicted_actions.shape[0]
    gt_horizon = gt_actions.shape[0]

    fig, axes = plt.subplots(joints, 1, figsize=(10, 2.5 * joints), sharex=True)
    if joints == 1:
        axes = [axes]

    time_pred = np.arange(pred_horizon)
    time_gt = np.arange(gt_horizon)

    for idx, ax in enumerate(axes):
        ax.plot(time_gt, gt_actions[:, idx], label="Ground truth", marker="o")
        ax.plot(time_pred, predicted_actions[:, idx], label="Policy", linestyle="--")
        ax.set_ylabel(f"Joint {idx}")
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend()

    axes[-1].set_xlabel("Prediction timestep")
    fig.suptitle(
        "Dataset vs. remote policy action trajectories\n"
        f"Episode {metadata['episode_index']} step {metadata['step_index']} | "
        f"Frame {metadata['frame_index']} | Repo {metadata['repo_id']}\n"
        f"Prompt: {metadata.get('prompt', '')}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    args = _parse_args()

    episode_index = args.episode
    if episode_index is None:
        episodes = _list_available_episodes(args.repo_id)
        if not episodes:
            raise RuntimeError(f"No episodes found in dataset {args.repo_id}.")
        episode_index = random.randrange(len(episodes))
        print(f"Randomly selected episode index: {episode_index}")

    parquet_path, videos, task_prompt = _download_dataset_artifacts(
        args.repo_id,
        episode_index,
        args.image_keys,
    )
    print(f"Loaded episode parquet: {parquet_path}")

    table = _load_episode_table(parquet_path)
    step_idx, step_row = _pick_step(table, args.step)
    print(f"Using step index: {step_idx}")

    prompt_text = args.prompt or task_prompt or ""

    observation, gt_action = _prepare_observation(
        step_row,
        videos,
        args.image_keys,
        prompt_text,
    )

    predicted_actions = _query_policy(
        host=args.server_host,
        port=args.server_port,
        observation=observation,
    )

    gt_actions_traj = _collect_ground_truth_actions(
        table=table,
        start_step=step_idx,
        horizon=predicted_actions.shape[0],
    )

    metadata = {
        "repo_id": args.repo_id,
        "episode_index": episode_index,
        "step_index": step_idx,
        "frame_index": int(step_row["frame_index"][0]),
        "prompt": prompt_text,
    }

    show_plot = not args.no_show and args.output is None
    _plot_actions(
        gt_actions=gt_actions_traj,
        predicted_actions=predicted_actions,
        output_path=args.output,
        metadata=metadata,
        show=show_plot,
    )

    diff = predicted_actions[0] - gt_actions_traj[0]
    l2_error = float(np.linalg.norm(diff))
    print(f"First prediction L2 error vs GT: {l2_error:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

