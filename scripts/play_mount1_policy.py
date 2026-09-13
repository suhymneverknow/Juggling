"""Play a JAX checkpoint with the MuJoCo interactive viewer."""

from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.mount1_env import Mount1Env  # noqa: E402
from train.train_mount1 import ActorCritic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    with args.checkpoint.expanduser().open("rb") as stream:
        checkpoint = pickle.load(stream)

    config = argparse.Namespace(**checkpoint["training_config"])
    env = Mount1Env(config)
    model = ActorCritic(checkpoint["act_dim"])
    normalizer = checkpoint["obs_norm"]
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    # Compile and execute reset before opening the viewer.  Otherwise MuJoCo's
    # XML qpos0 is displayed during JAX compilation instead of the actual
    # 15-degree-board IK state used by training.
    initial_state = reset(jax.random.PRNGKey(args.seed))
    # Compile the first physics/policy transition before opening the window as
    # well.  Discard this probe so playback still starts from the untouched
    # reset state instead of appearing frozen while XLA compiles ``step``.
    step(initial_state, jnp.zeros((checkpoint["act_dim"],))).reward.block_until_ready()
    viewer = None
    renderer = None
    video_process = None
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, -0.18, 0.60]
    camera.distance = 2.5
    camera.azimuth = -135
    camera.elevation = -18
    render_data = mujoco.MjData(env.cpu_model)
    render_data.qpos[:] = np.asarray(initial_state.data.qpos)
    render_data.qvel[:] = np.asarray(initial_state.data.qvel)
    mujoco.mj_forward(env.cpu_model, render_data)
    if args.video is not None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to save an MP4")
        args.video = args.video.expanduser().resolve()
        args.video.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(env.cpu_model, args.height, args.width)
        video_process = subprocess.Popen(
            [
                ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{args.width}x{args.height}", "-pix_fmt", "rgb24",
                "-r", str(args.fps), "-i", "-", "-an", "-vcodec", "libx264",
                "-pix_fmt", "yuv420p", str(args.video),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    elif not args.no_render:
        viewer = mujoco.viewer.launch_passive(env.cpu_model, render_data)
        viewer.cam.lookat[:] = [0.0, -0.18, 0.60]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = -135
        viewer.cam.elevation = -18

    try:
        for episode in range(args.episodes):
            state = (
                initial_state
                if episode == 0
                else reset(jax.random.PRNGKey(args.seed + episode))
            )
            if viewer is not None:
                render_data.qpos[:] = np.asarray(state.data.qpos)
                render_data.qvel[:] = np.asarray(state.data.qvel)
                mujoco.mj_forward(env.cpu_model, render_data)
                viewer.sync()
            if renderer is not None:
                renderer.update_scene(render_data, camera=camera)
                frame = renderer.render()
                assert video_process is not None and video_process.stdin is not None
                video_process.stdin.write(frame.tobytes())
            episode_return = 0.0
            while not bool(np.asarray(state.done)):
                start = time.time()
                obs = np.asarray(state.obs)
                normalized = np.clip(
                    (obs - normalizer["mean"]) / np.sqrt(normalizer["var"] + 1e-8),
                    -10.0,
                    10.0,
                )
                mean, _, _ = model.apply(
                    checkpoint["params"], jnp.asarray(normalized)[None]
                )
                action = jnp.tanh(mean[0])
                state = step(state, action)
                episode_return += float(np.asarray(state.reward))

                if viewer is not None:
                    render_data.qpos[:] = np.asarray(state.data.qpos)
                    render_data.qvel[:] = np.asarray(state.data.qvel)
                    mujoco.mj_forward(env.cpu_model, render_data)
                    viewer.sync()
                    time.sleep(max(0.0, env.control_dt - (time.time() - start)))
                elif renderer is not None:
                    render_data.qpos[:] = np.asarray(state.data.qpos)
                    render_data.qvel[:] = np.asarray(state.data.qvel)
                    mujoco.mj_forward(env.cpu_model, render_data)
                    renderer.update_scene(render_data, camera=camera)
                    frame = renderer.render()
                    assert video_process is not None and video_process.stdin is not None
                    video_process.stdin.write(frame.tobytes())

            metrics = state.metrics
            print(
                f"episode={episode + 1} "
                f"success={bool(np.asarray(metrics['is_success']))} "
                f"failure={bool(np.asarray(metrics['physical_failure']))} "
                f"timeout={bool(np.asarray(metrics['timeout']))} "
                f"time={float(np.asarray(metrics['episode_seconds'])):.2f}s "
                f"reward={episode_return:.2f}"
            )
    finally:
        if viewer is not None:
            viewer.close()
        if renderer is not None:
            renderer.close()
        if video_process is not None:
            assert video_process.stdin is not None
            video_process.stdin.close()
            error_output = video_process.stderr.read().decode("utf-8", errors="replace")
            return_code = video_process.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {error_output}")
            print(f"video={args.video}")


if __name__ == "__main__":
    main()
