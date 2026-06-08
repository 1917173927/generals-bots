"""Generate a tiled animated GIF of parallel Generals training rollouts.

Usage:
    uv run --with pillow python examples/generate_parallel_training_gif.py
    uv run --with pillow --with imageio --with imageio-ffmpeg python \
        examples/generate_parallel_training_gif.py --video-output generals/assets/videos/parallel_training_process.mp4

The project runtime does not depend on Pillow; it is only needed to produce this
presentation asset. Defaults render a 60 second loop at 0.8x playback speed.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generals import GeneralsEnv, get_observation
from generals.agents import ExpanderAgent, RandomAgent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "generals" / "assets" / "gifs" / "parallel_training_process.gif"
DEFAULT_VIDEO_OUTPUT = ROOT / "generals" / "assets" / "videos" / "parallel_training_process.mp4"
FONT_DIR = ROOT / "generals" / "assets" / "fonts"


CANVAS_SIZE = (1120, 790)
NUM_ENVS = 12
GRID_DIMS = (8, 8)
TOTAL_DURATION_MS = 60_000
PLAYBACK_SPEED = 0.8
BASE_FRAME_DURATION_MS = 90
FRAME_DURATION_MS = BASE_FRAME_DURATION_MS / PLAYBACK_SPEED
FRAMES = round(TOTAL_DURATION_MS / FRAME_DURATION_MS)
GIF_DURATION_QUANTUM_MS = 10
STEPS_PER_FRAME = 3
POLICY_UPDATE_EVERY = 180

BG = (27, 25, 22)
PANEL = (244, 239, 227)
PANEL_DARK = (55, 48, 42)
INK = (36, 31, 27)
MUTED = (121, 112, 99)
GRID_LINE = (45, 41, 37)
P0 = (42, 176, 161)
P0_DARK = (21, 105, 99)
P1 = (226, 92, 87)
P1_DARK = (145, 50, 49)
NEUTRAL = (226, 221, 207)
MOUNTAIN = (82, 80, 73)
CITY = (221, 164, 68)
GOLD = (244, 189, 76)
WHITE = (255, 255, 250)


@dataclass
class Metrics:
    global_step: int = 0
    episodes: int = 0
    p0_wins: int = 0
    p1_wins: int = 0
    win_rates: list[float] = field(default_factory=list)
    avg_land_gap: float = 0.0
    avg_army_gap: float = 0.0

    @property
    def decisive_games(self) -> int:
        return self.p0_wins + self.p1_wins

    @property
    def win_rate(self) -> float:
        if self.decisive_games == 0:
            return 0.5
        return self.p0_wins / self.decisive_games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Animated GIF destination.")
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help=f"Optional MP4 video destination, e.g. {DEFAULT_VIDEO_OUTPUT}.",
    )
    parser.add_argument("--preview", type=Path, default=None, help="Optional PNG preview of the last frame.")
    parser.add_argument("--seed", type=int, default=17, help="JAX PRNG seed.")
    return parser.parse_args()


def frame_durations() -> list[int]:
    """Return integer per-frame durations that sum exactly to the target length."""
    base_duration = (TOTAL_DURATION_MS // FRAMES // GIF_DURATION_QUANTUM_MS) * GIF_DURATION_QUANTUM_MS
    extra_units = (TOTAL_DURATION_MS - base_duration * FRAMES) // GIF_DURATION_QUANTUM_MS
    return [
        base_duration + (GIF_DURATION_QUANTUM_MS if frame_idx < extra_units else 0)
        for frame_idx in range(FRAMES)
    ]


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_name = "Quicksand-Bold.ttf" if bold else "Quicksand-SemiBold.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / font_name), size=size)
    except OSError:
        return ImageFont.load_default()


def mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = INK,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def rounded_bar(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: float,
    fg: tuple[int, int, int],
    bg: tuple[int, int, int] = (218, 210, 194),
    *,
    radius: int = 5,
) -> None:
    value = max(0.0, min(1.0, value))
    draw.rounded_rectangle(box, radius=radius, fill=bg)
    x0, y0, x1, y1 = box
    if value > 0:
        draw.rounded_rectangle((x0, y0, x0 + int((x1 - x0) * value), y1), radius=radius, fill=fg)


def draw_sparkline(
    draw: ImageDraw.ImageDraw,
    values: list[float],
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=6, fill=(238, 229, 210), outline=(210, 198, 178), width=1)
    if len(values) < 2:
        return
    recent = values[-36:]
    points = []
    for i, value in enumerate(recent):
        x = x0 + 8 + int((x1 - x0 - 16) * i / max(1, len(recent) - 1))
        y = y1 - 8 - int((y1 - y0 - 16) * max(0.0, min(1.0, value)))
        points.append((x, y))
    draw.line(points, fill=color, width=3, joint="curve")
    draw.ellipse((points[-1][0] - 4, points[-1][1] - 4, points[-1][0] + 4, points[-1][1] + 4), fill=color)


def draw_header(draw: ImageDraw.ImageDraw, metrics: Metrics, policy_strength: float) -> None:
    title_font = load_font(38, bold=True)
    label_font = load_font(14)
    metric_font = load_font(24, bold=True)
    small_font = load_font(12)

    draw_text(draw, (34, 24), "Parallel Policy Training", title_font, WHITE)
    draw_text(
        draw,
        (36, 66),
        "12 simultaneous rollouts   |   0.8x playback   |   60s training loop   |   optimizer pulse",
        label_font,
        (222, 211, 190),
    )

    cards = [
        ("env steps", f"{metrics.global_step:04d}", P0),
        ("policy mix", f"{policy_strength * 100:02.0f}%", GOLD),
        ("episodes", f"{metrics.episodes:03d}", (196, 144, 218)),
        ("win rate", f"{metrics.win_rate * 100:02.0f}%", P1),
    ]
    x = 620
    for label, value, color in cards:
        box = (x, 24, x + 112, 84)
        draw.rounded_rectangle(box, radius=8, fill=(48, 43, 37), outline=mix(color, WHITE, 0.25), width=2)
        draw_text(draw, (x + 12, 35), label.upper(), small_font, (202, 191, 170))
        draw_text(draw, (x + 12, 52), value, metric_font, color)
        x += 122

    timeline = (36, 104, 1084, 114)
    draw.rounded_rectangle(timeline, radius=5, fill=(67, 60, 52))
    progress = metrics.global_step / max(1, FRAMES * STEPS_PER_FRAME)
    rounded_bar(draw, timeline, progress, P0, (67, 60, 52), radius=5)
    for update_step in range(POLICY_UPDATE_EVERY, FRAMES * STEPS_PER_FRAME + 1, POLICY_UPDATE_EVERY):
        x_tick = timeline[0] + int((timeline[2] - timeline[0]) * update_step / (FRAMES * STEPS_PER_FRAME))
        pulse = 1.0 - ((metrics.global_step - update_step) % POLICY_UPDATE_EVERY) / POLICY_UPDATE_EVERY
        marker = 4 + int(4 * max(0.0, pulse if metrics.global_step >= update_step else 0.0))
        draw.ellipse((x_tick - marker, 101 - marker, x_tick + marker, 101 + marker), fill=GOLD)
    draw_text(draw, (36, 122), "rollout batch timeline", small_font, (213, 201, 181))


def action_endpoint(row: int, col: int, direction: int) -> tuple[int, int]:
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dr, dc = offsets[int(direction) % 4]
    return row + dr, col + dc


def draw_city(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int) -> None:
    half = size // 2
    draw.rounded_rectangle((cx - half, cy - half + 2, cx + half, cy + half), radius=3, fill=CITY, outline=(102, 76, 32))
    draw.rectangle((cx - half + 3, cy - half - 3, cx + half - 3, cy - half + 3), fill=mix(CITY, WHITE, 0.12))
    draw.rectangle((cx - 2, cy, cx + 2, cy + half), fill=(120, 82, 32))


def draw_mountain(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int) -> None:
    half = size // 2
    points = [(cx - half, cy + half), (cx, cy - half), (cx + half, cy + half)]
    draw.polygon(points, fill=MOUNTAIN, outline=(44, 43, 39))
    draw.polygon([(cx - 2, cy - half + 2), (cx + 4, cy + 4), (cx + half - 2, cy + half)], fill=(102, 100, 92))


def draw_crown(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, fill: tuple[int, int, int]) -> None:
    half = size // 2
    points = [
        (cx - half, cy + half),
        (cx - half + 2, cy - 1),
        (cx - 4, cy + 2),
        (cx, cy - half),
        (cx + 4, cy + 2),
        (cx + half - 2, cy - 1),
        (cx + half, cy + half),
    ]
    draw.polygon(points, fill=fill, outline=(33, 30, 26))
    draw.rectangle((cx - half + 1, cy + half - 3, cx + half - 1, cy + half), fill=fill, outline=(33, 30, 26))


def draw_board(
    draw: ImageDraw.ImageDraw,
    state_np: dict[str, np.ndarray],
    actions_np: np.ndarray,
    env_idx: int,
    board_box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = board_box
    armies = state_np["armies"][env_idx]
    ownership = state_np["ownership"][env_idx]
    mountains = state_np["mountains"][env_idx]
    cities = state_np["cities"][env_idx]
    generals = state_np["generals"][env_idx]
    cell = min((x1 - x0) // GRID_DIMS[1], (y1 - y0) // GRID_DIMS[0])
    board_size = cell * GRID_DIMS[0]
    x0 += ((x1 - x0) - board_size) // 2
    y0 += ((y1 - y0) - board_size) // 2

    draw.rounded_rectangle((x0 - 5, y0 - 5, x0 + board_size + 5, y0 + board_size + 5), radius=7, fill=(203, 193, 174))

    for row in range(GRID_DIMS[0]):
        for col in range(GRID_DIMS[1]):
            px = x0 + col * cell
            py = y0 + row * cell
            army = int(armies[row, col])
            if bool(mountains[row, col]):
                color = MOUNTAIN
            elif bool(ownership[0, row, col]):
                color = mix(P0, WHITE, 0.22 - min(0.20, math.log1p(max(army, 0)) / 32))
            elif bool(ownership[1, row, col]):
                color = mix(P1, WHITE, 0.20 - min(0.18, math.log1p(max(army, 0)) / 32))
            elif bool(cities[row, col]):
                color = mix(CITY, NEUTRAL, 0.18)
            else:
                color = NEUTRAL
            draw.rectangle((px, py, px + cell - 1, py + cell - 1), fill=color, outline=GRID_LINE)

            cx, cy = px + cell // 2, py + cell // 2
            if bool(mountains[row, col]):
                draw_mountain(draw, cx, cy + 1, max(7, cell - 6))
            elif bool(cities[row, col]):
                draw_city(draw, cx, cy + 1, max(7, cell - 7))
            if bool(generals[row, col]):
                crown_color = GOLD if bool(ownership[0, row, col]) else (255, 223, 145)
                draw_crown(draw, cx, cy, max(9, cell - 5), crown_color)

    tiny_font = load_font(9, bold=True)
    for row in range(GRID_DIMS[0]):
        for col in range(GRID_DIMS[1]):
            army = int(armies[row, col])
            if army <= 1 or bool(mountains[row, col]):
                continue
            px = x0 + col * cell
            py = y0 + row * cell
            text = str(min(army, 99))
            fill = WHITE if bool(ownership[:, row, col].any()) else INK
            draw_text(draw, (px + cell - 2, py + cell - 1), text, tiny_font, fill, anchor="rb")

    for player, color in [(0, P0_DARK), (1, P1_DARK)]:
        action = actions_np[env_idx, player]
        if int(action[0]) != 0:
            continue
        row, col, direction = int(action[1]), int(action[2]), int(action[3])
        target_row, target_col = action_endpoint(row, col, direction)
        if not (0 <= row < GRID_DIMS[0] and 0 <= col < GRID_DIMS[1]):
            continue
        if not (0 <= target_row < GRID_DIMS[0] and 0 <= target_col < GRID_DIMS[1]):
            continue
        start = (x0 + col * cell + cell // 2, y0 + row * cell + cell // 2)
        end = (x0 + target_col * cell + cell // 2, y0 + target_row * cell + cell // 2)
        draw.line((start, end), fill=color, width=3)
        draw.ellipse((end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3), fill=color)


def draw_env_panel(
    draw: ImageDraw.ImageDraw,
    state_np: dict[str, np.ndarray],
    info_np: dict[str, np.ndarray],
    actions_np: np.ndarray,
    done_np: np.ndarray,
    env_idx: int,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    small_font = load_font(12)
    label_font = load_font(15, bold=True)
    value_font = load_font(13, bold=True)

    draw.rounded_rectangle((x0 + 4, y0 + 5, x1 + 4, y1 + 5), radius=8, fill=(12, 11, 10))
    outline = GOLD if bool(done_np[env_idx]) else (210, 198, 178)
    draw.rounded_rectangle(box, radius=8, fill=PANEL, outline=outline, width=2)

    time_value = int(state_np["time"][env_idx])
    draw_text(draw, (x0 + 14, y0 + 11), f"env {env_idx + 1:02d}", label_font, INK)
    draw_text(draw, (x1 - 14, y0 + 12), f"t={time_value:03d}", small_font, MUTED, anchor="ra")

    board_box = (x0 + 15, y0 + 34, x1 - 15, y1 - 40)
    draw_board(draw, state_np, actions_np, env_idx, board_box)

    land = info_np["land"][env_idx]
    army = info_np["army"][env_idx]
    land_total = max(1, int(land[0] + land[1]))
    army_total = max(1, int(army[0] + army[1]))
    land_share = float(land[0] / land_total)
    army_share = float(army[0] / army_total)

    bar_y = y1 - 31
    draw_text(draw, (x0 + 14, bar_y - 1), "land", small_font, MUTED)
    rounded_bar(draw, (x0 + 52, bar_y, x1 - 86, bar_y + 8), land_share, P0, P1, radius=4)
    draw_text(draw, (x1 - 14, bar_y - 4), f"{int(land[0])}:{int(land[1])}", value_font, INK, anchor="ra")

    bar_y += 16
    draw_text(draw, (x0 + 14, bar_y - 1), "army", small_font, MUTED)
    rounded_bar(draw, (x0 + 52, bar_y, x1 - 86, bar_y + 8), army_share, P0_DARK, P1_DARK, radius=4)
    draw_text(draw, (x1 - 14, bar_y - 4), f"{int(army[0])}:{int(army[1])}", value_font, INK, anchor="ra")


def render_frame(
    state_np: dict[str, np.ndarray],
    info_np: dict[str, np.ndarray],
    actions_np: np.ndarray,
    done_np: np.ndarray,
    metrics: Metrics,
    policy_strength: float,
) -> Image.Image:
    image = Image.new("RGB", CANVAS_SIZE, BG)
    draw = ImageDraw.Draw(image)
    draw_header(draw, metrics, policy_strength)

    left, top = 34, 150
    gap_x, gap_y = 18, 18
    cols, rows = 4, 3
    panel_w = (CANVAS_SIZE[0] - 2 * left - gap_x * (cols - 1)) // cols
    panel_h = (CANVAS_SIZE[1] - top - 34 - gap_y * (rows - 1)) // rows

    for env_idx in range(NUM_ENVS):
        row, col = divmod(env_idx, cols)
        x0 = left + col * (panel_w + gap_x)
        y0 = top + row * (panel_h + gap_y)
        draw_env_panel(
            draw,
            state_np,
            info_np,
            actions_np,
            done_np,
            env_idx,
            (x0, y0, x0 + panel_w, y0 + panel_h),
        )

    spark_box = (804, 120, 1084, 140)
    draw_sparkline(draw, metrics.win_rates, spark_box, GOLD)
    draw_text(draw, (804, 143), "recent decisive win-rate signal", load_font(11), (213, 201, 181))
    return image


def batched_snapshot(states, timestep, actions: jnp.ndarray, done: jnp.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    state_np = {
        "armies": np.array(states.armies),
        "ownership": np.array(states.ownership),
        "mountains": np.array(states.mountains),
        "cities": np.array(states.cities),
        "generals": np.array(states.generals),
        "time": np.array(states.time),
    }
    info_np = {
        "land": np.array(timestep.info.land),
        "army": np.array(timestep.info.army),
    }
    return state_np, info_np, np.array(actions), np.array(done)


def update_metrics(metrics: Metrics, timestep) -> None:
    done = np.array(timestep.terminated | timestep.truncated)
    winners = np.array(timestep.info.winner)
    land = np.array(timestep.info.land)
    army = np.array(timestep.info.army)

    metrics.episodes += int(done.sum())
    metrics.p0_wins += int(np.logical_and(done, winners == 0).sum())
    metrics.p1_wins += int(np.logical_and(done, winners == 1).sum())
    metrics.avg_land_gap = float(np.mean(land[:, 0] - land[:, 1]))
    metrics.avg_army_gap = float(np.mean(army[:, 0] - army[:, 1]))
    metrics.win_rates.append(metrics.win_rate)


def build_animation(seed: int) -> list[Image.Image]:
    env = GeneralsEnv(
        grid_dims=GRID_DIMS,
        truncation=150,
        mountain_density_range=(0.12, 0.22),
        num_cities_range=(3, 6),
        min_generals_distance=4,
        max_generals_distance=7,
        pool_size=512,
    )
    learner_expert = ExpanderAgent(id="Policy")
    learner_noise = RandomAgent(id="Explorer", split_prob=0.35, idle_prob=0.08)
    opponent = RandomAgent(id="Opponent", split_prob=0.28, idle_prob=0.06)

    key = jrandom.PRNGKey(seed)
    key, pool_key, init_key = jrandom.split(key, 3)
    pool, _ = env.reset(pool_key)
    states = jax.vmap(env.init_state)(jrandom.split(init_key, NUM_ENVS))

    step_vmap = jax.vmap(lambda state, action: env.step(state, action, pool))
    get_obs_p0 = jax.vmap(lambda state: get_observation(state, 0))
    get_obs_p1 = jax.vmap(lambda state: get_observation(state, 1))
    act_expert = jax.vmap(learner_expert.act)
    act_noise = jax.vmap(learner_noise.act)
    act_opponent = jax.vmap(opponent.act)

    metrics = Metrics()
    frames: list[Image.Image] = []
    last_timestep = None
    last_actions = jnp.zeros((NUM_ENVS, 2, 5), dtype=jnp.int32)
    last_done = jnp.zeros((NUM_ENVS,), dtype=bool)

    for frame_idx in range(FRAMES):
        policy_strength = 0.18 + 0.78 * (frame_idx / max(1, FRAMES - 1)) ** 0.75
        for _ in range(STEPS_PER_FRAME):
            obs_p0 = get_obs_p0(states)
            obs_p1 = get_obs_p1(states)
            key, k_expert, k_noise, k_opp, k_mix = jrandom.split(key, 5)
            expert_actions = act_expert(obs_p0, jrandom.split(k_expert, NUM_ENVS))
            noise_actions = act_noise(obs_p0, jrandom.split(k_noise, NUM_ENVS))
            use_expert = jrandom.uniform(k_mix, (NUM_ENVS,)) < policy_strength
            actions_p0 = jnp.where(use_expert[:, None], expert_actions, noise_actions)
            actions_p1 = act_opponent(obs_p1, jrandom.split(k_opp, NUM_ENVS))
            actions = jnp.stack([actions_p0, actions_p1], axis=1)
            timestep, states = step_vmap(states, actions)
            metrics.global_step += 1
            update_metrics(metrics, timestep)
            last_timestep = timestep
            last_actions = actions
            last_done = timestep.terminated | timestep.truncated

        assert last_timestep is not None
        snapshot = batched_snapshot(states, last_timestep, last_actions, last_done)
        frames.append(render_frame(*snapshot, metrics=metrics, policy_strength=policy_strength))

    return frames


def save_video(frames: list[Image.Image], output: Path) -> None:
    """Write an MP4 version with the same visual timeline as the GIF."""
    import imageio.v2 as imageio

    fps = len(frames) / (TOTAL_DURATION_MS / 1000)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output,
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=2,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))


def main() -> None:
    args = parse_args()
    frames = build_animation(args.seed)
    durations = frame_durations()
    if sum(durations) != TOTAL_DURATION_MS:
        raise RuntimeError(f"Frame durations sum to {sum(durations)}ms, expected {TOTAL_DURATION_MS}ms")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=True,
    )
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        frames[-1].save(args.preview)
    if args.video_output:
        save_video(frames, args.video_output)
        print(f"Wrote {args.video_output}")
    print(f"Wrote {args.output}")
    print(
        f"Frames: {len(frames)}, size: {CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}, "
        f"speed: {PLAYBACK_SPEED}x, total duration: {sum(durations)}ms"
    )


if __name__ == "__main__":
    main()
