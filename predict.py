"""Replicate Cog predictor for AllTracker (https://github.com/aharley/alltracker).

Exposes a flexible point-tracking interface: dense grid, user points, or mask
ROI, with multiple rendering styles and output formats.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path

# Repo-local imports (AllTracker)
import utils.basic
import utils.improc
from nets.alltracker import Net


WEIGHTS_PATH = "/src/weights/alltracker.pth"
WEIGHTS_URL = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"


# ---------- Video I/O ----------

def read_video(path: str) -> tuple[list[np.ndarray], float]:
    """Read all frames of a video as RGB uint8 numpy arrays."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Video {path} has zero readable frames")
    return frames, float(fps)


def resample_fps(frames: list[np.ndarray], src_fps: float, target_fps: int) -> tuple[list[np.ndarray], float]:
    if target_fps <= 0 or abs(src_fps - target_fps) < 1e-3:
        return frames, src_fps
    n_src = len(frames)
    duration = n_src / src_fps
    n_out = max(1, int(round(duration * target_fps)))
    idx = np.linspace(0, n_src - 1, n_out).round().astype(int)
    return [frames[i] for i in idx], float(target_fps)


def write_video_ffmpeg(frames: np.ndarray, fps: float, out_path: str, codec: str = "h264") -> None:
    """frames: (T,H,W,3) uint8. Writes via ffmpeg for codec flexibility."""
    T, H, W, _ = frames.shape
    codec_map = {
        "h264": ["-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p"],
        "h265": ["-c:v", "libx265", "-crf", "24", "-pix_fmt", "yuv420p", "-tag:v", "hvc1"],
        "vp9":  ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0"],
        "prores": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
    }
    if codec not in codec_map:
        raise ValueError(f"Unsupported codec: {codec}")
    # Pad to even dims for h264/h265
    if codec in ("h264", "h265") and (H % 2 or W % 2):
        pad_h = H + (H % 2)
        pad_w = W + (W % 2)
        padded = np.zeros((T, pad_h, pad_w, 3), dtype=frames.dtype)
        padded[:, :H, :W] = frames
        frames = padded
        H, W = pad_h, pad_w

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps:.6f}",
        "-i", "-",
        *codec_map[codec],
        out_path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(frames.tobytes())
    p.stdin.close()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed (exit {rc}) writing {out_path}")


# ---------- Query sampling ----------

def make_grid_points(H: int, W: int, grid_size: int, region: Optional[tuple[int, int, int, int]] = None) -> np.ndarray:
    """Return (N,2) float32 grid points (x,y) within optional region (x1,y1,x2,y2)."""
    if region is None:
        x1, y1, x2, y2 = 0, 0, W, H
    else:
        x1, y1, x2, y2 = region
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"grid_region is empty after clipping: {region}")
    g = max(2, int(grid_size))
    xs = np.linspace(x1 + 0.5, x2 - 0.5, g)
    ys = np.linspace(y1 + 0.5, y2 - 0.5, g)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)


def points_from_mask(mask_path: str, target_hw: tuple[int, int], grid_size: int) -> np.ndarray:
    """Sample grid points constrained to a binary mask."""
    H, W = target_hw
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask > 127
    if not mask_bool.any():
        raise ValueError("Mask is empty (no pixels above 127)")
    cand = make_grid_points(H, W, grid_size)
    cand_int = cand.round().astype(int)
    cand_int[:, 0] = cand_int[:, 0].clip(0, W - 1)
    cand_int[:, 1] = cand_int[:, 1].clip(0, H - 1)
    keep = mask_bool[cand_int[:, 1], cand_int[:, 0]]
    pts = cand[keep]
    if len(pts) == 0:
        # fall back to a denser sweep
        ys, xs = np.where(mask_bool)
        if len(xs) > 4096:
            idx = np.linspace(0, len(xs) - 1, 4096).astype(int)
            xs, ys = xs[idx], ys[idx]
        pts = np.stack([xs, ys], axis=1).astype(np.float32) + 0.5
    return pts


def parse_query_points(json_str: str, scale_xy: tuple[float, float]) -> np.ndarray:
    pts = json.loads(json_str)
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"query_points JSON must be shape (N,2), got {arr.shape}")
    sx, sy = scale_xy
    arr[:, 0] *= sx
    arr[:, 1] *= sy
    return arr


# ---------- Color schemes ----------

def colors_rainbow(query_points: np.ndarray, H: int, W: int) -> np.ndarray:
    """RGB colors in [0,255] by 2D position — what demo.py uses."""
    return utils.improc.get_2d_colors(query_points, H, W).astype(np.float32)


def colors_single(n: int, rgb: tuple[int, int, int] = (255, 80, 80)) -> np.ndarray:
    return np.tile(np.array(rgb, dtype=np.float32), (n, 1))


def colors_motion(trajs: np.ndarray) -> np.ndarray:
    """Color by mean motion direction across the clip. trajs: (T,N,2)."""
    if trajs.shape[0] < 2:
        return colors_single(trajs.shape[1])
    delta = trajs[-1] - trajs[0]  # (N,2)
    angle = (np.arctan2(delta[:, 1], delta[:, 0]) + np.pi) / (2 * np.pi)  # 0..1
    mag = np.linalg.norm(delta, axis=1)
    mag = mag / (mag.max() + 1e-6)
    hsv = np.stack([angle * 179, np.full_like(angle, 220), 80 + mag * 175], axis=1).astype(np.uint8)[None]
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0].astype(np.float32)
    return rgb


def colors_cluster(trajs: np.ndarray, k: int = 8) -> np.ndarray:
    """K-means cluster on full trajectories. trajs: (T,N,2)."""
    from sklearn.cluster import KMeans
    T, N, _ = trajs.shape
    feats = trajs.transpose(1, 0, 2).reshape(N, -1)
    k = min(k, max(1, N))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(feats)
    palette_hsv = np.stack([(np.arange(k) * (179.0 / k)).astype(np.uint8),
                            np.full(k, 220, dtype=np.uint8),
                            np.full(k, 230, dtype=np.uint8)], axis=1)[None]
    palette = cv2.cvtColor(palette_hsv, cv2.COLOR_HSV2RGB)[0].astype(np.float32)
    return palette[km.labels_]


# ---------- Rendering ----------

def batched_flow2color(flow: torch.Tensor) -> np.ndarray:
    """Batched HSV-encoded optical flow viz. flow: (T,2,H,W) on CUDA. Returns (T,H,W,3) uint8.
    Uses a single global clip across all frames so colors are stable through the clip.
    """
    with torch.no_grad():
        T, _, H, W = flow.shape
        clip = flow.abs().max().clamp(min=1e-6)
        f = (flow / clip).clamp(-1, 1)
        radius = torch.sqrt(f[:, 0] ** 2 + f[:, 1] ** 2).clamp(0, 1)        # (T,H,W) value
        angle = torch.atan2(-f[:, 1], -f[:, 0]) / np.pi                     # (T,H,W) -1..1
        h = ((angle + 1.0) / 2.0).clamp(0, 1)                                # 0..1
        s = torch.full_like(h, 0.75)
        v = radius
        # HSV→RGB (numpy/matplotlib algorithm), fully vectorized
        i = (h * 6.0).floor()
        frac = h * 6.0 - i
        i = i.long() % 6
        p = v * (1.0 - s)
        q = v * (1.0 - frac * s)
        t = v * (1.0 - (1.0 - frac) * s)
        # build per-sextant RGB by gather
        # cases: 0:(v,t,p) 1:(q,v,p) 2:(p,v,t) 3:(p,q,v) 4:(t,p,v) 5:(v,p,q)
        r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p,
            torch.where(i == 3, p, torch.where(i == 4, t, v)))))
        g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v,
            torch.where(i == 3, q, torch.where(i == 4, p, p)))))
        b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t,
            torch.where(i == 3, v, torch.where(i == 4, v, q)))))
        rgb = torch.stack([r, g, b], dim=-1) * 255.0   # (T,H,W,3)
        return rgb.clamp(0, 255).byte().cpu().numpy()


def render_dots_gpu(
    rgbs: torch.Tensor,           # (T,3,H,W) float 0..255 on CUDA
    trajs: torch.Tensor,          # (T,N,2) on CUDA
    visibs: torch.Tensor,         # (T,N) bool on CUDA
    colors: torch.Tensor,         # (N,3) 0..255 float on CUDA
    point_size: int,
    background: str,
) -> np.ndarray:
    """Vectorized GPU dots renderer (port of demo.draw_pts_gpu). Returns (T,H,W,3) uint8 RGB."""
    device = rgbs.device
    T, C, H, W = rgbs.shape

    if background == "video":
        bkg_opacity, sat_boost = 0.5, False
    elif background == "dim":
        bkg_opacity, sat_boost = 0.25, False
    elif background == "black":
        bkg_opacity, sat_boost = 0.0, True
    elif background == "white":
        bkg_opacity, sat_boost = 1.0, False
    else:
        bkg_opacity, sat_boost = 0.5, False

    if background == "white":
        rgbs = torch.full_like(rgbs, 255.0)
    else:
        rgbs = (rgbs * bkg_opacity).clamp(0, 255)

    radius = max(1, int(point_size))
    sharpness = 0.15 + 0.05 * np.log2(max(1, radius))
    opacity = 0.9 if radius == 1 else 1.0

    D = radius * 2 + 1
    y = torch.arange(D, device=device).float()[:, None] - radius
    x = torch.arange(D, device=device).float()[None, :] - radius
    dist2 = x ** 2 + y ** 2
    icon = torch.clamp(1 - (dist2 - (radius ** 2) / 2.0) / (radius * 2 * sharpness), 0, 1)
    icon = icon.view(1, D, D)
    dx = torch.arange(-radius, radius + 1, device=device)
    dy = torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")

    # transpose to (N,T,2) / (N,T) like the demo
    trajs_nt = trajs.permute(1, 0, 2)
    visibs_nt = visibs.permute(1, 0)

    for t in range(T):
        mask = visibs_nt[:, t]
        if not mask.any():
            continue
        xy = trajs_nt[mask, t] + 0.5
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)
        colors_now = colors[mask]
        Nt = xy.shape[0]
        cx = xy[:, 0].long()
        cy = xy[:, 1].long()
        x_grid = cx[:, None, None] + disp_x
        y_grid = cy[:, None, None] + disp_y
        valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
        x_valid = x_grid[valid]
        y_valid = y_grid[valid]
        icon_weights = icon.expand(Nt, D, D)[valid]
        colors_valid = colors_now[:, :, None, None].expand(Nt, 3, D, D).permute(1, 0, 2, 3)[:, valid]
        idx_flat = (y_valid * W + x_valid).long()

        accum = torch.zeros_like(rgbs[t])
        weight = torch.zeros(1, H * W, device=device)
        img_flat = accum.view(C, -1)
        weighted_colors = colors_valid * icon_weights
        img_flat.scatter_add_(1, idx_flat.unsqueeze(0).expand(C, -1), weighted_colors)
        weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
        weight = weight.view(1, H, W)

        alpha = weight.clamp(0, 1) * opacity
        accum = accum / (weight + 1e-6)
        rgbs[t] = rgbs[t] * (1 - alpha) + accum * alpha

    out = rgbs.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    if sat_boost:
        for t in range(T):
            hsv = cv2.cvtColor(out[t], cv2.COLOR_RGB2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)
            out[t] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return out


def render_overlay(
    rgbs: torch.Tensor,           # (T,3,H,W) float, 0..255 on GPU
    trajs: np.ndarray,            # (T,N,2)
    visibs: np.ndarray,           # (T,N) bool
    colors: np.ndarray,           # (N,3) 0..255 float
    style: str,
    trail_length: int,
    point_size: int,
    background: str,
) -> np.ndarray:
    """Renders frames as (T,H,W,3) uint8."""
    T, _, H, W = rgbs.shape
    device = rgbs.device

    if background == "video":
        bkg = (rgbs * 0.5).clamp(0, 255)
    elif background == "dim":
        bkg = (rgbs * 0.25).clamp(0, 255)
    elif background == "black":
        bkg = torch.zeros_like(rgbs)
    elif background == "white":
        bkg = torch.full_like(rgbs, 255.0)
    else:
        bkg = rgbs.clone()

    bkg_np = bkg.byte().permute(0, 2, 3, 1).cpu().numpy()  # (T,H,W,3) uint8
    out = bkg_np.copy()

    radius = max(1, int(point_size))
    line_thick = max(1, radius - 1)
    colors_int = colors.astype(np.int32)

    N = trajs.shape[1]

    for t in range(T):
        frame = out[t]
        vis_now = visibs[t]
        pts_now = trajs[t]

        if style == "trails":
            t0 = max(0, t - trail_length + 1)
            for i in range(N):
                seg_vis = visibs[t0:t + 1, i]
                seg_pts = trajs[t0:t + 1, i]
                if seg_vis.sum() < 2:
                    if vis_now[i]:
                        cv2.circle(frame, tuple(pts_now[i].round().astype(int)), radius,
                                   tuple(int(c) for c in colors_int[i]), -1, lineType=cv2.LINE_AA)
                    continue
                pts_xy = seg_pts.round().astype(np.int32)
                col = tuple(int(c) for c in colors_int[i])
                for j in range(1, len(pts_xy)):
                    if seg_vis[j - 1] and seg_vis[j]:
                        cv2.line(frame, tuple(pts_xy[j - 1]), tuple(pts_xy[j]), col, line_thick, cv2.LINE_AA)
                if vis_now[i]:
                    cv2.circle(frame, tuple(pts_xy[-1]), radius, col, -1, lineType=cv2.LINE_AA)

        elif style == "arrows":
            prev_t = max(0, t - 1)
            for i in range(N):
                if not vis_now[i]:
                    continue
                p_now = tuple(pts_now[i].round().astype(int))
                col = tuple(int(c) for c in colors_int[i])
                if t > 0 and visibs[prev_t, i]:
                    p_prev = tuple(trajs[prev_t, i].round().astype(int))
                    cv2.arrowedLine(frame, p_prev, p_now, col, line_thick, cv2.LINE_AA, tipLength=0.4)
                else:
                    cv2.circle(frame, p_now, radius, col, -1, lineType=cv2.LINE_AA)

        elif style == "heatmap":
            # color by per-frame visibility — fully visible = bright point, low = faint
            for i in range(N):
                if pts_now[i, 0] < 0 or pts_now[i, 0] >= W or pts_now[i, 1] < 0 or pts_now[i, 1] >= H:
                    continue
                v = 1.0 if vis_now[i] else 0.2
                col = tuple(int(c * v) for c in colors_int[i])
                cv2.circle(frame, tuple(pts_now[i].round().astype(int)), radius, col, -1, lineType=cv2.LINE_AA)

        else:  # "dots"
            for i in range(N):
                if not vis_now[i]:
                    continue
                cv2.circle(frame, tuple(pts_now[i].round().astype(int)), radius,
                           tuple(int(c) for c in colors_int[i]), -1, lineType=cv2.LINE_AA)

        out[t] = frame

    return out


# ---------- Predictor ----------

class Predictor(BasePredictor):
    def setup(self):
        if not os.path.exists(WEIGHTS_PATH):
            os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
            print(f"weights not found at {WEIGHTS_PATH}; downloading…")
            torch.hub.download_url_to_file(WEIGHTS_URL, WEIGHTS_PATH, progress=False)
        sd = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
        self.model = Net(seqlen=16)
        self.model.load_state_dict(sd["model"], strict=True)
        self.model.cuda().eval()
        for p in self.model.parameters():
            p.requires_grad = False
        torch.set_grad_enabled(False)

        print("AllTracker loaded.")

    def predict(
        self,
        video: Path = Input(description="Input video (mp4/mov/webm/gif)."),
        fps: int = Input(
            description="Resample input to this fps before tracking (0 = keep source).",
            default=0, ge=0, le=120,
        ),
        max_frames: int = Input(
            description="Hard cap on frames after fps resample (memory guardrail). A100-80GB handles 1000+ frames at 768px comfortably.",
            default=512, ge=2, le=2000,
        ),
        start_frame: int = Input(description="Trim: start frame (0-indexed).", default=0, ge=0),
        end_frame: int = Input(description="Trim: end frame (-1 = last).", default=-1, ge=-1),
        resize_to: int = Input(
            description="Long-edge resize before tracking (0 = native, rounded to multiple of 8).",
            default=512, ge=0, le=2048,
        ),
        query_mode: str = Input(
            description="How to pick query points. 'dense' uses every dense_stride-th pixel of the model-resolution flow field for maximum density.",
            choices=["grid", "points", "mask", "dense"], default="grid",
        ),
        query_frame: int = Input(
            description="Seed frame for queries (negative = from end).",
            default=0,
        ),
        grid_size: int = Input(
            description="NxN grid size (used for grid/mask modes).",
            default=30, ge=2, le=120,
        ),
        dense_stride: int = Input(
            description="Pixel stride for query_mode=dense (1 = every pixel, 2 = every other, etc.). Smaller = denser but more memory and slower JSON export.",
            default=4, ge=1, le=32,
        ),
        grid_region: str = Input(
            description="Optional 'x1,y1,x2,y2' (in original video pixels) to confine grid. Empty = full frame.",
            default="",
        ),
        query_points: str = Input(
            description='JSON [[x,y],...] in original video pixel coords. Used when query_mode=points.',
            default="",
        ),
        query_mask: Path = Input(
            description="Binary PNG mask (white=track here). Used when query_mode=mask.",
            default=None,
        ),
        track_direction: str = Input(
            description="forward / backward / bidirectional (the last needs query_frame > 0).",
            choices=["forward", "backward", "bidirectional"], default="bidirectional",
        ),
        inference_iters: int = Input(
            description="Model refinement iterations per window (more = better, slower).",
            default=4, ge=1, le=8,
        ),
        visibility_threshold: float = Input(
            description="Hide track segments with confidence below this.",
            default=0.1, ge=0.0, le=1.0,
        ),
        output_format: str = Input(
            description="What to return. 'flow_video' is HSV-encoded optical-flow visualization; 'flow_npz' is the raw dense flow tensor (T,2,H,W) at model resolution.",
            choices=["mp4_overlay", "mp4_sidebyside", "mp4_flow", "trajectories_json", "trajectories_npz", "flow_npz", "all"],
            default="all",
        ),
        overlay_style: str = Input(
            description="How to draw points.",
            choices=["dots", "trails", "arrows", "heatmap"], default="trails",
        ),
        trail_length: int = Input(
            description="Frames of history drawn behind each point (style=trails).",
            default=16, ge=1, le=120,
        ),
        point_size: int = Input(description="Dot radius (px).", default=2, ge=1, le=12),
        color_scheme: str = Input(
            description="Point coloring.",
            choices=["rainbow", "motion_direction", "cluster", "single"], default="rainbow",
        ),
        single_color: str = Input(
            description="Hex color for color_scheme=single (e.g. #FF5050).",
            default="#FF5050",
        ),
        cluster_k: int = Input(description="Clusters for color_scheme=cluster.", default=8, ge=2, le=32),
        background: str = Input(
            description="Background under the points.",
            choices=["video", "dim", "black", "white"], default="video",
        ),
        output_fps: int = Input(description="Output fps (0 = same as input).", default=0, ge=0, le=120),
        output_codec: str = Input(
            description="Video codec.",
            choices=["h264", "h265", "vp9", "prores"], default="h264",
        ),
        seed: int = Input(description="Random seed (used for cluster init etc.).", default=0),
        precision: str = Input(
            description="Inference precision. bf16 is ~1.5x faster on A100 with negligible accuracy impact for this model; fp32 is the safe default.",
            choices=["fp32", "bf16"], default="fp32",
        ),
    ) -> dict:
        t_start = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ---- Read & preprocess video ----
        frames_rgb, src_fps = read_video(str(video))
        n_orig = len(frames_rgb)

        # trim
        if end_frame == -1 or end_frame > n_orig:
            end_frame = n_orig
        if start_frame >= end_frame:
            raise ValueError(f"start_frame ({start_frame}) >= end_frame ({end_frame})")
        frames_rgb = frames_rgb[start_frame:end_frame]

        # fps resample
        frames_rgb, eff_fps = resample_fps(frames_rgb, src_fps, fps)

        # cap length
        if len(frames_rgb) > max_frames:
            frames_rgb = frames_rgb[:max_frames]

        if len(frames_rgb) < 2:
            raise ValueError("Need at least 2 frames after trimming/resampling.")

        H0, W0 = frames_rgb[0].shape[:2]

        # resize (long edge)
        if resize_to > 0:
            scale = resize_to / max(H0, W0)
        else:
            scale = 1.0
        H = max(8, int(round(H0 * scale)) // 8 * 8)
        W = max(8, int(round(W0 * scale)) // 8 * 8)
        scale_x = W / W0
        scale_y = H / H0
        frames_resized = [cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in frames_rgb]

        T = len(frames_resized)

        # resolve query_frame (negative-indexed)
        qf = query_frame if query_frame >= 0 else T + query_frame
        qf = max(0, min(T - 1, qf))

        # ---- Build query points (in model-space pixel coords) ----
        if query_mode == "grid":
            region = None
            if grid_region.strip():
                try:
                    parts = [int(float(p)) for p in grid_region.replace(";", ",").split(",")]
                    if len(parts) != 4:
                        raise ValueError
                    x1, y1, x2, y2 = parts
                    region = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
                except Exception as e:
                    raise ValueError(f"Bad grid_region '{grid_region}': expected 'x1,y1,x2,y2'") from e
            q_points = make_grid_points(H, W, grid_size, region)
        elif query_mode == "points":
            if not query_points.strip():
                raise ValueError("query_points JSON is empty but query_mode=points.")
            q_points = parse_query_points(query_points, (scale_x, scale_y))
            q_points[:, 0] = q_points[:, 0].clip(0, W - 1)
            q_points[:, 1] = q_points[:, 1].clip(0, H - 1)
        elif query_mode == "mask":
            if query_mask is None:
                raise ValueError("query_mask file required when query_mode=mask.")
            q_points = points_from_mask(str(query_mask), (H, W), grid_size)
        elif query_mode == "dense":
            s = max(1, int(dense_stride))
            xs = np.arange(s // 2, W, s, dtype=np.float32) + 0.5
            ys = np.arange(s // 2, H, s, dtype=np.float32) + 0.5
            xx, yy = np.meshgrid(xs, ys, indexing="xy")
            q_points = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unknown query_mode: {query_mode}")

        N = q_points.shape[0]
        if N == 0:
            raise ValueError("No query points after sampling.")
        print(f"[alltracker] T={T} H={H} W={W} N={N} query_frame={qf} direction={track_direction}")

        # ---- Run model ----
        rgbs_t = torch.from_numpy(np.stack(frames_resized)).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()  # (1,T,3,H,W)

        grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device="cuda").float()
        grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)

        t_fwd = time.time()
        traj_maps_full, visconf_maps_full = self._run_directional(
            rgbs_t, qf, track_direction, grid_xy, inference_iters, precision=precision
        )
        fwd_time = time.time() - t_fwd
        print(f"[alltracker] forward: {fwd_time:.1f}s for {T} frames")

        # traj_maps_full: (1,T,2,H,W) — for every frame, dense (x,y) sampling
        # visconf_maps_full: (1,T,2,H,W) — channels: visibility, confidence
        # Sample at query pixel locations using grid_sample
        with torch.no_grad():
            traj_maps = traj_maps_full[0]  # (T,2,H,W)
            visconf_maps = visconf_maps_full[0]  # (T,2,H,W)

            # normalize query coords to [-1, 1] for grid_sample
            xs = (q_points[:, 0] / max(1, W - 1)) * 2 - 1
            ys = (q_points[:, 1] / max(1, H - 1)) * 2 - 1
            sample_grid = torch.from_numpy(np.stack([xs, ys], axis=1)).float().cuda()  # (N,2)
            sample_grid = sample_grid.view(1, N, 1, 2).expand(T, -1, -1, -1)  # (T,N,1,2)

            trajs_sampled = torch.nn.functional.grid_sample(
                traj_maps, sample_grid, mode="bilinear", align_corners=True
            )  # (T,2,N,1)
            visconf_sampled = torch.nn.functional.grid_sample(
                visconf_maps, sample_grid, mode="bilinear", align_corners=True
            )  # (T,2,N,1)

            trajs_np = trajs_sampled.squeeze(-1).permute(0, 2, 1).cpu().numpy()  # (T,N,2)
            visconf_np = visconf_sampled.squeeze(-1).permute(0, 2, 1).cpu().numpy()  # (T,N,2)

        # Visibility = channel 1 of visconf (matches demo: visconfs_e[..., 1])
        vis_score = visconf_np[..., 1]  # (T,N)
        visibs = vis_score > visibility_threshold  # (T,N) bool

        # ---- Colors ----
        if color_scheme == "rainbow":
            colors = colors_rainbow(q_points, H, W)
        elif color_scheme == "single":
            try:
                rgb = tuple(int(single_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
            except Exception:
                rgb = (255, 80, 80)
            colors = colors_single(N, rgb)
        elif color_scheme == "motion_direction":
            colors = colors_motion(trajs_np)
        elif color_scheme == "cluster":
            colors = colors_cluster(trajs_np, k=cluster_k)
        else:
            colors = colors_rainbow(q_points, H, W)

        # ---- Render outputs ----
        out_fps = output_fps if output_fps > 0 else eff_fps
        tmpdir = Path(tempfile.mkdtemp(prefix="alltracker_"))
        outputs: dict = {
            "video": None,
            "trajectories": None,
            "preview_frame": None,
            "stats": {},
        }

        # Always produce preview frame of query points
        preview = frames_resized[qf].copy()
        for i, (x, y) in enumerate(q_points.round().astype(int)):
            col = tuple(int(c) for c in colors[i])
            cv2.circle(preview, (int(x), int(y)), max(2, point_size + 1), col, -1, lineType=cv2.LINE_AA)
        preview_path = tmpdir / "preview.png"
        cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        outputs["preview_frame"] = Path(preview_path)

        want_overlay = output_format in ("mp4_overlay", "mp4_sidebyside", "all")
        want_json = output_format in ("trajectories_json", "all")
        want_npz = output_format in ("trajectories_npz", "all")
        want_flow_video = output_format in ("mp4_flow", "all")
        want_flow_npz = output_format in ("flow_npz", "all")

        # Guardrail: trajectories.json is O(N*T) chars and json.dump is single-threaded.
        # Auto-skip when the user implicitly requested 'all' but N*T is huge.
        N_T = N * T
        JSON_LIMIT = 5_000_000
        if want_json and output_format == "all" and N_T > JSON_LIMIT:
            print(f"[alltracker] skipping trajectories_json (N*T={N_T:,} > {JSON_LIMIT:,}); NPZ still written")
            want_json = False

        # Prepare per-output payloads (CPU-side work shared across outputs)
        traj_orig = None
        q_points_orig = None
        if want_json or want_npz:
            traj_orig = trajs_np.copy()
            traj_orig[..., 0] /= scale_x
            traj_orig[..., 1] /= scale_y
            q_points_orig = q_points.copy()
            q_points_orig[:, 0] /= scale_x
            q_points_orig[:, 1] /= scale_y

        # ---- Render overlay video (GPU/CPU, frame buffer in numpy) ----
        if want_overlay:
            if overlay_style == "dots":
                trajs_gpu = torch.from_numpy(trajs_np).float().cuda()
                visibs_gpu = torch.from_numpy(visibs).cuda()
                colors_gpu = torch.from_numpy(colors).float().cuda()
                frames_drawn = render_dots_gpu(
                    rgbs_t[0], trajs_gpu, visibs_gpu, colors_gpu,
                    point_size=point_size, background=background,
                )
            else:
                frames_drawn = render_overlay(
                    rgbs_t[0], trajs_np, visibs, colors,
                    style=overlay_style, trail_length=trail_length,
                    point_size=point_size, background=background,
                )
            if output_format == "mp4_sidebyside":
                orig_np = rgbs_t[0].byte().permute(0, 2, 3, 1).cpu().numpy()
                frames_drawn = np.concatenate([orig_np, frames_drawn], axis=2)

        # ---- Pre-compute flow tensors on GPU once for both flow outputs ----
        flow_orig_np = None
        flow_vis_np = None
        flow_rgb_frames = None
        if want_flow_npz or want_flow_video:
            with torch.no_grad():
                flow_model = (traj_maps_full[0] - grid_xy[0])              # (T,2,H,W) model-pixel
                if want_flow_video:
                    flow_rgb_frames = batched_flow2color(flow_model)        # (T,H,W,3) uint8
                if want_flow_npz:
                    flow_orig_x = flow_model[:, 0:1] / scale_x
                    flow_orig_y = flow_model[:, 1:2] / scale_y
                    flow_orig = torch.cat([flow_orig_x, flow_orig_y], dim=1)
                    flow_orig_np = flow_orig.cpu().numpy().astype(np.float32)
                    flow_vis_np = visconf_maps_full[0, :, 1].cpu().numpy().astype(np.float32)

        # ---- Parallel disk writes ----
        # Each task does file I/O (json.dump, np.savez, ffmpeg pipe) — overlapping
        # them with threads saves wall-clock time since they release the GIL during I/O.
        def _write_json():
            p = tmpdir / "trajectories.json"
            with open(p, "w") as f:
                json.dump({
                    "frames": int(T), "points": int(N), "fps": float(eff_fps),
                    "query_frame": int(qf),
                    "query_points": q_points_orig.tolist(),
                    "tracks": traj_orig.tolist(),
                    "visibility": vis_score.tolist(),
                }, f)
            return ("trajectories", Path(p))

        def _write_traj_npz():
            p = tmpdir / "trajectories.npz"
            np.savez_compressed(  # tracks are sparse + smooth → compression wins big
                p,
                tracks=traj_orig.astype(np.float32),
                visibility=vis_score.astype(np.float32),
                query_points=q_points_orig.astype(np.float32),
                query_frame=np.int32(qf),
                fps=np.float32(eff_fps),
            )
            return ("trajectories_npz", Path(p))

        def _write_flow_npz():
            p = tmpdir / "flow.npz"
            np.savez(           # uncompressed: dense per-pixel flow is too varied for zlib to help much
                p,
                flow=flow_orig_np,                  # (T,2,H,W) original-pixel displacement
                visibility=flow_vis_np,             # (T,H,W)
                query_frame=np.int32(qf),
                model_resolution=np.int32([H, W]),
                fps=np.float32(eff_fps),
            )
            return ("flow_npz", Path(p))

        def _write_overlay_mp4():
            p = tmpdir / "overlay.mp4"
            write_video_ffmpeg(frames_drawn, out_fps, str(p), output_codec)
            return ("video", Path(p))

        def _write_flow_mp4():
            p = tmpdir / "flow.mp4"
            write_video_ffmpeg(flow_rgb_frames, out_fps, str(p), output_codec)
            return ("flow_video", Path(p))

        jobs = []
        if want_overlay:    jobs.append(_write_overlay_mp4)
        if want_flow_video: jobs.append(_write_flow_mp4)
        if want_json:       jobs.append(_write_json)
        if want_npz:        jobs.append(_write_traj_npz)
        if want_flow_npz:   jobs.append(_write_flow_npz)

        if jobs:
            t_w = time.time()
            with ThreadPoolExecutor(max_workers=min(5, len(jobs))) as ex:
                for fut in [ex.submit(j) for j in jobs]:
                    key, path = fut.result()
                    outputs[key] = path
            print(f"[alltracker] output writes: {time.time()-t_w:.2f}s ({len(jobs)} files)")

        runtime = time.time() - t_start
        outputs["stats"] = {
            "num_points": int(N),
            "frames_processed": int(T),
            "mean_visibility": float(vis_score.mean()),
            "runtime_seconds": round(runtime, 2),
            "forward_seconds": round(fwd_time, 2),
            "model_resolution": [int(H), int(W)],
            "source_fps": float(src_fps),
            "effective_fps": float(eff_fps),
            "query_frame": int(qf),
        }
        return outputs

    # ---- Internal helpers ----

    def _run_directional(
        self,
        rgbs_t: torch.Tensor,         # (1,T,3,H,W)
        qf: int,
        direction: str,
        grid_xy: torch.Tensor,        # (1,1,2,H,W)
        iters: int,
        precision: str = "fp32",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward / backward / bidirectional sliding inference.
        Returns (traj_maps, visconf_maps) each shape (1,T,2,H,W) in float32."""
        B, T, C, H, W = rgbs_t.shape
        device = rgbs_t.device

        traj_maps_full = torch.zeros(B, T, 2, H, W, device=device)
        visconf_maps_full = torch.zeros(B, T, 2, H, W, device=device)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16"
            else torch.autocast(device_type="cuda", enabled=False)
        )

        # Forward portion (qf..end) — always anchored at qf
        if direction in ("forward", "bidirectional") and qf < T - 1:
            with autocast_ctx:
                fwd_flows, fwd_vc, _, _ = self.model.forward_sliding(
                    rgbs_t[:, qf:], iters=iters, sw=None, is_training=False
                )
            fwd_traj = fwd_flows.float().to(device) + grid_xy
            traj_maps_full[:, qf:] = fwd_traj
            visconf_maps_full[:, qf:] = fwd_vc.float().to(device)
        else:
            traj_maps_full[:, qf:qf + 1] = grid_xy
            visconf_maps_full[:, qf:qf + 1] = 1.0

        if direction in ("backward", "bidirectional") and qf > 0:
            with autocast_ctx:
                bwd_flows, bwd_vc, _, _ = self.model.forward_sliding(
                    rgbs_t[:, :qf + 1].flip([1]), iters=iters, sw=None, is_training=False
                )
            bwd_traj = bwd_flows.float().to(device) + grid_xy
            bwd_traj = bwd_traj.flip([1])[:, :-1]
            bwd_vc = bwd_vc.float().to(device).flip([1])[:, :-1]
            traj_maps_full[:, :qf] = bwd_traj
            visconf_maps_full[:, :qf] = bwd_vc
        elif direction == "forward" and qf > 0:
            traj_maps_full[:, :qf] = grid_xy
            visconf_maps_full[:, :qf] = 0.0

        return traj_maps_full, visconf_maps_full
