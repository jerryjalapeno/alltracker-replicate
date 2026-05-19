# AllTracker on Replicate

A flexible Cog wrapper around [AllTracker](https://github.com/aharley/alltracker) (Harley et al., ICCV 2025) for dense long-range point tracking.

**Model:** https://replicate.com/jerryjalapeno/alltracker
**Hardware:** L40S (48 GB)
**Inference cost on the zombie test clip (8s, 1080p, 192 frames):** ~6s GPU time at 768px / 83k points.

---

## Quick start

### Web UI

1. Open https://replicate.com/jerryjalapeno/alltracker
2. Upload a video, leave the defaults, hit **Run**.
3. You'll get back an MP4 overlay, a trajectories JSON, an NPZ, a preview frame, and a stats dict.

### Python

```python
import replicate, os
os.environ["REPLICATE_API_TOKEN"] = "r8_…"

out = replicate.run(
    "jerryjalapeno/alltracker",  # uses latest version
    input={
        "video": open("clip.mp4", "rb"),
        "max_frames": 192,
        "resize_to": 768,
        "query_mode": "dense",
        "dense_stride": 2,
        "overlay_style": "dots",
        "background": "dim",
        "output_format": "mp4_overlay",
    },
)
print(out["video"])  # https://replicate.delivery/.../overlay.mp4
```

### curl

```bash
curl -s -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "63bead4682f97e370f62c14726456959bb8f193ab0b9dddeb3875ca6ad139c6e",
    "input": {
      "video": "https://example.com/clip.mp4",
      "query_mode": "grid",
      "grid_size": 60,
      "resize_to": 768
    }
  }'
```

---

## Inputs

### Video & sampling

| Input | Type | Default | Notes |
|---|---|---|---|
| `video` | File | **required** | MP4/MOV/WebM/GIF. URL or upload. |
| `fps` | int | 0 | Resample to N fps before tracking. 0 = keep source. |
| `max_frames` | int | 256 | Hard cap on processed frames (memory guardrail). Up to 2000. |
| `start_frame` | int | 0 | Trim: first frame to include. |
| `end_frame` | int | -1 | Trim: last frame (-1 = end of video). |
| `resize_to` | int | 512 | Long-edge resize before tracking. Rounded to multiple of 8. **0 = native**, max 2048. Bigger = denser pixel lattice and finer flow. |

**Frame budget:** processed frames = `min(max_frames, frames_in(start..end) after fps resample)`. The model uses a sliding window so longer is fine; only memory limits you.

### Query points (what to track)

| Input | Type | Default | Notes |
|---|---|---|---|
| `query_mode` | enum | `grid` | `grid` / `points` / `mask` / `dense` |
| `query_frame` | int | 0 | Seed frame index (negative = from end). |
| `grid_size` | int | 30 | NxN grid (modes: `grid`, `mask`). Up to 120 → 14,400 points. |
| `grid_region` | str | "" | Optional `"x1,y1,x2,y2"` in original-video pixels to confine the grid. |
| `query_points` | str | "" | JSON `[[x,y], …]` in original video pixel coords. Mode: `points`. |
| `query_mask` | File | None | Binary PNG mask (white = track here). Mode: `mask`. |
| `dense_stride` | int | 4 | Pixel stride for dense mode. 1 = every pixel of model-res. Mode: `dense`. |

**Density cheat sheet** (at `resize_to=768`, so model-res ~768×432):

| Mode | Setting | Points |
|---|---|---|
| grid | `grid_size=30` | 900 |
| grid | `grid_size=60` | 3,600 |
| grid | `grid_size=120` | 14,400 |
| dense | `dense_stride=4` | ~21k |
| dense | `dense_stride=2` | ~83k |
| dense | `dense_stride=1` | ~330k (heavy; export NPZ only) |
| dense | `dense_stride=1, resize_to=0` (1080p) | **~2.07M** — every native pixel |

> **"Pixel-level" tracking** = `query_mode=dense, dense_stride=1`. The lattice resolution is `resize_to × (resize_to × aspect)` rounded to multiples of 8. At `resize_to=0` (native) on a 1920×1080 input that's every native pixel.

### Tracking

| Input | Type | Default | Notes |
|---|---|---|---|
| `track_direction` | enum | `bidirectional` | `forward` / `backward` / `bidirectional` (needs `query_frame > 0`). |
| `inference_iters` | int | 4 | Refinement iterations per window. More = better/slower (1–8). |
| `visibility_threshold` | float | 0.1 | Hide track segments with confidence below this. 0..1. |

### Output

| Input | Type | Default | Notes |
|---|---|---|---|
| `output_format` | enum | `all` | `mp4_overlay` / `mp4_sidebyside` / `trajectories_json` / `trajectories_npz` / `all` |
| `overlay_style` | enum | `trails` | `dots` (GPU, fast) / `trails` / `arrows` / `heatmap` |
| `trail_length` | int | 16 | Frames of history drawn behind each point (`trails` style). |
| `point_size` | int | 2 | Dot radius in pixels (1–12). |
| `color_scheme` | enum | `rainbow` | `rainbow` / `motion_direction` / `cluster` / `single` |
| `single_color` | str | `#FF5050` | Hex color for `color_scheme=single`. |
| `cluster_k` | int | 8 | K for `color_scheme=cluster`. |
| `background` | enum | `video` | `video` (50% dim) / `dim` (75% dim) / `black` / `white` |
| `output_fps` | int | 0 | 0 = same as input (after resample). |
| `output_codec` | enum | `h264` | `h264` / `h265` / `vp9` / `prores` |
| `seed` | int | 0 | Random seed (affects cluster init). |

---

## Outputs

Cog returns a dict:

```json
{
  "video": "https://replicate.delivery/.../overlay.mp4",
  "trajectories": "https://replicate.delivery/.../trajectories.json",
  "trajectories_npz": "https://replicate.delivery/.../trajectories.npz",
  "preview_frame": "https://replicate.delivery/.../preview.png",
  "stats": {
    "num_points": 14400,
    "frames_processed": 192,
    "mean_visibility": 0.97,
    "runtime_seconds": 9.0,
    "forward_seconds": 6.1,
    "model_resolution": [432, 768],
    "source_fps": 24.0,
    "effective_fps": 24.0,
    "query_frame": 0
  }
}
```

`trajectories.json` schema:

```json
{
  "frames": 192,
  "points": 14400,
  "fps": 24.0,
  "query_frame": 0,
  "query_points": [[x, y], ...],        // original-pixel coords, length N
  "tracks":      [[[x, y], ...], ...],  // shape T × N × 2, original-pixel coords
  "visibility":  [[v, ...], ...]        // shape T × N, 0..1
}
```

`trajectories.npz` contains the same arrays as float32 (much smaller than JSON for dense outputs).

---

## Recipes

### Maximum density (visualization only)

```python
{
    "query_mode": "dense", "dense_stride": 2,
    "resize_to": 768,
    "overlay_style": "dots", "point_size": 1, "background": "dim",
    "output_format": "mp4_overlay",   # skip JSON — N×T can be enormous
}
```

### True pixel-level tracking (every native pixel)

```python
{
    "resize_to": 0,                   # native; on 1080p input that's 1920x1080
    "query_mode": "dense", "dense_stride": 1,
    "output_format": "trajectories_npz",   # JSON would be many GB
    "overlay_style": "dots", "point_size": 1, "background": "dim",
}
```

Decode the dense field locally:

```python
import numpy as np
d = np.load("trajectories.npz")
T = d["tracks"].shape[0]
H, W = d["query_points"][:, 1].max() + 1, d["query_points"][:, 0].max() + 1  # or read from stats.model_resolution
tracks = d["tracks"].reshape(T, int(H), int(W), 2)        # destination (x,y) per pixel
flow   = tracks - d["query_points"].reshape(int(H), int(W), 2)[None]  # displacement per pixel
vis    = d["visibility"].reshape(T, int(H), int(W))
```

### Sparse grid with trajectories for downstream use

```python
{
    "query_mode": "grid", "grid_size": 40,
    "resize_to": 512,
    "overlay_style": "trails", "trail_length": 20,
    "output_format": "trajectories_npz",
}
```

### Track a single object (clicked points)

```python
{
    "query_mode": "points",
    "query_points": "[[820, 540], [890, 600], [780, 720]]",
    "query_frame": 0,
    "overlay_style": "trails", "trail_length": 32,
    "color_scheme": "single", "single_color": "#00FFAA",
    "background": "video",
}
```

### Track only a region of interest

```python
{
    "query_mode": "grid", "grid_size": 50,
    "grid_region": "640,360,1280,720",      # x1,y1,x2,y2 in original pixels
    "overlay_style": "dots",
}
```

### Track inside a mask

```python
{
    "query_mode": "mask",
    "query_mask": open("foreground_mask.png", "rb"),
    "grid_size": 40,
}
```

### Side-by-side comparison

```python
{
    "output_format": "mp4_sidebyside",   # input | overlay
    "overlay_style": "trails",
    "background": "black",
}
```

### Track backward from end of clip

```python
{
    "query_frame": -1,                # last frame
    "track_direction": "backward",
}
```

### Bidirectional from a chosen middle frame

```python
{
    "query_frame": 64,
    "track_direction": "bidirectional",
}
```

---

## Performance

L40S timings observed on the 8-second 1080p test clip:

| Resolution | Frames | Points (dense_stride=2) | Forward (s) | Total (warm, s) |
|---|---|---|---|---|
| 512 | 128 | ~37k | ~1.5 | ~5 |
| 768 | 128 | ~83k | ~4.4 | ~7 |
| 768 | 192 | ~83k | ~6.1 | ~9 |
| 1024 | 128 | ~147k | ~10 | ~15 |
| 1920 (native) | 192 | ~518k (stride=2) | ~30–40 | ~50 |

Cold start (container boot + weights load): **~60–95s** on the first request after idle. After warm, each prediction reuses the loaded model.

**Memory:** dominant cost is the dense flow field, `B × T × 2 × H × W × 4 bytes`. At 192 frames × 768×432 that's ~250 MB. Fits comfortably in 48 GB up to ~1024px / 500 frames.

---

## Notes & gotchas

- **JSON output at dense settings is large.** 83k points × 192 frames × 2 floats × ~15 chars = ~480 MB JSON. Prefer NPZ (~50× smaller). For visualization-only runs set `output_format=mp4_overlay`.
- **`dots` is the fast path.** Rendered on GPU via scatter-add (ported from the demo). `trails`, `arrows`, and `heatmap` run on CPU; at >20k points these slow the render step significantly.
- **Coordinates in outputs are in original-video pixel space.** Internal model coords (resized) are not exposed.
- **`bidirectional` with `query_frame=0` just runs forward** — there's nothing behind frame 0 to track.
- **The `mp4_overlay` background colors are pre-baked into the video.** Re-render if you change your mind.
- **GIF inputs:** treated as videos; framerate detected from the file.

---

## Iterating on the model

The Cog project lives at `~/Desktop/alltracker-cog/`. To ship a new version:

```bash
cd ~/Desktop/alltracker-cog
# edit predict.py / cog.yaml
REPLICATE_API_TOKEN=r8_… cog push r8.im/jerryjalapeno/alltracker
```

Cached layers make rebuilds ~30s when only `predict.py` changes. The model weights are baked into the image at build time (`cog.yaml: build.run`), so cold starts don't re-download them.
