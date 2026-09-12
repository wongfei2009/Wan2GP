# Adding MiniMax H3 still-image / reference-editing mode to WanGP

## What the ComfyUI pack actually does

`astropuzzo/ComfyUI-MiniMax-H3-Image-Studio` does **not** load a different
checkpoint. Its own README says it plainly:

> "MiniMax H3 is an audio-video model. These nodes generate a short frame
> packet, decode it, and select one still image."

- Frame packet: **5 frames** by default (profiles 1–20)
- Decoded with the ordinary `minimax_h3_video_vae_fp16` — an "Exact Frame
  Decode" node then picks one frame out of the batch
- Audio is skipped: *"The audio VAE is not required for image output."*
- Reference editing is just Ref2VA with ordered `<Picture N>` references
  ("Keep the identity, face, hair, clothing, camera and environment from
  `<Picture 1>`"), plus a `source_fidelity` strength

There is an optional dedicated image VAE (`minimax_h3_image_vae_step1597`) for
1-frame decode, but that is an optimisation, not the mechanism.

## Why this is small in WanGP

WanGP **already has** the generic "video model emits a still" path, and it is
the same trick:

| piece | where |
|---|---|
| `v2i_switch_supported: True` in `model_def` → Text-to-Image tab, `image_mode=1` | `wgp.py:11487`, `:11504` |
| image mode forces a short generation | `wgp.py:6880-6888` |
| the frame-packet dropdown ("generate N frames, keep the first") | `wgp.py:12401-12418` |
| PNG/JPG save branch, audio ignored | `wgp.py:8307-8314` |
| `image+video` in the model listing | `models/model_metadata.py:95` |

And the **frame-packet sizes are derived from the model's own grid**
(`wgp.py:12405-12409`):

```
temporal_latent  = latent_size            # H3: 17
frames_offset    = model_def["frames_offset"]   # H3: 5
first_frame_count = frames_offset if frames_offset > 1 else temporal_latent + frames_offset
frame_counts      = [first + 17*i for i in range(4)]
```

For H3 that is **[5, 22, 39, 56]**. But see the VAE floor below: **5 does not
decode**, so the usable sizes start at 22.

The H3 pipeline also already supports 5 frames natively
(`models/minimax_h3/pipeline.py:62`):

```python
def video_latent_frames(frame_count):
    frame_count = normalize_frame_count(max(5, int(frame_count)), 5, 17, 5)
    return 2 + ((frame_count - 5) // 17) * 5
```

`video_latent_frames(5) == 2`. The `17n+5, min 107` rule is a **model_def /
planner** constraint, not a pipeline one.

### The VAE floor — why the packet is 22 frames, not 5

The denoiser happily produces a 2-latent-frame result, and then the video VAE
cannot decode it:

```
RuntimeError: MiniMax H3 VAE decoded 0 frames, expected 5
```

`components/video_autoencoder.py` decodes in chunks with `clip_length 17`,
`token_drop 3` and temporal ratio 4, so `tokens_chunk_size = 5` and

```
num_tokens = latent_frames + token_drop
num_chunks = (num_tokens + pad) // tokens_chunk_size - int(token_drop > 0)
```

With 2 latent frames that is `(2+3+0)//5 - 1 = 0` chunks — nothing is written,
and the write-position check raises. The decoder needs **>= 7 latent frames**,
and `video_latent_frames(22) == 7` is the next size on the 17n+5 grid:

```
F= 5  latent=2  chunks=0  -> FAILS
F=22  latent=7  chunks=1  -> OK
F=39  latent=12 chunks=2  -> OK
```

So `frames_minimum_image` is **22**. This is presumably why the ComfyUI pack
ships an optional dedicated *image* VAE for its 1-frame profiles — the video
VAE cannot go that short. Taking its "5 frames by default" at face value is
what produced this failure.

## The four changes

### 1. Declare it — `models/minimax_h3/minimax_h3_handler.py` (base def, ~line 404)

```python
"v2i_switch_supported": True,
"frames_minimum_image": 5,     # consumed by change 2
"image_batch_size_max": 4,
```

On Ref2VA also set `image_video_prompt_type` so switching to the image tab adds
the reference letters itself (`wgp.py:10208` reads it, default `"KI"`).

### 2. Stop the 107 floor clobbering the packet — `wgp.py:7026`

This is the one real blocker, and it bites in **two** places, which the first
attempt at this patch missed. Today:

```python
video_length = floor_frame_count(video_length, frames_minimum, latent_size, frames_offset)
```

and `floor_frame_count` opens with `frame_count = max(minimum, frame_count)`, so
`floor_frame_count(5, 107, 17, 5) == 107` — every "still" would silently be a
4.5-second video. Fix generically (no H3 hardcoding):

```python
frames_offset = model_def.get("frames_offset", 1)
floor_minimum = model_def.get("frames_minimum_image", frames_minimum) if is_image else frames_minimum
video_length = floor_frame_count(video_length, floor_minimum, latent_size, frames_offset)
```

Every other model keeps today's behaviour (the key is absent → `frames_minimum`).

**Patching only `floor_frame_count` is not enough.** The sliding-window plan
(`build_default_window_plan`, `wgp.py:7342`) takes its own `minimum=frames_minimum`
and re-floors the packet, which the job events report as:

```
Requested frame contribution adjusted from 5 to 107 for model-compatible scheduling (Sliding Window 1)
```

— i.e. `success=true` and a 4.5-second video where a still was asked for. So the
rebinding happens **once, at the assignment** (`wgp.py:7026`), and every
downstream floor and window plan inherits it. Verified offline against the real
scheduler:

```
build_default_window_plan(total_frames=5, window_size=362, step=17, frame_offset=5, ...)
  minimum=107 -> [(frame_num 107, output 107, requested 5)]
  minimum=5   -> [(frame_num 5,   output 5,   requested None)]
```

### 3. Trim to one frame and drop audio — `models/minimax_h3/pipeline.py`

`image_mode` already arrives: `wgp.py:8006` passes it and `generate()` has
`**kwargs` (`pipeline.py:660`). Mirror LTX-2's three lines
(`models/ltx2/ltx2.py:1918-1920`) just before the return at `pipeline.py:1476`:

```python
image_mode = int(kwargs.get("image_mode", 0) or 0)
...
if image_mode > 0:
    return {"x": decoded_video[:, :1]}
```

Omitting the `audio` key is enough — the `elif is_image:` save branch never
looks at it.

The audio VAE is skipped by returning before it — worth doing, since it is a
full VAE pass for 0.2 s of audio that is thrown away.

### 4. Keep the native fps — `models/minimax_h3/pipeline.py`

**This one is not optional and is easy to miss.** `wgp.py:7131` sets
`fps = 1 if is_image else …` and passes it straight to the model. H3 conditions
the DiT on fps (`payload["fps"]`) *and* sizes the audio latent block from it
(`audio_t = round(frames / fps * AUDIO_LATENT_FPS)`), so `fps=1` would be far
out of distribution and would allocate a 24x oversized audio block. In still
mode the pipeline pins `fps = H3_NATIVE_FPS` (24.0).

LTX-2 gets away without this because it does not condition on fps the same way
— do not read its silence as "fps does not matter".

## What lights up for free

- `wangp models` flips the H3 rows to `image+video` — `infer_main_outputs`
  reads `v2i_switch_supported` (`models/model_metadata.py:95`). No CLI change.
- `wangp generate --model minimax_h3_ref2va_pruned --image-ref face.png
  --ref-as-subject --set image_mode=1 --set min_frames_if_references=5` reaches
  it through the existing `--set` escape hatch.

## Test plan (CLAUDE.md A/B rule)

1. `--dry-run` and confirm `image_mode=1` and the video_length that survives.
2. **Plain 20-step `minimax_h3_ref2va_pruned` first, not PDD** — a 5-frame
   packet is out of distribution for an 8-step distillation; test PDD second.
   One phase.
3. Confirm a JPG lands in the images dir, no audio error, and note wall time.
4. A/B with vs. without `--image-ref`: `success=true` says nothing about
   whether the reference conditioned anything.
5. Try packet 1 vs 5 vs 22, and first frame vs a middle frame — the video VAE
   is causal, so frame 0 may be the softest one in the packet. Don't assume.

## Open questions

- Why is `frames_minimum` 107? If the sliding-window planner or VDN needs it,
  the image-mode guard above keeps video untouched either way.
- The dedicated image VAE is worth a look only if the 5-frame packet proves
  slow or soft.
- Deploy: the fork runs on the Windows box; this needs a pull + MCP server
  restart, and the box is single-session.
