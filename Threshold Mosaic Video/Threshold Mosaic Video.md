# Threshold Mosaic

# Intro

For film to read clearly, the eye needs anchors—regions of stillness against which motion can be understood. A mosaic made from images contains vastly more information than the footage beneath it; each tile is its own tiny world. As Walter Benjamin wrote, “Every image is the bearer of a message.” But the viewer needs time to receive that message. If tiles changed every frame, their meaning would disappear before the nervous system could grasp it.

By letting tiles remain fixed in still regions and only shifting them when the video content moves, rather than small lighting or texture changes, the system gives the viewer time to inhabit each fragment. Stillness becomes a stage for attention; big deliberate movement becomes a disruption that writes a second story over the first.

This produces layered meaning inside a single visual space. The still tiles tell one story through their embedded imagery. The motion tells another through rhythm and alteration. With the complete piece becomes the conversation between the two. A negotiation between what remains and what is constantly becoming.



# Starting with the code from the image handing class:

```python
from PIL import Image
import numpy as np, os

INPUT_IMAGE   = "/Users/stonesavage/Desktop/Coding for media/Code/image.png"
DATASET_DIR   = "/Users/stonesavage/Desktop/Coding for media/Code/images"
OUTPUT_IMAGE  = "/Users/stonesavage/Desktop/Coding for media/Code/Mosaic.png"

TILE_INPUT_SIZE  = 32      
TILE_OUTPUT_SIZE = 128    

UPSCALE = 2               

# LOAD IMAGES
main_img = Image.open(INPUT_IMAGE).convert("RGB")

dataset_tiles = [
    Image.open(os.path.join(DATASET_DIR, f)).convert("RGB")
    for f in os.listdir(DATASET_DIR)
    if f.lower().endswith((".jpg", ".png"))]


# Colour Av
def tile_average(im):
    return np.array(im).mean(axis=(0,1))

dataset_avgs = [tile_average(t) for t in dataset_tiles]

# Tile match
def best_tile_for(col):
    dists = [np.linalg.norm(a - col) for a in dataset_avgs]
    i = np.argmin(dists)
    return dataset_tiles[i].resize((TILE_OUTPUT_SIZE, TILE_OUTPUT_SIZE))


# Build mosaic
W, H = main_img.size
GRID_W, GRID_H = W // TILE_INPUT_SIZE, H // TILE_INPUT_SIZE

small = main_img.resize((GRID_W, GRID_H))
pix = np.array(small)

mosaic = Image.new(
    "RGB",
    (GRID_W * TILE_OUTPUT_SIZE, GRID_H * TILE_OUTPUT_SIZE))

print("Building mosaic...")
for y in range(GRID_H):
    for x in range(GRID_W):
        mosaic.paste(
            best_tile_for(pix[y, x]),
            (x * TILE_OUTPUT_SIZE, y * TILE_OUTPUT_SIZE))
print("Mosaic complete.")

# Upsalce 
if UPSCALE > 1:
    mosaic = mosaic.resize(
        (mosaic.width * UPSCALE, mosaic.height * UPSCALE),
        Image.LANCZOS)

mosaic.save(OUTPUT_IMAGE)
print("Saved high-resolution mosaic:", OUTPUT_IMAGE)
```

![image](RatMosaic2.png)


# Then turning that in to a video system:

The static image mosaic system works by reducing the source image to a low-resolution grid derived from local pixel colour averaging, and selecting the closest-matching tile from the dataset. The matching is done in RGB. This works well for a single image because all computation happens once, and colour averaging over RGB is good enough when the source never changes.

To extend this to video, the system needs to select new tiles for every frame. This immediately makes colour distance more critical. RGB does not represent perceptual similarity very well, two colours that look similar to the human eye may be far apart numerically. For a video mosaic that recomputes thousands of tile matches per second, these perceptual errors become more visible. For this reason, the video version converts both dataset tiles and frame samples into LAB colour, which is designed so numerical distance correlates with human colour difference. The matching step changes from comparing np.linalg.norm(tile_avg_rgb - pixel_rgb) to computing squared LAB distances between the frame grid and a matrix of LAB tile means. This change produces smoother colour matching over time and prevents flickering caused by imperfect RGB matching.

The static mosaic also computes averages only once per tile. In video, the target colours change every frame, so the system reshapes the frame-LAB image into (H*W, 3) and performs a vectorised distance computation against all tile LAB vectors using broadcasting. This replaces the loop matching used in the image version with a fully NumPy batch operation. The speed difference is the reason the video version can compute matches fast enough to be used on video data with 28-60 frames per second of film.

The static app pastes resized PIL tiles directly into a blank output image. The video app must instead build mosaics as NumPy arrays, because OpenCV’s cv2.VideoWriter expects BGR frames in array form. As a result, the mosaic construction is rewritten from:

```
mosaic.paste(best_tile_for(pixel), position)
```

to:

```
mosaic[oy:oy+H, ox:ox+W] = tiles[index]
```

This keeps the entire pipeline in OpenCV/NumPy and avoids the latency of mixing PIL and NumPy operations inside a per-frame loop.

To produce a stable video output, the system enforces several constraints inside the main encoding loop. First, the output dimensions are fixed (out_w = GRID_W * TILE_W, out_h = GRID_H * TILE_H) so every frame has the exact same resolution required by the video codec. Next, OpenCV’s VideoWriter is configured with the input frame rate (fps = cap.get(... )) for consistency across thousands of generated frames. After the mosaic is assembled in RGB space, it must be converted back into BGR using cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR) because OpenCV’s encoder expects BGR pixel order. Finally, each completed frame is pushed into the writer via writer.write(...), guaranteeing that the tile-based reconstruction remains synchronized, evenly sized, and codec-compatible throughout the entire video sequence.

```python
# Load dataset
def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)

def load_tiles(path):
    files = list_images(path)
    imgs = []
    labs = []
    for f in files:
        im = cv2.imread(f)
        if im is None: continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (TILE_W, TILE_H))
        imgs.append(im)
        lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
        labs.append(lab.reshape(-1,3).mean(axis=0))
    return np.stack(imgs), np.stack(labs)

tiles, tile_labs = load_tiles(DATASET_DIR)

# Video 
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
out_w, out_h = GRID_W*TILE_W, GRID_H*TILE_H

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (out_w, out_h))

# Main loop
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(frame, (GRID_W, GRID_H))
    small_lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Compute target for each cell
    target = small_lab.reshape(-1,3)

    # Compute distance to tile 
    diff = target[:,None,:] - tile_labs[None,:,:]
    dists = (diff**2).sum(axis=2)

    # Best tile index per cell
    best = np.argmin(dists, axis=1)
    best = best.reshape(GRID_H, GRID_W)

    # Build mosaic
    mosaic = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for y in range(GRID_H):
        for x in range(GRID_W):
            oy, ox = y*TILE_H, x*TILE_W
            mosaic[oy:oy+TILE_H, ox:ox+TILE_W] = tiles[best[y,x]]

    writer.write(cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
```


The normal video mosaic behaves in a twitchy, almost classic tv static signal way, even in areas of the footage that should be visually stable. For example, the plain white side of a van causes tiles to flicker constantly, not because the scene is changing, but because tiny frame-to-frame variations in lighting trigger different best-match tiles from the dataset. The result feels digitally noisy rather than artistically intentional. This continual flicker also overloads the viewer’s perceptual system: the brain must process both the high-density information inside each tile and the rapid pattern changes across the grid simultaneously, leaving little room for deeper meaning to settle.

Basic best-match colour mapping also produces a different limitation: flat regions of similar colour—for instance, the brown of someone’s trousers, the white side of a van, collapse into a single repeatedly selected dataset tile. Large objects become monotonous blocks of the same repeated image, which adds little expressive value to the scene. In contrast, a system that deliberately controls when tiles are allowed to change—keeping still regions stable while introducing subtle variation in broad colour areas—creates a more balanced aesthetic. It becomes less visually chaotic but also more alive, replacing digital twitch with intentional pacing and allowing the viewer’s attention to rest and roam in more meaningful ways.

[Watch the video on YouTube](https://www.youtube.com/watch?v=0lgUnzSkyVw)


![Video Screenshot](/Screenshot%202025-12-06%20at%2014.05.53.png)

# Stability through thresholding   
## Colour threshold, Cooldown, and top K

The enhanced mosaic generator extends the behaviour of the simpler version by introducing two new stabilising mechanisms: colour-difference thresholding and a per-tile cooldown timer. These additions fundamentally change how and when tiles update, producing a more controlled, less chaotic mosaic compared to the fully reactive approach used previously.

The earlier mosaic simply converted each frame of video into a low-resolution grid, computed the LAB colour of every grid cell, and then selected the closest-matching tile for that cell by computing squared Euclidean distance between the target LAB value and the LAB values of every tile in the dataset. This meant tiles updated continuously and instantly, even with tiny pixel-level fluctuations. The result was visually unstable, producing frame-to-frame jitter because the best tile frequently changed for minor colour variations.

To address this instability, the new system evaluates colour difference more intentionally. For each tile location (y, x), the code now computes a ΔE-like metric using Euclidean distance in LAB space:

```
diff = np.linalg.norm(target - cur_tile_lab)
```

Instead of always using the closest tile, the system only updates the tile when this difference exceeds a predefined **STABILITY_THRESHOLD**. LAB colour space is used because Euclidean distances in LAB correlate more closely with human-perceived colour differences, meaning that ΔE values correspond to intuitive visual changes. This thresholding produces a form of noise-gating: small colour fluctuations no longer cause tile flipping, and only meaningful motion or lighting changes trigger updates.

A second mechanism, the **cooldown timer**, further stabilises the output by regulating update frequency at the tile level. Each tile position maintains a countdown value stored in the array `cooldown[y, x]`. When a tile changes, its cooldown is set to `TILE_COOLDOWN_FRAMES`, preventing it from updating again until this timer reaches zero. The check occurs here:

```
if diff >= STABILITY_THRESHOLD and cooldown[y,x] == 0:
```

This ensures updates occur only if the colour difference is large enough and the tile is not cooling down. Because the cooldown is decremented every frame via:

```
cooldown = np.maximum(cooldown - 1, 0)
```

tiles effectively become “locked” for a short period after change. This creates temporal persistence and gives the mosaic surface a more structured, analogue quality, similar to physical media with inertia rather than a purely digital, instantaneous reaction.

---

The cooldown system is implemented to prevent any individual tile from updating too rapidly, which would otherwise introduce visual flicker and temporal instability into the mosaic. Conceptually, cooldown works by giving each tile its own countdown timer that delays further updates after a change. This is achieved using a 2-D integer matrix, `cooldown`, aligned with the grid geometry (GRID_H × GRID_W). Each element in this matrix stores the number of frames remaining before the corresponding tile is allowed to update again. When a tile changes, the algorithm assigns it a cooldown value using:

```
cooldown[y, x] = TILE_COOLDOWN_FRAMES
```

effectively “locking” that tile.

On every subsequent frame, this timer is reduced through a decay step, implemented as:

```
cooldown = np.maximum(cooldown - 1, 0)
```

This vectorised subtraction decrements all active cooldown values simultaneously while ensuring they never drop below zero.

Cooldown then operates as a gating condition during tile selection: a tile is only eligible for replacement when two criteria are satisfied—

1. its colour difference from the target frame exceeds the stability threshold, **and**  
2. its cooldown timer has reached zero.

This second condition, encoded as:

```
cooldown[y, x] == 0
```

enforces temporal spacing between updates, ensuring that even if a region of the video changes wildly from frame to frame, tiles cannot oscillate rapidly in response. The result is a form of temporal regularisation: instead of every frame independently driving tile selection, cooldown introduces memory into the system, allowing each tile to evolve more slowly and coherently over time.

Mathematically, this transforms the mosaic from a purely instantaneous mapping function to a **dynamic system with per-cell state**, where update opportunities occur at most once every *T* frames. This behaviour is crucial for producing a more stable and aesthetically controlled mosaic, especially in videos with noise, motion, or luminance fluctuations.


Once a tile is eligible for updating, the system selects a replacement using a **top-k matching strategy**. Instead of always picking the single best match, the system retrieves the ten closest matches and randomly chooses one that **differs from the current tile**.

The relevant operation is:

```
tk = topk_match(target, tile_labs, TOPK)
choices = [i for i in tk if i != current_idx[y, x]]
new_tile = random.choice(choices) if choices else tk[0]
```

By selecting from a small set of nearest neighbours rather than the absolute best match, the system introduces variation into stable regions while avoiding chaotic flickering. This creates a mosaic that evolves subtly over time, even when parts of the scene remain roughly the same colour.


# SnapShot Video Mosaic

I liked the trails from the cooldown, so I thought to try and push that. Seeing as how the trails are left by freezing change I thought setting points to compare change to rather than comparing for every frame would really exaggerate the effect.

This variation of the mosaic generator replaces the per-frame colour-thresholding approach used in the stable system with a **snapshot-driven update model**. Instead of comparing each incoming video frame directly to the currently displayed tile, the algorithm samples the video periodically, taking a reference snapshot that acts as a baseline for all tile updates within a fixed window of time.

`SNAP_DURATION_SEC` defines how often the algorithm takes a new reference frame.  
The number of frames in this interval is computed as:

```
SNAP_FRAMES = int(fps * SNAP_DURATION_SEC)
```

Whenever:

```
frame_count % SNAP_FRAMES == 0
```

the system copies the current downsampled LAB frame into `snapshot_lab`.

This array holds the target colour values for the **entire duration of the snapshot window**, insulating the system from noise and micro-fluctuations in the original video.


Tile update logic is also reorganised around this snapshot boundary.  
At the start of every snapshot window, the system clears a boolean `locked` mask:

```
locked = np.zeros((GRID_H, GRID_W), np.bool_)
```

Each tile begins the window in an **unlocked** state, meaning it may undergo **at most one change** during the snapshot period. This introduces a strict **one-change-per-window rule** that stabilises behaviour and prevents tiles from oscillating or updating too frequently.


During the window, each incoming frame is compared **not against the current tile**, but against the **frozen snapshot reference**.

```
target = small_lab[y, x]
snap   = snapshot_lab[y, x]
delta  = np.linalg.norm(target - snap)
```

Using a fixed reference frame means that tiles update only when the video diverges significantly from the snapshot baseline.  
This has the effect of producing **coherent waves of change** across the mosaic as the scene evolves, rather than reacting to every minor fluctuation.

Once a tile is updated—determined by:

```
delta >= COLOR_THRESHOLD
```

—the tile index is replaced via the same top-k matching scheme, *and then* the tile becomes locked:

```
locked[y, x] = True
```

This prevents further updates until the next snapshot window.  
In essence, **tile locking acts as a coarse-grained cooldown** that resets only when a new snapshot is taken.


The rendering stage proceeds identically to the stable system: the tile indices are mapped back onto the output canvas and resized to final resolution. However, because only a subset of tiles update per frame (and each tile updates at most once per snapshot), the resulting animation has a distinctive aesthetic governed by larger temporal blocks rather than per-frame thresholds.

The snapshot-driven system transforms the mosaic into a **temporally chunked process**.  
Where the stable algorithm continuously evaluates frame-to-frame ΔE in LAB space, this design periodically freezes the reference, enforces a single update per tile per window, and evaluates colour change against a **temporally stable baseline**.

The result is a more intentional, progressive form of image transformation, where visual change unfolds in structured waves tied to the rhythm of the snapshot interval.

The snapshot-driven system transforms the mosaic into a temporally chunked process. Where the stable algorithm continuously evaluates frame-to-frame delta in LAB colour space, this design periodically freezes the reference, enforces a single update per tile per window, and evaluates colour change against a temporally stable baseline. The result is a more intentional, progressive form of image transformation, where visual change unfolds in structured waves tied to the rhythm of the snapshot interval.
