# Intro

For film to read clearly, the eye needs anchors, regions of stillness against which motion becomes legible. A mosaic built from images holds far more information than the footage beneath it; each tile is its own small world. As Walter Benjamin wrote, “Every image is the bearer of a message.” But the viewer needs time to receive that message. If tiles changed every frame, their meaning would disappear before the nervous system could catch it.

By keeping tiles fixed in still regions and only updating them when the video actually moves—rather than when tiny lighting variations occur—the system gives the viewer time to inhabit each fragment. Stillness becomes a place where attention rests; deliberate motion becomes a new medium that writes a second story over the message of each tile.

This creates layered meaning: the still tiles tell one story with their embedded imagery, and the moving areas tell another through rhythm and interruption. The final piece is the negotiation between what remains and what constantly becomes.

⸻

## Starting with the image-based system

The static mosaic works by shrinking the source image into a low-resolution grid, computing the average colour of each grid cell, and selecting the closest-matching dataset tile using Euclidean distance in RGB. This is computationally cheap and entirely adequate for a single image, where the colour comparison happens once and never needs to be updated.

![iamge from class](RatMosaic2.png)

When expanding this idea to video, each frame needs a new set of tile matches. At 25–60 frames per second, RGB’s poor perceptual accuracy becomes a problem: colours that look similar to the eye may be numerically far apart, causing visible flicker frame-to-frame. To fix this, the video version converts both dataset tiles and frame samples into LAB colour space, where numerical distance aligns more closely with human perception. Matching accuracy becomes smoother and temporal flicker is reduced.

The static system performs matching inside a loop. For speed, the video system replaces this with a vectorised NumPy operation that reshapes the frame grid into (H*W, 3) and computes all LAB distances at once via broadcasting.

In the image version, tiles are pasted using PIL. In video, the entire pipeline is moved to NumPy and OpenCV, since cv2.VideoWriter expects BGR arrays. The mosaic construction changes from:

```
mosaic.paste(best_tile_for(pixel), position)
```

to:

```
mosaic[oy:oy+H, ox:ox+W] = tiles[index]
```

This avoids PIL’s overhead inside the frame loop.

To ensure stable video output, the system fixes output dimensions (required by the codec), uses the camera’s actual frame rate, converts the RGB mosaic back to BGR for encoding, and writes each frame to VideoWriter. This guarantees synchronisation and prevents drift or resizing artefacts.

⸻

## Instability in the basic video mosaic

The naïve video mosaic flickers heavily. Flat areas—such as the white side of a van—cause the system to oscillate between similar tiles because tiny lighting fluctuations shift the best LAB match. The result looks like digital noise rather than intentional motion. It also overwhelms the viewer: each tile contains detailed imagery, and rapid tile swapping forces the brain to re-interpret hundreds of micro-images every second.

Large regions of similar colour also collapse into repeatedly chosen tiles, making big objects visually monotonous. Allowing tiles to remain stable in still regions, while introducing controlled variation elsewhere, produces a more readable and expressive mosaic. Instead of chaotic twitching, the system gains pacing and rhythm.

[![Basic mosaic video](https://img.youtube.com/vi/0lgUnzSkyVw/mqdefault.jpg)](https://www.youtube.com/watch?v=0lgUnzSkyVw)

⸻

## Introducing motion detection

I tried to stabilise the flicker by borrowing an idea from video class: updating a background model for motion detection, and only changing a tile when something genuinely moves in that region of the frame.

I added a moving-average background model:

```
cv2.accumulateWeighted(small, background, BG_ALPHA)
```

This produces a slowly updated estimate of what the scene looks like when nothing is changing. Motion is then detected by subtracting this background from the incoming frame:

```
diff = cv2.absdiff(small, bg_uint8)
_, motion_mask = cv2.threshold(gray, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
```

If a pixel differs enough from the background, that tile becomes eligible to update.

This immediately reduced much of the chaotic flicker: changes were now connected to movement in the scene, not noise.

Motion detection improved stability, but it created a new problem: when an object moved, some of the tiles it had flipped were left behind—little bits of car or patches of trouser colour floating oddly in an otherwise normal scene.

Motion created meaningful change, but nothing told the system when to undo that change if it didn’t meet its own threshold.

I thought the system needed colour-difference thresholding to clean up the scene once movement had passed. Motion would trigger updates; colour difference would restore any trails back to a closer fit with the updating background.

⸻

## Stability Through Thresholding

### Colour Threshold, Cooldown, and Top-K

The system develops further by adding two stabilising mechanisms: colour-difference thresholding and per-tile cooldown.

The new system evaluates colour difference more intentionally. For each tile position, it computes a ΔE-style LAB distance:

```
diff = np.linalg.norm(target - cur_tile_lab)
```

The colour thresholding was set up to work only after motion has occurred. A tile is updated only if diff exceeds STABILITY_THRESHOLD.

This acts as a noise gate: tiny fluctuations are ignored, and only meaningful changes trigger new tiles.

⸻

### Cooldown

Cooldown gives each tile a small period of rest after it changes. Every tile has a counter stored in:

```
cooldown[y, x]
```

When a tile updates, its cooldown is set:

```
cooldown[y, x] = TILE_COOLDOWN_FRAMES
```

and it cannot update again until the counter reaches zero.

Each frame decrements all cooldown values:

```
cooldown = np.maximum(cooldown - 1, 0)
```

The system only replaces a tile if its cooldown is finished and the colour difference exceeds threshold:

```
if diff >= STABILITY_THRESHOLD and cooldown[y,x] == 0:
```

This simple mechanism introduces temporal memory: tiles gain persistence and cannot oscillate rapidly even if the input video is noisy.

⸻

### Top-K Matching

To tackle the grouping of block tiles—where an area of uniform colour tends to get mapped to the exact same tile—I introduced an element of randomised selection. Once a tile is eligible to update, it selects not the single best match but one tile from the K closest options:

```
tk = topk_match(target, tile_labs, TOPK)
choices = [i for i in tk if i != current_idx[y, x]]
new_tile = random.choice(choices) if choices else tk[0]
```

This adds subtle variation while avoiding repetitive patterns.

⸻

## Result

These mechanisms—thresholding, cooldown, and top-K—collectively remove the digital twitch of the naïve version. Still regions remain still; broad surfaces hold gentle variation; movement is clearly readable.

Videos showing behaviour:
Stable & clean version

[![Stable & clean version](https://img.youtube.com/vi/SzB06SICfe8/maxresdefault.jpg)](https://www.youtube.com/watch?v=SzB06SICfe8)

Cooldown pushed for trails

[![Cooldown pushed for trails](https://img.youtube.com/vi/usaoD4HNqfE/maxresdefault.jpg)](https://www.youtube.com/watch?v=usaoD4HNqfE)

No cooldown / hyper-reactive

[![No cooldown / hyper reactive](https://img.youtube.com/vi/qXGXfKK2Xb0/maxresdefault.jpg)](https://www.youtube.com/watch?v=qXGXfKK2Xb0)

Cooldown ≈ framerate = frozen background

[![Cooldown ≈ framerate (freeze background)](https://img.youtube.com/vi/6kUFOrU2RWs/maxresdefault.jpg)](https://www.youtube.com/watch?v=6kUFOrU2RWs)

⸻

## Rethinking motion and colour

In reflection, I realised that detecting motion was over-powered when I was already using LAB colour-difference to clean up after motion had passed. If colour-difference could stabilise the image on its own, why not let it drive the tile flips entirely? Also by controlling the background-updating cycle needed to compare colours, I could push the trail effect much further.

In the motion-detection version, the system uses a continuously updating moving-average background model to detect fast temporal change. Each frame nudges the background slightly:

```
cv2.accumulateWeighted(small, background, BG_ALPHA)  # α = 0.02
```

This creates a background that changes slowly, representing the long-term state of the scene. The system detects motion by comparing the current frame to this slow background:

```
diff = cv2.absdiff(small, background_uint8)
motion_mask = diff > MOTION_THRESHOLD
```

If something changes faster than the background can adapt, the difference spikes, and the tile is allowed to flip. This detects instant movement, then quickly returns tiles to stable states once movement stops.

⸻

## The snapshot idea

The snapshot version does not detect motion. Instead, it freezes a reference frame every few seconds:

```
snapshot_lab = small_lab.copy()   # once per SNAP_FRAMES
```

All updates within that window compare the live frame to this frozen snapshot:

```
delta = np.linalg.norm(current_lab - snapshot_lab)
```

Tiles change only when the scene diverges far enough from the snapshot—not when it moves quickly. This detects accumulated change over time, not motion. It produces slow waves of transformation and long trails because each tile can update only once per snapshot window.

In the motion-based version, the background slowly adapts:

```
cv2.accumulateWeighted(small, background, BG_ALPHA)
```

⸻

### Snapshot Window

The system periodically captures a LAB snapshot of the video:

```
SNAP_FRAMES = int(fps * SNAP_DURATION_SEC)
```

Whenever:

```
frame_count % SNAP_FRAMES == 0
```

the downsampled LAB frame becomes the new snapshot_lab.
This snapshot acts as the baseline for all tile updates until the next window begins.

⸻

### Tile Locking

At the start of each snapshot window, the system resets a locked mask:

```
locked = np.zeros((GRID_H, GRID_W), np.bool_)
```

Each tile is now allowed one update per window.
Once a tile changes:

```
locked[y, x] = True
```

it cannot change again until the next snapshot.
This is essentially a coarse-grained cooldown tied to the snapshot rhythm.

The effect is strong temporal pacing: tiles update in slow, coherent waves rather than reacting independently every frame.

⸻

### Update Logic

During a snapshot window, each incoming frame is compared to the frozen snapshot:

```
delta = np.linalg.norm(target - snapshot_lab[y, x])
```

If delta exceeds the threshold:

```
delta >= COLOR_THRESHOLD
```

the tile selects a replacement via top-k matching, then locks:

```
locked[y, x] = True
```

Because the reference frame doesn’t change until the next window, all updates relate back to the same baseline, giving the mosaic a sense of continuity and direction.

⸻

### Rendering

The rendering stage is identical to the stable system:
tiles are placed into a NumPy mosaic canvas and resized to final resolution.

But because tiles update only once per window, the animation shifts from granular flicker to large-scale, sculpted movement.

⸻

## Resulting Behaviour

The snapshot mosaic turns the system into a rhythmic, temporally chunked process.
Instead of responding to each frame, the mosaic evolves in blocks tied to the snapshot interval.
The effect is more intentional and progressive — change arrives in waves, not flicker.

Motion has weight and direction.
Tile updates feel choreographed rather than chaotic.

⸻

## Motion-Only Video

After building the snapshot-based system, I wanted to test the opposite extreme:
what happens if colour analysis is removed entirely and tile changes are driven only by motion?

As a quick experiment, I wrote a minimal version of the mosaic that keeps LAB and top-k matching out of the loop. The system simply tracks motion using a running background model (cv2.accumulateWeighted) and flips a tile to a random image whenever the motion mask is active at that location.

Because no colour comparison is involved, the results feel chaotic and impulsive—tiles pop like static wherever the background registers movement.

This stripped-down version helped clarify what the motion detector was doing in the fuller system, and it showed how much visual structure comes from colour-based matching rather than motion alone. Also it looks cool.

⸻

## Conclusion

Each technical change served the same artistic goal: giving the viewer time to read both layers of the image—the dataset tiles and the underlying footage—without one overwhelming the other.

Stability lets the embedded images speak; controlled change provides a counterpoint.
The final system negotiates between stillness and motion, memory and update, creating an image that evolves in a way that feels more like choreography than computation.

In the end, the mosaic becomes a conversation between what the frame wants to do and what the system allows to change.
This balance between control and emergence is what gives the piece its character, and what makes the experiment feel complete.

⸻
