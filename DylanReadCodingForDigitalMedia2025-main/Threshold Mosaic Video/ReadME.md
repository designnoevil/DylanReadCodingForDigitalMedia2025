# Intro

For film to read clearly, the eye needs anchors—regions of stillness against which motion becomes legible. A mosaic built from images holds far more information than the footage beneath it; each tile is its own small world. As Walter Benjamin wrote, “Every image is the bearer of a message.” But the viewer needs time to receive that message. If tiles changed every frame, their meaning would disappear before the nervous system could catch it.

By keeping tiles fixed in still regions and only updating them when the video actually moves—rather than when tiny lighting variations occur—the system gives the viewer time to inhabit each fragment. Stillness becomes a place where attention rests; deliberate motion becomes a rupture that writes a second story over the first.

This creates layered meaning: the still tiles tell one story with their embedded imagery, and the moving areas tell another through rhythm and interruption. The final piece is the negotiation between what remains and what constantly becomes.

⸻

## Starting with the image-based system

The static mosaic works by shrinking the source image into a low-resolution grid, computing the average colour of each grid cell, and selecting the closest-matching dataset tile using Euclidean distance in RGB. This is computationally cheap and entirely adequate for a single image, where the colour comparison happens once and never needs to be updated.

When expanding this idea to video, each frame needs a new set of tile matches. At 25–60 frames per second, RGB’s poor perceptual accuracy becomes a problem: colours that look similar to the eye may be numerically far apart, causing visible flicker frame-to-frame. To fix this, the video version converts both dataset tiles and frame samples into LAB colour space, where numerical distance aligns more closely with human perception. Matching accuracy becomes smoother and temporal flicker is reduced.

The static system performs matching inside a Python loop. The video system replaces this with a vectorised NumPy operation that reshapes the frame grid into (H*W, 3) and computes all LAB distances at once via broadcasting. This is the main reason the system can run fast enough to reconstruct video.

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

## Explaining instability in the basic video mosaic

The naive video mosaic flickers heavily. Flat areas, such as the white side of a van, cause the system to oscillate between similar tiles because tiny lighting fluctuations shift the best LAB match. The result looks like digital noise rather than intentional motion. It also overwhelms the viewer: each tile contains detailed imagery, and rapid tile swapping forces the brain to re-interpret hundreds of micro-images every second.

Large regions of similar colour also collapse into repeatedly chosen tiles, making big objects visually monotonous. Allowing tiles to remain stable in still regions, while introducing controlled variation elsewhere, produces a more readable and expressive mosaic. Instead of chaotic twitching, the system gains pacing and rhythm.

[![Basic mosaic video](https://img.youtube.com/vi/0lgUnzSkyVw/maxresdefault.jpg)](https://www.youtube.com/watch?v=0lgUnzSkyVw)

## Stability Through Thresholding

### Colour Threshold, Cooldown, and Top-K

The enhanced mosaic builds on the basic system by adding two stabilising mechanisms: colour-difference thresholding and per-tile cooldown. Together they prevent the rapid flicker seen in the naive version and give the mosaic a more deliberate visual rhythm.

Originally, each frame was reduced to a LAB grid, and each cell simply picked the closest tile. Because LAB distance is sensitive, minor pixel-level fluctuations caused tiles to change constantly. The result was temporal jitter even in visually still regions.

The new system evaluates colour difference more intentionally. For each tile position, it computes a ΔE-style LAB distance:

```
diff = np.linalg.norm(target - cur_tile_lab)
```

A tile is only updated if diff exceeds STABILITY_THRESHOLD.
This acts as a noise gate: tiny fluctuations are ignored, and only meaningful changes trigger new tiles.

⸻

### Cooldown

Cooldown gives each tile a small period of “inertia” after it changes. Every tile has a counter in:

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

This simple mechanism introduces temporal memory: tiles gain persistence and cannot oscillate rapidly even if the input video is noisy. It transforms the mosaic from a per-frame mapping into a dynamic system with per-tile state, helping it feel more analogue and less digital.

⸻

### Top-K Matching

Once a tile is eligible to update, it selects not the single best match, but one tile from the K closest options. This is done by:

```
tk = topk_match(target, tile_labs, TOPK)
choices = [i for i in tk if i != current_idx[y, x]]
new_tile = random.choice(choices) if choices else tk[0]
```

Choosing from several near-equal matches introduces controlled variation and prevents large colour regions from collapsing into a single repeating tile. The mosaic becomes more expressive but stays stable.

⸻

## Result

These mechanisms — thresholding, cooldown, and top-K — collectively remove the digital twitch of the naive version. Still regions remain still; broad surfaces hold gentle variation; movement is clearly readable. The viewer can now follow both the underlying footage and the embedded dataset imagery.

Videos showing behaviour:

Stable & clean version
[![Stable & clean version](https://img.youtube.com/vi/SzB06SICfe8/maxresdefault.jpg)](https://www.youtube.com/watch?v=SzB06SICfe8)

Cooldown pushed for trails
[![Cooldown pushed for trails](https://img.youtube.com/vi/usaoD4HNqfE/maxresdefault.jpg)](https://www.youtube.com/watch?v=usaoD4HNqfE)

[No cooldown / hyper reactive
[![No cooldown / hyper reactive](https://img.youtube.com/vi/qXGXfKK2Xb0/maxresdefault.jpg)](https://www.youtube.com/watch?v=qXGXfKK2Xb0)

Cooldown ≈ framerate gives freeze 
[![Cooldown ≈ framerate (freeze background)](https://img.youtube.com/vi/6kUFOrU2RWs/maxresdefault.jpg)](https://www.youtube.com/watch?v=6kUFOrU2RWs)
















## Snapshot Video Mosaic

I liked the trail effects produced by cooldown, so I pushed the idea further. Instead of comparing each frame to the current tile—where even slight fluctuations can trigger change—the snapshot system compares every frame in a window to a fixed reference frame. This exaggerates the sense of temporal layering and produces more coherent waves of transformation.

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

Because the reference remains stable for the entire interval, the mosaic no longer reacts to micro-fluctuations. Only genuine divergence from the snapshot triggers updates, producing more meaningful, structured changes.

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

### Update Logic (

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
The effect is more intentional and progressive, Change arrives in waves, not flicker. Motion has weight and direction. Tile updates feel choreographed rather than chaotic.

The result has a distinctive, almost cinematic quality:

Snapshot Mosaic Video:
https://www.youtube.com/watch?v=bl6-KoTTQKA

⸻

## Conclusion

Each technical change served the same artistic goal: giving the viewer time to read both layers of the image—the dataset tiles and the underlying footage—without one overwhelming the other. Stability lets the embedded images speak; controlled change provides a counterpoint. The final system negotiates between stillness and motion, memory and update, creating an image that evolves in a way that feels more like choreography than computation.

In the end, the mosaic becomes a conversation between what the frame wants to do and what the system allows to change. This balance between control and emergence is what gives the piece its character, and what makes the experiment feel complete.
