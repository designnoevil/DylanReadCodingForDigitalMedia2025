# Prefix

Having studied cabinetry and blacksmithing for three years, then a BA in product design, and now working in Architecture and Design at the University of Brighton, I think a lot about human-centred practice. The user of a product, the user of a space—these contexts shape how we design and how ideas are judged. Design never happens in a vacuum; it happens through people, for people. (Mostly)

In my undergraduate years, and throughout my subsequent career, I’ve often had to map spaces for how a person moves through them. When people leave a space, the objects they touch, the wear that gathers in particular spots, and the space itself—the layout, the décor, the areas of differing wear, all keep a of memory of people.

The image below is a map of my daily use of the apartment I lived in during my second year of undergrad. It shows a memory of my passage through 3D spaces, a re-projection of the time dimension that’s usually lost in a single photo. 
![Ghosts Screenshot](Screenshot%202025-12-05%20at%2014.48.56.png)

In reflecting on Making an Interactive Dance Piece: Tensions in Integrating Technology in Art, I set out to make a piece with a quieter, deeper argument about how people and technology co-produce meaning. As we move through rooms, we inscribe memory into objects, surfaces, and arrangements; spaces carry a residue of use. By visualising motion as a live, camera-driven “ghost,” I try to reveal that residue—an added temporal layer draped over ordinary three-dimensional space. The work asks two questions: does memory belong to the room itself, or to the observer who reads it there? And as in architecture where we design for others who use the spaces, is memory ever anything but relational, made between bodies, tools, and place?

⸻

## OVERVIEW

Ghosts turns a live webcam feed into a memory of motion. It maintains a slowly updating background model, detects per-pixel deviation from that model, and stores recent motion masks in a ring buffer. Each frame blends the live camera image toward white only where motion has persisted, so gesture accumulates into translucent trails that fade with time. Movement becomes a residue: a visual memory field that lingers on the screen the way physical actions linger in spaces.

⸻

## START

The system opens the webcam and captures an initial frame:

```python
cap = cv2.VideoCapture(CAMERA_INDEX)
ok, frame = cap.read()
```

OpenCV provides the frame in BGR; Dorothy expects RGB, so the colours are converted:

```python
bg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

The image is resized to the canvas:

```python
background = cv2.resize(bg, (dot.width, dot.height))
```

Each frame in draw() follows the same structure: read → convert → resize.

```python
ok, frame = cap.read()
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (dot.width, dot.height))
```

To detect change, the system compares the new frame with the stored background using absolute difference:

```python
diff = cv2.absdiff(rgb, background)
gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
_, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
```

Pixels that differ enough turn white; stable pixels remain black. Because this comparison depends entirely on brightness relative to the stored snapshot, objects can appear inconsistently white or black depending on the lighting at startup. This is part of the system’s behaviour: it reveals difference.

Dorothy expects three-channel images, so the mask is expanded:

```python
mask_rgb = np.dstack([mask, mask, mask])
```

I fixed the canvas size (960×540) to maintain predictable performance—echoing Week 7’s reminder that reliability in an interactive system often outweighs ideal architectural design. A stable feed supports the conceptual work more than perfect resolution.

The thresholded mask becomes a live, binary comparison between the body and the room’s initial state. 
![Ghosts Screenshot](Screenshot%202025-12-05%20at%2014.53.17.png)

Because this comparison is purely based on brightness differences, my hand is treated differently depending on what part of the background it is covering. Against the darker interior of the room from the snapshot, my hand appears bright; against the brighter window from the snapshot, it appears dark. due to the bianry being this or that, The result is an outline of my hand with the original snapshot imagery (the trees outside the window) showing through inside its silhouette. This happens because, wherever the hand is similar in brightness to the stored background, the system simply reuses the snapshot pixels instead of marking them as different.
⸻

## MID

### Adding a Moving-Average Background Model

The first-frame background method is extremely fragile: tiny exposure shifts, screen flicker, sensor noise, and ambient light changes all appear as false “motion.” To stabilise the system, the background becomes a moving average rather than a fixed snapshot.

The first frame seeds the background:

```python
if self.background is None:
    self.background = rgb.astype(np.float32)
```

Then each new frame updates it slowly:

```python
cv2.accumulateWeighted(rgb, self.background, BG_ALPHA)
bg_uint8 = self.background.astype(np.uint8)
```

A small learning rate (α = 0.02) means the model absorbs only long-term changes while ignoring momentary fluctuations. Shadows, flicker, and camera jitter fade into the background, while actual movement stands out.

Motion is then computed as:

```python
diff = cv2.absdiff(rgb, bg_uint8)
gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
_, mask = cv2.threshold(gray, MOVEMENT_THRESHOLD, 255, cv2.THRESH_BINARY)
mask = cv2.medianBlur(mask, 5)
```

The median blur removes isolated noise pixels, giving the mask a more intentional, less jittery presence.

This simple temporal smoothing changes the character of the system. The background becomes a kind of temporal surface—a slowly adapting memory of what the room “should” look like. When the EMA settles, only meaningful motion registers as white, and stillness gradually disappears back into black.

You can see this in the slight glowing trail beside my head in the frame below, where the system briefly remembers my previous position before the background model absorbs it:

⸻

## ADDING RING BUFFER & FIRST PROTOTYPE

This stage introduces the project’s first explicit model of memory over time. Instead of reacting only to the current frame, the system stores a short history of binary masks in a ring buffer:

```python
motion_buffer = np.zeros((WINDOW_SIZE, dot.height, dot.width), dtype=np.uint8)
ptr = 0
count = 0
```

After producing each new motion mask:

```python
motion_buffer[ptr] = mask
ptr = (ptr + 1) % WINDOW_SIZE
count = min(count + 1, WINDOW_SIZE)
```

This circular memory lets the system recall where movement has occurred over the last few seconds.

The simplest version of the ghost layer overlays motion whenever any frame in the buffer detected change:

```python
ghost = np.zeros_like(mask)
for i in range(count):
    ghost = np.where(motion_buffer[i] > 0, 255, ghost)
```

This produces bright, block-white trails that remain until they age out of the buffer. In this prototype, ghosts are opaque: motion stamps itself forcefully onto the live feed.

```python
output = rgb.copy()
output[ghost == 255] = [255, 255, 255]
```

Conceptually, this is where the piece first becomes aware of time.
In Week 7 we discussed ring buffers as a way of holding past information so systems don’t react one frame at a time. What interested me is how this mirrors the questions raised in SKIN: how recognition systems embed their own timing into the body. The EMA already imposes a short lag—motion echoes for a moment before fading. The ring buffer makes this explicit and adjustable: I decide how long gestures should linger. In this sense, the buffer becomes a choreographer, determining what the system remembers and what it lets go.

In this first prototype:
- motion leaves a strong, unmistakable white imprint
- trails persist until overwritten
- fading is abrupt rather than gradual

The result has a certain bluntness—memory appears as a series of solid stamps. But this stage was crucial: it proved that the system could carry motion forward in time and that the interaction between live video and temporal residue was visually compelling.

Below is a still from this stage, with opaque white ghosts overlaying the colour feed:

⸻

## UPDATES TO PROTOTYPE / MAKING THE FINAL APP

The final version shifts from a binary notion of motion (on/off) to a graded, time-weighted memory. Instead of treating each stored mask equally, every frame in the buffer contributes according to its age. This produces a ghost trail that fades smoothly rather than vanishing abruptly.

The system creates a floating-point accumulation layer:

```python
ghost_layer = np.zeros((self.h, self.w), dtype=np.float32)
for i in range(self.count):
    index = (self.ptr - 1 - i) % self.memory
    age_factor = 1.0 - (i / self.count)
    ghost_layer += (self.motion_buffer[index] / 255.0) * age_factor
```

Recent motion has the strongest influence; older movement contributes only a faint trace.
This transforms the buffer from a set of static stamps into a temporal gradient of motion.

After normalising:

```python
ghost_layer = np.clip(ghost_layer, 0.0, 1.0)
```

the system blends toward white using a soft alpha mask:

```python
ghost_alpha = np.clip(ghost_layer * GHOST_STRENGTH, 0.0, GHOST_STRENGTH)[..., None]
blended = ((1.0 - ghost_alpha) * base + ghost_alpha * white).astype(np.uint8)
```

This changes the visual language completely:
- ghosts become semi-transparent fog instead of hard silhouettes
- motion appears as blooming, layered clouds
- trails fade continuously rather than snapping off
- the live video remains visible beneath the memory

Although the data structures are the same, their role changes: the buffer no longer represents “where motion happened,” but how intensely and how recently it occurred.

⸻

## Conceptual Link: Memory as Surface

A key idea in SKIN is that digital surfaces are never static—they carry traces, delays, and inscriptions of what has passed across them. Moving from a binary mask to a weighted accumulation makes that idea tangible. Each gesture leaves an imprint that slowly dissolves, creating a surface that remembers.

This version of the system behaves less like an analytical tool and more like a material.
The camera feed becomes a kind of dynamic skin, where motion lingers, settles, and blends with the present.
Memory is no longer a switch; it is a thickness.

The ghost trail becomes a computational analogue to how touch or movement leaves warmth or pressure behind.
Instead of simply detecting presence, the system shows persistence—the duration and emotional weight of a gesture.

It also feels spatially richer: the background remains visible, the ghosts hover above it, and the boundary between “now” and “just before” becomes soft rather than binary.

A still from this phase shows how walking through a room produces a layered apparition; a mirror multiplies these traces in interesting ways:

<img src="Screenshot 2025-12-05 at 15.10.06.png" width="700">

And the final video demonstrates the system operating:

<video src="GhostsFinalVid(1).mp4" width="700" controls></video>

[![Ghosts – final video](GhostsThumb.png)](GhostsFinalVid(1).mp4)

⸻

## CONCLUSIONS AND REFLECTIONS

The system succeeds in turning motion into a lingering residue, aligning the computational behaviour with the conceptual aim. Under stable lighting it behaves predictably, and the combination of EMA background, thresholding, and weighted temporal memory produces a soft, expressive field of traces.

Working on this forced me to confront how computer vision actually “sees”:
not as bodies, rooms, or objects—but as changing brightness values.
Every improvement—EMA, ring buffer, fading, median filtering—was about shaping ambiguity rather than eliminating it.

The biggest challenges came from situations where brightness relationships broke down:
skin against skin, dark clothing against dark surfaces, sudden exposure shifts.
These moments highlight the limits of subtractive motion detection and reveal the system’s biases toward certain kinds of movement.

If extended further, I would explore:
- optical flow or MOG2 background subtraction for scenes where luminance is unstable
- per-pixel or adaptive thresholds to improve robustness
- depth sensing or body-tracking for separating layered gestures and overlapping limbs
- and possibly expanding the memory model so different types of movement leave different kinds of traces.

But at this stage, the system balances technical clarity with poetic behaviour, supporting the overarching concept:
a computational memory of movement—fading, accumulating, and drifting in the space between presence and absence.

⸻

## References

Alaoui, S.F. (2019) ‘Making an Interactive Dance Piece: Tensions in Integrating Technology in Art’, Proceedings of the 2019 on Designing Interactive Systems Conference (DIS ’19), ACM, New York, pp. 1195–1208. Available at: https://doi.org/10.1145/3322276.3322289
