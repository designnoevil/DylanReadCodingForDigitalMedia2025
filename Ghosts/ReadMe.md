# Prefix

Having studied cabinetry, and blacksmithing, for 3 years. Then a BA in product design, and now with a few years of industry under my belt, and now finding myself working on the technical team of architecture and design at the University of Brighton: I think a lot about human-centred practice. The user of a product and the user of a space. This is the context and the data we use to guide design and judge ideas; design doesn’t happen in a vacuum. It happens by people, for people (mostly).

In my undergraduate years, and throughout my subsequent career, I’ve often had to map spaces for how a person moves through them. When people leave a space, the objects they touch, the wear that gathers in particular spots, and the space itself—the layout, the décor, the areas of differing wear, all keep a of memory of people.

The image below is a map of my daily use of the apartment I lived in during my second year of undergrad. It shows a memory of my passage through 3D spaces, a re-projection of the time dimension that’s usually lost in a single photo. 
![Ghosts Screenshot](Screenshot%202025-12-05%20at%2014.48.56.png)

In reflecting on Making an Interactive Dance Piece: Tensions in Integrating Technology in Art, I set out to make a piece with a quieter, deeper argument about how people and technology co-produce meaning. As we move through rooms, we inscribe memory into objects, surfaces, and arrangements; spaces carry a residue of use. By visualising motion as a live, camera-driven “ghost,” I try to reveal that residue—an added temporal layer draped over ordinary three-dimensional space. The work asks two questions: does memory belong to the room itself, or to the observer who reads it there? And as in architecture where we design for others who use the spaces, is memory ever anything but relational, made between bodies, tools, and place?

# Overview

Ghosts turns live webcam feed into a memory of motion. It maintains a updating background model, detects per-pixel change against that background, and stores recent motion masks in a ring buffer. Each frame blends the live image toward white only where motion has persisted, so movement leaves translucent ghost trails that like memories that fade with time.

# Start

Opening the webcam and capturing a single frame at startup. This is performed in setup():

```python
cap = cv2.VideoCapture(CAMERA_INDEX)
ok, frame = cap.read()
```

cv2.VideoCapture() returns a camera object.  
.read() grabs the first available frame and returns two things: a boolean indicating success, and “frame” a BGR image from the camera.

OpenCV captures images in BGR, but Dorothy expects RGB. The code converts the frame:

```python
bg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

In my early prototype I stacked the binary mask into an RGB image for display, but in the final system the mask remains single-channel and is used only internally.

It then resizes the background image to match the canvas:

```python
background = cv2.resize(bg, (dot.width, dot.height))
```

Every frame captured in draw() goes through several steps.  
First, grab the next camera frame:

```python
ok, frame = cap.read()
```

Convert to RGB and resize:

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (dot.width, dot.height))
```

At this point, rgb and background share the same size and colour space.

Computing pixel-wise difference between the current frame (rgb), and the stored background snapshot (background):

```python
diff = cv2.absdiff(rgb, background)
```

cv2.absdiff computes:

```
diff_pixel = abs(current_pixel - background_pixel)
```

For each pixel, for each colour channel (R, G, B).

Pixels that match the background are close to 0, and pixels that differ from the background are a large positive value. This step does not detect motion—only brightness differences relative to the frozen background snapshot.

The difference image is reduced to a single intensity channel:

```python
gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
```

This collapses the 3-channel RGB difference into a single brightness measure.

Now, each pixel is a single number from 0–255 representing how different it is from the background. To turn the grayscale difference into a mask, thresholding is used:

```python
_, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
```

The value 100 is a fixed cutoff that can be changed to play with the system.

Pixels that differ enough from the background turn white; others disappear into black.

This is why objects of different brightness behave inconsistently, your hand may appear white over something of the same or darker light at start but black over a bright window that might be lighter than your hand colour at start up, because the comparison depends entirely on brightness relative to the stored snapshot.

Dorothy requires a 3-channel image, so the binary mask is stacked:

```python
mask_rgb = np.dstack([mask, mask, mask])
```

This converts a single-channel binary image into a full RGB image where: white = [255, 255, 255], black = [0, 0, 0].

I locked the canvas size (960×540) to keep a predictable pixel budget and latency, echoing Week 7’s reminder that every analysis step eats into frame rate.

“favour the experience on stage (or screen) over ideal architecture.” If a stable feed means a few compromises like fixed resolution, its best to take the compromise because reliability in the foundation holds up everything else down the line.

In the video you see a binary filter that shows a comparison of the difference in light value of the curent frame when compared to the snapshot taken at app start up. Understand what this is helps us use the app for performance/interactiviy later on, as we know that changing the lighting of the backgrabd compared to the subject when using the camra with yeialed difrent resolts. In the video you can see how my hand which is light at snapshot becomes dark when compared to something lighter at snapshot, like the wind behind me.

# Mid

## Adding a Moving-Average Background Model

Issues in practice as the first-frame background method is extremely fragile: auto exposure, changes, sensor noise, compression noise, flicker from screens, ambient light drift, camera shake.

Upgraded from using a single frozen background frame to maintaining a continuously updating background model.

The background is seeded from the very first camera frame, but instead of remaining fixed, it is updated every draw-cycle:

```python
cv2.accumulateWeighted(rgb, background, ALPHA)
```

Where ALPHA controls the learning rate. A small value (here, α = 0.02) means the background adapts slowly, absorbing only the stable elements of the scene. This allows the model to follow long-term light level changes, but ignore short-lived fluctuations such as sensor noise, flicker, or shadows.

The stored background is held in float32 rather than uint8. The high precision prevents rounding artefacts from accumulating and allows the moving average to evolve smoothly. Before subtraction, the background is converted back to 8-bit:

```python
bg_uint8 = background.astype(np.uint8)
```

so that subsequent operations (absdiff, threshold) operate in the expected range.

After thresholding, I apply a small 5×5 median filter to remove isolated noise pixels caused by sensor jitter and compression artefacts, which stabilises the mask and prevents accidental “sparkle” motion from appearing in the ghost trail:

```python
mask = cv2.medianBlur(mask, 5)
```

Once the background model is updated, the motion extraction pipeline remains structurally the same: absolute difference → grayscale → threshold → binary mask. However, the behaviour changes substantially because the background is now a temporal model rather than a snapshot.

```python
diff = cv2.absdiff(rgb.astype(np.uint8), bg_uint8)
gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
_, mask = cv2.threshold(gray, THRESH, 255, cv2.THRESH_BINARY)
```

The result is a mask that responds only to pixels whose brightness diverges significantly from their long-term average.

For the first ~1–2 seconds, the EMA learns the room. The mask flickers until the mean settles.  
Small lighting fluctuations and sensor jitter are absorbed into the background and no longer produce false white pixels.  
Only meaningful changes exceed the threshold and appear as white shapes.  
If an object stops moving, it gradually fades into black as the EMA incorporates it into the background model.  
Sudden illumination shifts briefly cause white regions, but the EMA quickly adapts and returns to a stable mask.


# Adding Ring buffer and 1st prototype

This stage introduces the first major structural change to the system: it extends motion detection from a single frame into a short history of recent frames. Technically, this requires three new components: a ring buffer to store past binary masks, a temporal accumulation loop, and a final compositing step that mix’s the white difference from the binary filter on top of the live feed.

The new `motion_buffer` is a fixed-size 3D NumPy array:

```python
motion_buffer = np.zeros((WINDOW_SIZE, dot.height, dot.width), dtype=np.uint8)
ptr = 0
count = 0
```

Each slice `motion_buffer[i]` stores one binary motion mask (0 or 255).  
WINDOW_SIZE determines how many past frames the system remembers.  
`ptr` is a write cursor; it moves forward each frame and wraps back to zero (circular memory).  
`count` increases until the buffer is full.

After motion detection produces a new binary mask:

```python
motion_buffer[ptr] = mask
ptr = (ptr + 1) % WINDOW_SIZE
count = min(count + 1, WINDOW_SIZE)
```

This ensures that every frame contributes to the evolving trail, and older frames are automatically overwritten. The modulo operator (`%`) guarantees wrap-around, maintaining constant memory and predictable temporal depth.

Instead of displaying only the current mask, the system accumulates masks from the entire memory window:

```python
ghost = np.zeros_like(mask, dtype=np.uint8)
for i in range(count):
    ghost = np.where(motion_buffer[i] > 0, 255, ghost)
```

If any mask in the buffer shows motion at pixel (x, y), that pixel becomes white in `ghost`.

Rather than rendering the mask directly like previously, the new system overlays onto the RGB camera feed:

```python
ghost_rgb = np.dstack([ghost, ghost, ghost])
output = rgb.copy()
output[ghost == 255] = [255, 255, 255]
```

The background remains the real camera feed, and the ghost trail replaces only the pixels marked as active by the temporal buffer.

Remember how long and where motion has been happening, then fade it out over time.  
In the Week 7 class, ring buffers were introduced as a way to hold a number of past frames so the system can understand change over time rather than reacting frame-by-frame. But what struck me in connection with the SKIN reading, is how memory in interactive performance has to be carefully shaped—not simply collected.

The dancers in SKIN struggled with recognition systems that imposed their own timing and their own interpretation on the body; the buffer becomes a way of negotiating that relationship, controlling what the system remembers and for how long.

Even without the ring buffer, there’s already a kind of memory in the pipeline.  
The exponential moving-average background (`accumulateWeighted` at 0.02) lags behind the present, so freshly vacated pixels stay “different” for a few frames; the hard threshold then crystallises that lag into bright echoes.

Add the camera’s auto-exposure drift and the slight blur from resizing, and the eye stitches those momentary mismatches into trails.

In SKIN terms, that’s memory imposed by the system’s own timing—an implicit choreography authored by the machine.  
**The ring buffer is where I take back authorship:** I make that persistence explicit and adjustable, deciding what lingers and what’s forgotten.  
In my case, the buffer is the choreographer: it decides which movements echo and which are forgotten.


In the video you can see:  
Motion leaves a bright white imprint that lingers until the buffer cycles out.  
At this point in development, the system produces block-white trails; every detected motion pixel becomes white and remains so for multiple frames.  
Trails do not fade gradually — they vanish abruptly when overwritten.

# Updates to Prototype / Making final app

At this final stage, the system shifts from treating motion as a binary, all-or-nothing event to treating it as a graded, time-weighted phenomenon. Now each past frame contributes proportionally less the further back in time it is, creating a decaying memory gradient rather than a static imprint.

The new system constructs a floating-point accumulation buffer and weights each frame by its age:

```python
ghost_layer = np.zeros((self.h, self.w), dtype=np.float32)

for i in range(self.count):
    index = (self.ptr - 1 - i) % self.memory
    age_factor = 1.0 - (i / self.count)
    ghost_layer += (self.motion_buffer[index] / 255.0) * age_factor
```

Here, frames closest to the present (i = 0) contribute most strongly, while older frames contribute progressively less.  
The effect is that recent motion appears brighter and more dominant, while older motion fades naturally rather than collapsing at once.

Where the predecessor produced a mask of raw 0/255 values, the new system outputs a continuous 0–1 field:

```python
ghost_layer = np.clip(ghost_layer, 0.0, 1.0)
```

This rescales the accumulated values into a stable display range.  
This step establishes motion intensity rather than motion presence, allowing for variable-strength ghosts rather than fixed-strength.

Now we blend toward white using an alpha mask derived from the temporal ghost field:

```python
ghost_alpha = np.clip(ghost_layer * 0.6, 0.0, 0.6)[..., None]
base = rgb.astype(np.float32)
white = np.full_like(base, 255.0, dtype=np.float32)

blended = ((1.0 - ghost_alpha) * base + ghost_alpha * white).astype(np.uint8)
```

This makes several conceptual changes:

- Motion no longer overwrites reality; it **modulates** it.  
- Ghosts now appear **semi-transparent, soft white fog**, rather than solid white.  
- The intensity of a ghost is proportional to the strength of recent motion.  

Although the buffer structure is unchanged, its role evolves. Before it was a list of on/off stamps, now it is a **time-indexed function** contributing graded values to the final alpha mask.

This replaces the binary compositing of the previous stage with proper **RGBA-style blending**.  
The visual outcome is a memory trail that feels more continuous and atmospheric.


A key idea in *SKIN* is that digital surfaces are never static: they hold traces, delays, and inscriptions of what has passed across them.  
The shift from a hard binary mask to a weighted temporal accumulation embodies this.  
Instead of treating each frame as an isolated event, the system now treats motion as something that **lingers on the surface of the image**, gradually fading but never disappearing instantly.

The ghost trail functions as a computational analogue to the memory of touch described in the reading:  
each interaction leaves an imprint whose intensity depends on the recency and duration of contact.

By introducing temporal decay and alpha-based blending, the image becomes a **dynamic skin** that records the residue of gesture rather than a strict detector of presence.

With these changes, the system produces a noticeably different aesthetic:

- Ghosts no longer appear as harsh white silhouettes; they **bloom softly** into the scene.  
- Recent movement appears as bright, cloud-like traces; older movement fades out gradually.  
- Motion feels smoother because the weighted accumulation bridges frame-to-frame gaps.  
- The camera feed remains visible under the ghosts, giving a layered sense of depth and of the memory of the subject mixing with the space to become one.


# Conclusions and reflections

The system performs well at communicating my intention: turning motion into a lingering residue.  
Under controlled lighting it behaves predictably, and the combination of the EMA background, thresholding, and temporal accumulation produces a surprisingly expressive “memory field” over the live camera feed.  
It works best when the contrast between subject and background is clear, and when the scene is not drifting in exposure.

The main challenge was handling ambiguity at the pixel level.  
Skin against skin, dark fabric against dark backgrounds, or rapid exposure changes all create situations where the algorithm cannot reliably classify motion.

Thinking through these issues forced me to understand how computer vision actually sees the world:  
**not as objects, but as fluctuating brightness values over time.**

Almost every improvement —  
- adding EMA,  
- adding the ring buffer,  
- adding decay,  
- adding median filtering —  

was about **shaping that ambiguity rather than eliminating it**.

With more time, I would explore alternative motion descriptors that rely less on raw brightness differences.  
Optical flow or background subtraction models like MOG2 would allow the system to understand motion even when the subject and background share similar luminance.

Another direction would be to replace the binary threshold with adaptive or per-pixel thresholds, making the system more robust in dynamic environments.

A final step could involve experimenting with depth sensing or body-tracking to let the system distinguish between overlapping limbs, producing more articulate ghost trails.

But for this stage, the current method strikes a balance between realistic time frames, legibility, and expressive behaviour that strengthens the concept:  
**a computational memory of movement, fading and accumulating like traces left in a room.**
