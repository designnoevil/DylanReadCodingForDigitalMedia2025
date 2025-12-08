# Shape of Music

I come from a background in wood and metal workshop practice, and design for production, so I’m often thinking about human movement and workflow. Over the summer, the corse leader for Product Design I work with, at the University of Brighton, was developing jigs for Windsor style chairs made from timber sourced from local woodland. He moved through the workshop in a kind of flow state, dancing as he used his jigs to cut and profile trunks into form. We discussed how the outcomes may differ, yet choreography of movement is a shared foundation that connects all creative arts. This emphasis on gesture and embodied making resonates with the embodied interaction perspectives of Penny (2017) and Manning (2009), who describe movement as a primary site of meaning-making.

The cutting and splicing of tape not only created a new sound world but also introduced a new set of movements involved in making audio compared to the movements of playing an instrument. Developed from my reflections on the themes of musique concrète and the idea of working directly with recorded sound as material (Schaeffer, 1966). In class, we treated audio as a list of numerical samples that could be sliced, reversed, and reassembled. I became interested in what might happen if those same data structures were used not to edit sound, but to draw with it. In this project, I wanted to communicate the movement of sound through shape, to represent the different gestures of motion that give rise to different kinds of sound: the shape of music.

---

## Early Prototype

The early version of the project loaded MP3s from a folder and streamed them through Dorothy:

```python
AUDIO_FOLDER = "/Users/stonesavage/Desktop/Coding for media/data/MP3s"
files = sorted(os.listdir(AUDIO_FOLDER))
audio_files = [f for f in files if f.lower().endswith((".wav", ".mp3", ".aiff", ".flac"))]
file_path = os.path.join(AUDIO_FOLDER, audio_files[0])

def setup():
    dot.music.start_file_stream(file_path, fft_size=512, buffer_size=512)
    dot.music.play()
```

Dorothy provides both the RMS amplitude and a NumPy array of FFT magnitudes. start_file_stream internally loads and decodes the audio file and continuously feeds buffers to analysis.

---

## Single-Polygon Prototype

I set up a single polygon driven by the average energy of the spectrum:

```python
env = 0.0
points = 3.0
angle = 0.0
```

env stores a smoothed amplitude envelope, points determines polygon vertices, and angle provides rotation.

---

## Envelope Behaviour

```python
def attack_release(current, target, attack=0.3, release=0.1):
    coeff = attack if target > current else release
    return current + coeff * (target - current)
```

This attack–release envelope helps stabilise the visuals by smoothing rapid changes.

---

## Drawing the Polygon

```python
def draw_polygon(n, angle):
    cx, cy = dot.width / 2, dot.height / 2
    r = 200
    verts = [(cx + math.cos(i / n * math.tau + angle) * r,
              cy + math.sin(i / n * math.tau + angle) * r)
             for i in range(n)]
    for i in range(n):
        dot.line(verts[i], verts[(i + 1) % n])
```

I asked ChatGPT for this maths, I knew what I wanted but wasn’t sure on the maths for how to make shifting morphing perfect polygons. The polygon is generated using polar coordinates. Dividing i by n distributes vertices around a circle, and adding angle rotates the shape.

---

## FFT Mapping

```python
fft = dot.music.fft()
amp = dot.music.amplitude()
env = attack_release(env, amp)
env = 0.3 * env + 0.7 * amp
energy = float(np.mean(fft)) if len(fft) else 0.0
```

Taking the FFT mean provides a coarse descriptor used to drive complexity. The idea that spectral structures can serve as compositional or structural material aligns with Roads’ discussion of microsound and spectral organisation (Roads, 2001).

---

## Driving Polygon Complexity

```python
target = 2 + int(energy * 10)
points += (target - points) * 0.2
```

As FFT mean increases, the polygon gains more sides.

---

## Accumulating Trails

```python
dot.fill((0, 0, 0, 25))
dot.rectangle((0, 0), (dot.width, dot.height))
```

A semi-transparent rectangle fades previous frames to create motion trails.

---

## Final Behaviour of the Prototype

```python
angle += 0.01
draw_polygon(int(points), angle)
```

The prototype produces a single rotating polygon that expands and contracts with spectral energy.


![Shape of Music Demo](Untitled.gif)

---

## Revisiting Audio Handling

To visualise live system audio (Spotify), I switched to VB-Cable and added device detection.
I didn’t want to look up IDs so the code searches for the cable inserted.

```python
def choose_audio_source(dot):
    devices = sd.query_devices()
    cable_id = None
    for idx, dev in enumerate(devices):
        name = str(dev.get("name", "")).lower()
        if "cable" in name:
            cable_id = idx
            break
```

If VB-Cable is unavailable, the system falls back to local files.

---

## Transition to the Multi-Layer System

I designed a system with five concentric polygons reading overlapping FFT bands:

```python
BANDS = [
    (0.0, 1.00),
    (0.0, 0.90),
    (0.0, 0.80),
    (0.0, 0.70),
    (0.0, 0.60),
]
```

With testing, I found that overlapping bands produced a more collective motion. As each band is different, they each represent a different player in the choreography; the overlap gives the visual impression of cohesion. They move like a group, each with their own part but with an obvious connection to one another, performing together. This interest in collective gesture echoes the relational emphasis in Manning’s writing on movement and group dynamics (Manning, 2009).

---

## Layer Styles

```python
STYLES = [
    {"scale": 0.18, "hue": 0.02, "dir":  1},
    {"scale": 0.40, "hue": 0.18, "dir": -1},
    {"scale": 0.70, "hue": 0.38, "dir":  1},
    {"scale": 0.95, "hue": 0.62, "dir": -1},
    {"scale": 1.20, "hue": 0.82, "dir":  1},
]
```

Alternating rotation directions introduce interference patterns, which added a pleasant layer of complexity to the visuals with minimal code complexity.

---

## Per-Layer State

```python
layers = [{"angle": 0.0, "energy": 0.0, "points": 3.0} for _ in BANDS]
```

Each polygon maintains its own rotation, energy, and point count.

---

## draw_layers snippet

```python
start = int(band[0] * n)
end   = int(band[1] * n)
slice_vals = fft[start:end]
band_mean = float(np.mean(slice_vals)) if len(slice_vals) else 0.0
```

Local band energy differentiates layers; global energy ties the system together.

---

## Envelope and Energy Stability

Two smoothing mechanisms prevent jitter and give the visuals inertia.

### Hybrid Amplitude Envelope

```python
global_env = attack_release(global_env, amp, ENV_ATTACK, ENV_RELEASE)
global_env = ENV_SMOOTH * global_env + (1 - ENV_SMOOTH) * amp
```

This produces a shared breathing value.

### Energy Blending

```python
e_raw = 0.75 * band_mean + 0.25 * global_mean
state["energy"] = state["energy"] * 0.4 + e_raw * 0.25
```

Mixing local and global energies makes the motion cohesive. By setting each band up as an individual but tying them together in different ways, I began to see more and more cohesion in the polygons representing each frequency band. The basic idea already worked in the prototype; scaling it to multiple bands became an experiment in balancing the freedom of individual expression with the empathy of a collective responding together. This balance between individual motion and shared structure echoes Penny’s discussion of embodied systems, where coordinated behaviour emerges from distributed, gestural processes rather than centralised control (Penny, 2017).

---

## Rotation From Energy

```python
speed = ROTATE_BASE + state["energy"] * ROTATE_GAIN
state["angle"] += speed * style["dir"]
```

Energy determines rotation speed; alternating directions generate complex motion.

---

## Smoothly Morphing Polygon Complexity

```python
drive = state["energy"] ** POINT_CURVE
target_pts = MIN_POINTS + drive * (MAX_POINTS - MIN_POINTS)
state["points"] += (target_pts - state["points"]) * POINT_SMOOTH
```

---

## Rhythm Through Envelope Slope

```python
slope = max(0.0, global_env - prev_env)
beat_imp = min(1.0, slope * 20.0)
beat_env = beat_env * 0.85 + beat_imp
```

A rapid rise in amplitude produces a beat impulse.

---

## Order of Operations Inside draw()

Dorothy calls the draw function every frame. The system updates audio-derived variables before drawing:

```python
fft = dot.music.fft()
amp = dot.music.amplitude()
```

After computing the beat impulse and updating trails:

```python
dot.fill((0, 0, 0, TRAIL_ALPHA))
dot.rectangle((0, 0), (dot.width, dot.height))
draw_layers(loud, fft, beat_env)
```

---

## Colour Mapping

To avoid harsh RGB transitions, I moved to HSV colour space:

```python
def hsv_to_rgb(h, s, v):
    ...
```

HSV allowed hue to cycle with rotation, saturation to increase with energy, brightness to expand with loudness

```python
hue = (style["hue"] + 0.18 * math.sin(state["angle"] * 0.4)) % 1
sat = 0.6 + 0.25 * state["energy"]
val = 0.75 + 0.2 * loudness
```

Hue shifts create slow colour motion; brightness responds to volume.

---

## Trail System as Visual Memory

```python
dot.fill((0, 0, 0, TRAIL_ALPHA))
dot.rectangle((0, 0), (dot.width, dot.height))
```
This draws a semi-transparent black layer over the entire canvas every frame.
Because the layer is not fully opaque, it darkens the previous frame rather than erasing it. Older drawings fade gradually, while new polygons are drawn at full brightness.

Applied to all polygons, this reinforces the momentum of movement.

---

## Behaviour at Low Signal Levels

```python
signal = max(loudness, state["energy"], beat)
count = 2 if signal < LINE_THRESH else max(2, int(round(state["points"])))
```

When energy is low, the polygon collapses into a line aligned to its rotation angle.
When energy rises, the polygon reconstructs itself.
This emerged accidentally with playing around but became central.

---

## Energy Clamping for Numerical Stability

After smoothing, each polygon’s energy is restricted to 0–1:

```python
state["energy"] = min(1.0, max(0.0, state["energy"]))
```

Clamping ensures the visual system remains stable. Without clamping, high energy could push rotation, scale, or colour into extremes. Clamping creates a consistent energy budget so the choreography remains expressive but contained. Discovering the need for clamping was part of understanding how digital motion needs boundaries to remain readable.

---

## Vertex Generation

When polygon complexity is above the threshold:

```python
ang = t * math.tau + state["angle"]
```

Rotation is integrated into vertex generation. Each vertex is generated in its rotated position, giving internal coherence.

The layering of these morphing polygons, each driven by overlapping FFT bands, each with its own hue cycle, each accelerating and decelerating with energy, creates a choreography of shapes that behave as a unified field rather than separate entities. This relational motion echoes Manning’s argument that movement emerges through the dynamic interplay of forms acting within the same field of action, rather than from isolated bodies moving independently (Manning, 2009).

---

## Reflections on Gesture, Material, and Computation

The project draws from the idea that sound is motion. By translating sound into geometric behaviour, I treated audio as a tool capable of shaping form.

In musique concrète, recorded fragments are manipulated physically: cut, spliced, reversed, stretched. The body’s movement becomes part of the sound. In my system, sound becomes the mover. Its acceleration becomes rotation. Its loudness becomes breath. Its energy divides into five overlapping motions. The trails bake inertia into movements.

The shape of music.
![Five Shapes Demo](5%20shapes.gif)

[demo video](https://www.youtube.com/watch?v=SiqGvRQfKkk)
---

## References

Manning, E. (2009) Relational Movement: Choreography as Mobile Architecture. Cambridge, MA: MIT Press.
(If you prefer: Relationscapes: Movement, Art, Philosophy — tell me which edition your course uses.)

Penny, S. (2017) Making Sense: Cognition, Computing, Art, and Embodiment. Cambridge, MA: MIT Press.

Roads, C. (2001) Microsound. Cambridge, MA: MIT Press.

Schaeffer, P. (1966) Treatise on Musical Objects: An Essay across Disciplines. Paris: Éditions du Seuil.
(English translation published 2017, University of California Press — choose whichever one your course requires.)
