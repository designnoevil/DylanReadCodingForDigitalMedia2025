from dorothy import Dorothy
import numpy as np
import sounddevice as sd
import math
import os

FALLBACK_DIR = "/Users/stonesavage/Desktop/Coding for media/data/MP3s"

def choose_audio_source(dot):
    """Select VB-Cable if installed, otherwise choose first audio file."""
    devices = sd.query_devices()

    # Look for VB-Cable by substring.
    cable_id = None
    for idx, dev in enumerate(devices):
        name = str(dev.get("name", "")).lower()
        if "cable" in name:
            cable_id = idx
            break

    if cable_id is not None:
        # Use live VB-Cable input
        dot.music.start_device_stream(cable_id, fft_size=512, buffer_size=512)
        return

    # Otherwise: fallback to 1st audio file found in directory
    files = sorted(os.listdir(FALLBACK_DIR))
    audio_files = [f for f in files if f.lower().endswith((".wav", ".mp3", ".aiff", ".flac"))]

    file_path = os.path.join(FALLBACK_DIR, audio_files[0])
    dot.music.start_file_stream(file_path, fft_size=512, buffer_size=512)
    dot.music.play()

# Visual Mapping
def hsv_to_rgb(h, s, v):
    # Convert HSV to RGB
    h = float(h) % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i %= 6
    if   i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else:        r, g, b = v, p, q
    return int(r*255), int(g*255), int(b*255)

def attack_release(current, target, attack, release):
    coeff = attack if target > current else release
    return current + coeff * (target - current)

# Tuning 
ENV_ATTACK = 0.40
ENV_RELEASE = 0.12
ENV_SMOOTH = 0.35

BASE_RADIUS = 140
LOUD_GAIN = 260
ENERGY_GAIN = 0.6
BEAT_GAIN = 0.7

ROTATE_BASE = 0.015
ROTATE_GAIN = 0.20

MIN_POINTS = 2
MAX_POINTS = 12
POINT_SMOOTH = 0.22
POINT_CURVE = 1.20
LINE_THRESH = 0.001

TRAIL_ALPHA = 30

# Frequency bands 
BANDS = [
    (0.0, 1.00),
    (0.0, 0.90),
    (0.0, 0.80),
    (0.0, 0.70),
    (0.0, 0.60),
]

#Base style for each band polygon, scale for nesting, colour, rotation direction
STYLES = [
    {"scale": 0.18, "hue": 0.02, "dir":  1},
    {"scale": 0.40, "hue": 0.18, "dir": -1},
    {"scale": 0.70, "hue": 0.38, "dir":  1},
    {"scale": 0.95, "hue": 0.62, "dir": -1},
    {"scale": 1.20, "hue": 0.82, "dir":  1},
]

dot = Dorothy(1600, 900)

global_env = 0.0
prev_env = 0.0
beat_env  = 0.0

# Each ring keeps its own rotation, energy and polygon complexity
layers = [{"angle": 0.0, "energy": 0.0, "points": 2.0} for _ in BANDS]

# Draw
def draw_layers(loudness, fft, beat):

    cx, cy = dot.width / 2, dot.height / 2
    n = len(fft)
    global_mean = float(np.mean(fft)) if n else 0.0

    # Base scale for all shapes, modulated by loudness and beat
    base_radius = (BASE_RADIUS + loudness * LOUD_GAIN) * (1 + beat * BEAT_GAIN)

    # Process each shape separately
    for (band, style, state) in zip(BANDS, STYLES, layers):

        # Select the frequency range from the FFT array
        start = int(band[0] * n)
        end   = int(band[1] * n)
        end   = max(start + 1, min(end, n))
        slice_vals = fft[start:end]

        band_mean = float(np.mean(slice_vals)) if len(slice_vals) else 0.0

        # Blend local band energy with global average for motion sustain 
        e_raw = 0.75 * band_mean + 0.25 * global_mean
        state["energy"] = state["energy"] * 0.4 + e_raw * 0.25
        state["energy"] = min(1.0, max(0.0, state["energy"]))

        # Rotation
        speed = ROTATE_BASE + state["energy"] * ROTATE_GAIN
        state["angle"] += speed * style["dir"]

        # Radius growth
        r = base_radius * style["scale"] * (1 + state["energy"] * ENERGY_GAIN)

        # Polygon complexity 
        drive = state["energy"] ** POINT_CURVE
        target_pts = MIN_POINTS + drive * (MAX_POINTS - MIN_POINTS)
        state["points"] += (target_pts - state["points"]) * POINT_SMOOTH

        # Decide shape complexity
        signal = max(loudness, state["energy"], beat)
        count = 2 if signal < LINE_THRESH else max(2, int(round(state["points"])))

        # Colour mapping 
        hue = (style["hue"] + 0.18 * math.sin(state["angle"] * 0.4)) % 1
        sat = 0.6 + 0.25 * state["energy"]
        val = 0.75 + 0.2 * loudness
        r_col, g_col, b_col = hsv_to_rgb(hue, sat, min(1, val))
        alpha = int(120 + 110 * signal)

        dot.stroke((r_col, g_col, b_col, alpha))
        dot.no_fill()

        # Draw polygon or  line depending on count
        if count == 2:
            # A line through the centre
            a = state["angle"]
            x1, y1 = cx + math.cos(a) * r, cy + math.sin(a) * r
            x2, y2 = cx - math.cos(a) * r, cy - math.sin(a) * r
            dot.line((x1, y1), (x2, y2))
        else:
            verts = []
            for i in range(count):
                t = i / count
                ang = t * math.tau + state["angle"]
                verts.append((cx + math.cos(ang) * r, cy + math.sin(ang) * r))

            # Connect polygon vertices
            for i in range(count):
                dot.line(verts[i], verts[(i + 1) % count])

# Draw loop
def setup():
    choose_audio_source(dot)
    dot.background((0, 0, 0))

def draw():
    global global_env, prev_env, beat_env

    # Real-time FFT magnitudes and amplitude
    fft = dot.music.fft()
    amp = dot.music.amplitude()

    # Smooth amplitude envelope
    global_env = attack_release(global_env, amp, ENV_ATTACK, ENV_RELEASE)
    global_env = ENV_SMOOTH * global_env + (1 - ENV_SMOOTH) * amp
    loud = min(1.0, max(0.0, global_env * 3.0))

    # Beat estimation using slope of envelope
    slope = max(0.0, global_env - prev_env)
    prev_env = global_env
    beat_imp = min(1.0, slope * 20.0)
    beat_env = beat_env * 0.85 + beat_imp

    # Draw a transparent black rectangle over the entire canvas every frame
    dot.fill((0, 0, 0, TRAIL_ALPHA))
    dot.rectangle((0, 0), (dot.width, dot.height))

    draw_layers(loud, fft, beat_env)

dot.start_loop(setup, draw)
