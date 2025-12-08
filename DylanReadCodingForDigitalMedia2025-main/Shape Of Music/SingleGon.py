from dorothy import Dorothy
import numpy as np
import os
import math

AUDIO_FOLDER = "/Users/stonesavage/Desktop/Coding for media/data/MP3s"

# Pick 1st file and works for muptipule file types
files = sorted(os.listdir(AUDIO_FOLDER))
audio_files = [f for f in files if f.lower().endswith((".wav", ".mp3", ".flac"))]

file_path = os.path.join(AUDIO_FOLDER, audio_files[0])

dot = Dorothy(width=1600, height=900)

env = 0.0          # smoothed amplitude envelope
points = 3.0       # polygon vertex count
angle = 0.0        # rotation accumulator

def attack_release(current, target, attack=0.3, release=0.1):
    coeff = attack if target > current else release
    return current + coeff * (target - current)

def draw_polygon(n, angle):
    cx, cy = dot.width / 2, dot.height / 2
    r = 200

    verts = [
        (
            cx + math.cos(i / n * math.tau + angle) * r,
            cy + math.sin(i / n * math.tau + angle) * r
        )
        for i in range(n)]

    # Draw edges
    for i in range(n):
        dot.line(verts[i], verts[(i + 1) % n])

def setup():
    # Start file stream with FFT 
    dot.music.start_file_stream(file_path, fft_size=512, buffer_size=512)
    dot.music.play()

    dot.background((0, 0, 0))

    dot.stroke((255, 255, 255, 255))   # draw lines in white
    dot.no_fill()

def draw():
    global env, points, angle

    # Audio analysis 
    fft = dot.music.fft()                # 512-bin magnitude spectrum
    amp = dot.music.amplitude()          # amplitude of current frame

    # Envelope smoothing
    env = attack_release(env, amp)
    env = 0.3 * env + 0.7 * amp          # extra inertia

    # Average spectral energy
    energy = float(np.mean(fft)) if len(fft) else 0.0

    # Polygon complexity
    target = 2 + int(energy * 10)        # map energy > number of sides
    points += (target - points) * 0.2    # interpolation for smooth change

    # Trails 
    dot.fill((0, 0, 0, 25))
    dot.rectangle((0, 0), (dot.width, dot.height))

    # Rotation 
    angle += 0.01

    # Draw polygon 
    draw_polygon(int(points), angle)

dot.start_loop(setup, draw)
