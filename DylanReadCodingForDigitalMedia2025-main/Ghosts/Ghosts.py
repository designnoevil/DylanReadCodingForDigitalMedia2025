import atexit
import cv2
import numpy as np
from dorothy import Dorothy

WINDOW_SIZE = 30              # ring buffer length
BG_ALPHA = 0.02               # EMA learning rate
MOVEMENT_THRESHOLD = 30       # binary threshold for motion detection
GHOST_STRENGTH = 0.6          # opacity of ghost layer
CAMERA_INDEX = 0


class TemporalGhosts:
    def __init__(self, dot: Dorothy, memory: int):
        self.dot = dot
        self.memory = memory

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.w = dot.width
        self.h = dot.height

        # ring buffer of past masks
        self.motion_buffer = np.zeros(
            (self.memory, self.h, self.w), dtype=np.uint8
        )
        self.ptr = 0
        self.count = 0

        # background model (initialized on first frame)
        self.background = None

    def setup(self):
        # set solid black canvas background
        self.dot.background((0, 0, 0))

    def draw(self):
        ret, frame = self.cap.read()

        # grab live frame and resize to canvas
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.w, self.h))

        # seed background model
        if self.background is None:
            self.background = rgb.astype(np.float32)

        # update exponential moving-average background
        cv2.accumulateWeighted(rgb, self.background, BG_ALPHA)
        bg_uint8 = self.background.astype(np.uint8)

        # motion = abs difference from EMA background
        diff = cv2.absdiff(rgb, bg_uint8)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

        # threshold to binary motion mask
        _, mask = cv2.threshold(gray, MOVEMENT_THRESHOLD, 255, cv2.THRESH_BINARY)

        # remove isolated noise pixels
        mask = cv2.medianBlur(mask, 5)

        # write new mask into ring buffer
        self.motion_buffer[self.ptr] = mask
        self.ptr = (self.ptr + 1) % self.memory
        self.count = min(self.count + 1, self.memory)

        # age weighted ghost layer 
        ghost_layer = np.zeros((self.h, self.w), dtype=np.float32)
        for i in range(self.count):
            index = (self.ptr - 1 - i) % self.memory
            age_factor = 1.0 - (i / self.count)
            ghost_layer += (self.motion_buffer[index] / 255.0) * age_factor

        ghost_layer = np.clip(ghost_layer, 0.0, 1.0)

        # ghost intensity → alpha mask
        ghost_alpha = np.clip(
            ghost_layer * GHOST_STRENGTH,
            0.0,
            GHOST_STRENGTH
        )[..., None]

        # blend based on ghost alpha
        base = rgb.astype(np.float32)
        white = np.full_like(base, 255.0, dtype=np.float32)

        blended = ((1.0 - ghost_alpha) * base + ghost_alpha * white).astype(np.uint8)

        # display result
        self.dot.canvas = blended

    def close(self):
        # release webcam safely
        if self.cap:
            self.cap.release()

def main():
        dot = Dorothy(width=960, height=540)
        sketch = TemporalGhosts(dot, WINDOW_SIZE)

        def setup():
            sketch.setup()

        def draw():
            sketch.draw()

        def on_exit():
            sketch.close()

        dot.on_exit = on_exit
        atexit.register(on_exit)

        dot.start_loop(setup, draw)

if __name__ == "__main__":
    main()
