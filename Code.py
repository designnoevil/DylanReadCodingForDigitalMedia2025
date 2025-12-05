from typing import Dict, List, Tuple

import numpy as np

from dorothy import Dorothy

# Audio engine and sample preparation
class SimpleSampler:
    def __init__(self, dot: Dorothy, sample_rate: int = 22050, buffer_size: int = 128):
        # Each index holds one drum sample, its playback head, and gain
        self.samples: List[np.ndarray] = []
        self.positions: List[int] = []
        self.gains: List[float] = []

        def get_frame(size: int) -> np.ndarray:
            # Mixes any active sample slices into the outgoing buffer
            audio = np.zeros(size, dtype=np.float32)
            for idx, pos in enumerate(self.positions):
                if pos < 0:
                    continue
                sample = self.samples[idx]
                end = min(pos + size, len(sample))
                chunk = sample[pos:end] * self.gains[idx]
                audio[: len(chunk)] += chunk
                self.positions[idx] = -1 if end >= len(sample) else end
                if self.positions[idx] == -1:
                    self.gains[idx] = 0.0
            return np.clip(audio, -1.0, 1.0)

        dot.music.start_dsp_stream(get_frame, sr=sample_rate, buffer_size=buffer_size, analyse=True)

    def set_samples(self, sample_list: List[np.ndarray]) -> None:
        self.samples = [np.asarray(sample, dtype=np.float32) for sample in sample_list]
        self.positions = [-1 for _ in self.samples]
        self.gains = [0.0 for _ in self.samples]

    def trigger(self, index: int, velocity: float = 0.85) -> None:
        if 0 <= index < len(self.samples):
            self.positions[index] = 0
            self.gains[index] = max(0.0, min(1.0, float(velocity)))


def _exp_env(length: float, sr: int, decay: float) -> np.ndarray:
    # Simple exponential decay envelope used across the drum kit sounds
    steps = int(sr * length)
    if steps <= 0:
        return np.zeros(1, dtype=np.float32)
    t = np.linspace(0.0, length, steps, endpoint=False)
    return np.exp(-decay * t)


def _normalize(sample: np.ndarray, peak: float = 0.8) -> np.ndarray:
    # Keeps samples within a predictable level so they mix cleanly
    sample = np.array(sample, dtype=np.float32)
    max_val = float(np.max(np.abs(sample))) if sample.size else 0.0
    if max_val > 0:
        sample = sample * (peak / max_val)
    return sample


def build_sample_pack(sr: int) -> List[np.ndarray]:
    # Procedurally builds a small drum kit so the sequencer can run standalone
    rng = np.random.default_rng(404)
    kit = []

    env = _exp_env(0.45, sr, 6.2)
    t = np.arange(env.size) / sr
    sweep = 90.0 * np.exp(-6.0 * t) + 42.0
    tone = np.sin(2 * np.pi * np.cumsum(sweep) / sr)
    click = np.exp(-220 * t) * (rng.random(env.size) * 0.2)
    kit.append(_normalize(tone * env + click, 0.86))

    env = _exp_env(0.38, sr, 12.5)
    t = np.arange(env.size) / sr
    kit.append(_normalize(rng.normal(0.0, 1.0, env.size) * env * 0.8 + np.sin(2 * np.pi * 190 * t) * np.exp(-18 * t) * 0.4, 0.78))

    env = _exp_env(0.18, sr, 28.0)
    kit.append(_normalize(np.sign(rng.normal(0.0, 1.0, env.size)) * env * 0.6, 0.72))

    env = _exp_env(0.32, sr, 16.0)
    noise = rng.normal(0.0, 1.0, env.size)
    pulse = np.zeros_like(noise)
    for offset in (0, int(0.012 * sr), int(0.024 * sr)):
        end = min(env.size, offset + int(0.02 * sr))
        pulse[offset:end] += env[0 : end - offset]
    kit.append(_normalize(noise * env * 0.45 + pulse * 0.5, 0.75))

    env = _exp_env(0.46, sr, 7.0)
    t = np.arange(env.size) / sr
    kit.append(_normalize(np.sin(2 * np.pi * 140 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.2 * t)) * env, 0.82))

    env = _exp_env(0.8, sr, 3.4)
    base = rng.normal(0.0, 1.0, env.size)
    kit.append(_normalize(np.convolve(base, np.ones(5) / 5, mode="same") * env * 0.55, 0.7))

    env = _exp_env(0.36, sr, 10.5)
    t = np.arange(env.size) / sr
    kit.append(_normalize((np.sin(2 * np.pi * 540 * t) + 0.6 * np.sin(2 * np.pi * 810 * t)) * env, 0.8))

    env = _exp_env(0.22, sr, 22.0)
    noise = rng.normal(0.0, 1.0, env.size)
    kit.append(_normalize(noise * env * (0.5 + 0.5 * np.sin(2 * np.pi * 16 * np.arange(env.size) / env.size)), 0.68))

    env = _exp_env(1.2, sr, 1.8)
    noise = rng.normal(0.0, 1.0, env.size)
    kit.append(_normalize(np.convolve(noise, np.ones(12) / 12, mode="same") * env * 0.6, 0.72))

    return kit

# UI constants and overall layout metrics 
WIDTH, HEIGHT = 960, 675
BACKGROUND = (12, 12, 12)
LABEL_BG = (20, 20, 20)
LABEL_TEXT = (245, 245, 245)
STEP_OFF = (58, 58, 58)
STEP_ON = (255, 158, 14)
BUTTON_BG = (36, 36, 36)
BUTTON_ACTIVE = STEP_ON
BUTTON_TEXT = (235, 235, 235)
BUTTON_TEXT_DARK = (22, 22, 22)

# GRID_ROWS defines the instruments, STEP_COUNT controls loop length, and the
# audio constants keep Dorothy + the sampler in sync.
GRID_ROWS = [
    "KICK",
    "SNARE",
    "HIHAT",
    "CLAP",
    "TOM",
    "RIDE",
    "COWBELL",
    "SHAKER",
    "CRASH",
]
STEP_COUNT = 8
SAMPLE_RATE = 22050
BUFFER_SIZE = 1024

LABEL_WIDTH = 170
LABEL_GAP = 24
CELL_SIZE = 40
CELL_GAP = 10
LABEL_SCALE = 3

BUTTON_WIDTH = 120
BUTTON_HEIGHT = 36
BUTTON_GAP = 24

bpm = 110
step_millis = 60000.0 / bpm
TAP_WINDOW = 4000

dot = Dorothy(width=WIDTH, height=HEIGHT)
dot.background(BACKGROUND)

# Build the audio engine and hand it the procedurally generated drum kit
sampler = SimpleSampler(dot, sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE)
sampler.set_samples(build_sample_pack(SAMPLE_RATE))

GRID_WIDTH = 0
GRID_HEIGHT = 0
GRID_LEFT = 0
GRID_TOP = 0
BUTTON_TOP = 0
HEADER_POS = (0, 0)
BUTTON_ORDER = [
    ("start", "START"),
    ("stop", "STOP"),
    ("clear", "CLEAR"),
    ("tap", "TAP TEMPO"),
]
HEADER_TEXT = "SEQUENCE STORE"
HEADER_SCALE = 3
INSTRUCTION_TEXT = "CLICK STEPS OR TAP TEMPO"
INSTRUCTION_SCALE = 2
MOUSE_DOWN_ATTRS = ("mouse_down", "mousePressed", "mouse_pressed", "mouseDown", "mouseButton")
button_rects: Dict[str, Tuple[int, int, int, int]] = {}
INSTRUCTION_POS = (0, 0)
tap_times: List[float] = []  # Captures recent tap tempo timestamps

# 5x7 bitmap font used for all on-screen text

FONT_5X7: Dict[str, Tuple[str, ...]] = {
    " ": ("00000",) * 7,
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "00110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "11110", "10001", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "11110", "10000", "10000", "10000", "11111"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "11111", "10001", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10001", "01010", "00101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "W": ("10001", "10001", "10101", "10101", "10101", "10101", "01010"),
}


def draw_text(message: str, pos: Tuple[int, int], colour: Tuple[int, int, int] = (220, 220, 220), scale: int = 4) -> None:
    # Renders bitmap text by stamping scaled rectangles into the Dorothy canvas
    x_cursor = int(pos[0])
    y_cursor = int(pos[1])
    dot.fill(colour)
    chars = message.upper()
    for idx, ch in enumerate(chars):
        glyph = FONT_5X7.get(ch, FONT_5X7[" "])
        glyph_width = len(glyph[0])
        for row, row_data in enumerate(glyph):
            for col, bit in enumerate(row_data):
                if bit != "1":
                    continue
                x0 = x_cursor + col * scale
                y0 = y_cursor + row * scale
                dot.rectangle((x0, y0), (x0 + scale, y0 + scale))
        x_cursor += glyph_width * scale
        if idx < len(chars) - 1:
            x_cursor += scale


def measure_text(message: str, scale: int) -> Tuple[int, int]:
    # Calculates how much space a message will occupy before rendering
    chars = message.upper()
    width = 0
    for idx, ch in enumerate(chars):
        glyph = FONT_5X7.get(ch, FONT_5X7[" "])
        glyph_width = len(glyph[0])
        width += glyph_width * scale
        if idx < len(chars) - 1:
            width += scale
    height = len(FONT_5X7["A"]) * scale
    return width, height


def compute_layout() -> None:
    # Pre-computes all drawing offsets so runtime work stays minimal
    global GRID_WIDTH, GRID_HEIGHT, GRID_LEFT, GRID_TOP, BUTTON_TOP, HEADER_POS, INSTRUCTION_POS, button_rects
    GRID_WIDTH = LABEL_WIDTH + LABEL_GAP + STEP_COUNT * CELL_SIZE + (STEP_COUNT - 1) * CELL_GAP
    GRID_HEIGHT = len(GRID_ROWS) * CELL_SIZE + (len(GRID_ROWS) - 1) * CELL_GAP
    GRID_LEFT = (WIDTH - GRID_WIDTH) // 2
    header_y = 54
    HEADER_POS = (GRID_LEFT, header_y)
    header_width, header_height = measure_text(HEADER_TEXT, HEADER_SCALE)
    instruction_y = header_y + header_height + 12
    INSTRUCTION_POS = (GRID_LEFT, instruction_y)
    inst_height = measure_text(INSTRUCTION_TEXT, INSTRUCTION_SCALE)[1]
    GRID_TOP = max(instruction_y + inst_height + 28, (HEIGHT - GRID_HEIGHT) // 2)
    BUTTON_TOP = header_y + (header_height - BUTTON_HEIGHT) // 2
    start_x = HEADER_POS[0] + header_width + 40
    button_rects.clear()
    for idx, (key, _) in enumerate(BUTTON_ORDER):
        x1 = start_x + idx * (BUTTON_WIDTH + BUTTON_GAP)
        button_rects[key] = (x1, BUTTON_TOP, x1 + BUTTON_WIDTH, BUTTON_TOP + BUTTON_HEIGHT)


compute_layout()

grid = [[False] * STEP_COUNT for _ in GRID_ROWS]

# Sequencer runtime state
current_step = 0            # Column that is currently playing/highlighted
last_step_time = 0.0        # Timestamp of the previous step advance
mouse_was_down = False      # Tracks click transitions so toggles fire once
is_playing = True           # Transport flag

def point_in_rect(x: float, y: float, rect: Tuple[int, int, int, int]) -> bool:
    # Hit-test helper for buttons and grid cells
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_button(label: str, rect: Tuple[int, int, int, int], active: bool, hover: bool) -> None:
    # Draws one of the transport/control buttons with hover + active feedback
    x1, y1, x2, y2 = rect
    base_colour = BUTTON_ACTIVE if active else BUTTON_BG
    if hover:
        base_colour = highlight(base_colour, 28 if active else 36)
    dot.fill(base_colour)
    dot.rectangle((x1, y1), (x2, y2))
    text_colour = BUTTON_TEXT_DARK if active else BUTTON_TEXT
    scale = 3
    text_width, text_height = measure_text(label, scale)
    while text_width > BUTTON_WIDTH - 6 and scale > 1:
        scale -= 1
        text_width, text_height = measure_text(label, scale)
    tx = x1 + (BUTTON_WIDTH - text_width) // 2
    ty = y1 + (BUTTON_HEIGHT - text_height) // 2
    draw_text(label, (tx, ty), colour=text_colour, scale=scale)


def clear_pattern() -> None:
    # Resets every step in the grid
    for row in grid:
        row[:] = [False] * STEP_COUNT


def set_bpm(value: float) -> None:
    # Quantises and applies an external BPM change
    global bpm, step_millis, last_step_time
    bpm = max(40, min(240, int(round(value / 5.0) * 5)))
    step_millis = 60000.0 / bpm
    last_step_time = read_millis()


def tap_tempo() -> None:
    # Collects tap timestamps and derives a tempo from their average spacing
    now = read_millis()
    tap_times.append(now)
    while tap_times and now - tap_times[0] > TAP_WINDOW:
        tap_times.pop(0)
    if len(tap_times) < 2:
        return
    intervals = [t2 - t1 for t1, t2 in zip(tap_times, tap_times[1:])]
    avg = sum(intervals) / len(intervals)
    if avg <= 0:
        return
    set_bpm(60000.0 / avg)


def handle_button(name: str) -> None:
    # Routes button presses to the appropriate action
    global is_playing, last_step_time
    if name == "start":
        last_step_time = read_millis()
        if not is_playing:
            is_playing = True
            trigger_step(current_step)
    elif name == "stop":
        is_playing = False
    elif name == "clear":
        clear_pattern()
    elif name == "tap":
        tap_tempo()


def draw_buttons(mouse_x: float, mouse_y: float) -> None:
    # Renders the whole row of buttons with per-button state
    active_map = {"start": is_playing, "stop": not is_playing}
    for key, label in BUTTON_ORDER:
        rect = button_rects[key]
        draw_button(label, rect, active_map.get(key, False), point_in_rect(mouse_x, mouse_y, rect))


def highlight(colour: Tuple[int, int, int], lift: int = 38) -> Tuple[int, int, int]:
    # Lightens a colour for hover/active feedback
    return tuple(min(255, c + lift) for c in colour)


def read_mouse() -> Tuple[bool, float, float]:
    # Normalises different Dorothy mouse attribute names into a single tuple
    down = any(bool(getattr(dot, attr, 0)) for attr in MOUSE_DOWN_ATTRS)
    for nx, ny in (("mouse_x", "mouse_y"), ("mouseX", "mouseY")):
        if hasattr(dot, nx) and hasattr(dot, ny):
            return down, float(getattr(dot, nx)), float(getattr(dot, ny))
    return down, 0.0, 0.0


def locate_cell(mx: float, my: float) -> Tuple[int, int] | None:
    # Converts a mouse position into a grid row/column index if possible
    start_x = GRID_LEFT + LABEL_WIDTH + LABEL_GAP
    for row in range(len(GRID_ROWS)):
        row_y = GRID_TOP + row * (CELL_SIZE + CELL_GAP)
        if not (row_y <= my <= row_y + CELL_SIZE):
            continue
        for col in range(STEP_COUNT):
            cell_x = start_x + col * (CELL_SIZE + CELL_GAP)
            if cell_x <= mx <= cell_x + CELL_SIZE:
                return row, col
    return None


def trigger_step(step_index: int) -> None:
    # Fires any samples whose row is active for the current column
    for row_index, row in enumerate(grid):
        if row[step_index]:
            sampler.trigger(row_index)


def read_millis() -> float:
    # Safely reads Dorothy's clock, tolerating different API shapes
    value = getattr(dot, "millis", 0)
    if callable(value):
        try:
            return float(value())
        except Exception:
            return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def draw_grid(active_col: int) -> None:
    # Draws the main sequencer grid plus row labels
    start_x = GRID_LEFT + LABEL_WIDTH + LABEL_GAP
    label_height = LABEL_SCALE * len(FONT_5X7["A"])
    for row_index, name in enumerate(GRID_ROWS):
        row_y = GRID_TOP + row_index * (CELL_SIZE + CELL_GAP)

        dot.fill(LABEL_BG)
        dot.rectangle((GRID_LEFT, row_y), (GRID_LEFT + LABEL_WIDTH, row_y + CELL_SIZE))
        text_y = int(row_y + (CELL_SIZE - label_height) / 2)
        draw_text(name, (GRID_LEFT + 18, text_y), colour=LABEL_TEXT, scale=LABEL_SCALE)

        for col in range(STEP_COUNT):
            cell_x = start_x + col * (CELL_SIZE + CELL_GAP)
            colour = STEP_ON if grid[row_index][col] else STEP_OFF
            if col == active_col:
                colour = highlight(colour, 54 if grid[row_index][col] else 36)
            dot.fill(colour)
            dot.rectangle((cell_x, row_y), (cell_x + CELL_SIZE, row_y + CELL_SIZE))


def setup() -> None:
    # Initialisation for Dorothy's draw loop
    global last_step_time
    dot.background(BACKGROUND)
    last_step_time = read_millis()
    if is_playing:
        trigger_step(current_step)


def draw() -> None:
    # Main frame loop
    global current_step, last_step_time, mouse_was_down
    now = read_millis()
    if is_playing:
        while now - last_step_time >= step_millis:
            # Step in fixed increments so downbeat timing stays stable
            last_step_time += step_millis
            current_step = (current_step + 1) % STEP_COUNT
            trigger_step(current_step)
    else:
        last_step_time = now

    mouse_down, mx, my = read_mouse()
    if mouse_down and not mouse_was_down:
        # Buttons get priority; fall back to toggling grid cells
        for key, _ in BUTTON_ORDER:
            if point_in_rect(mx, my, button_rects[key]):
                handle_button(key)
                break
        else:
            hit = locate_cell(mx, my)
            if hit:
                r, c = hit
                grid[r][c] = not grid[r][c]
    mouse_was_down = mouse_down

    dot.background(BACKGROUND)
    draw_text(HEADER_TEXT, HEADER_POS, colour=LABEL_TEXT, scale=HEADER_SCALE)
    draw_buttons(mx, my)
    draw_text(INSTRUCTION_TEXT, INSTRUCTION_POS, colour=(180, 180, 180), scale=INSTRUCTION_SCALE)
    draw_grid(current_step)
    bpm_y = GRID_TOP + GRID_HEIGHT + 24
    draw_text(f"BPM {bpm}", (GRID_LEFT, bpm_y), colour=(180, 180, 180), scale=2)


if __name__ == "__main__":
    try:
        dot.start_loop(setup, draw)
    except KeyboardInterrupt:
        pass
