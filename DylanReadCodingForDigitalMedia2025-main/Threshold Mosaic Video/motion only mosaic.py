import os, glob, random
import cv2
import numpy as np


INPUT_VIDEO   = "/Users/stonesavage/Desktop/Coding for media/854100-hd_1920_1080_25fps.mp4"
OUTPUT_VIDEO  = "mosaic_motionOnly.mp4"
DATASET_DIR   = "/Users/stonesavage/Desktop/Coding for media/Code/images"

# Mosaic grid resolution
GRID_W, GRID_H = 128, 64

# Tile size in pixels
TILE_W, TILE_H = 64, 64

# Output video resolution
OUTPUT_W, OUTPUT_H = 1080, 720
OUTPUT_INTERP = cv2.INTER_NEAREST

# Motion detector parameters
BG_ALPHA         = 0.02
MOTION_THRESHOLD = 32

random.seed(36)
np.random.seed(36)


def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    out = []
    for e in exts:
        out.extend(glob.glob(os.path.join(folder, e)))
    return sorted(out)


def load_tiles(path):
    files = list_images(path)
    imgs = []
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (TILE_W, TILE_H), cv2.INTER_AREA)
        imgs.append(im)
    return np.stack(imgs)


def main():
    tiles = load_tiles(DATASET_DIR)
    N = tiles.shape[0]

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (OUTPUT_W, OUTPUT_H)
    )

    # start with random tiles
    current_idx = np.random.randint(0, N, (GRID_H, GRID_W))

    # background model
    background = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # downsample
        small = cv2.resize(rgb, (GRID_W, GRID_H))

        # motion detection
        if background is None:
            background = small.astype(np.float32)

        cv2.accumulateWeighted(small, background, BG_ALPHA)
        bg_uint8 = background.astype(np.uint8)

        diff = cv2.absdiff(small, bg_uint8)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

        _, motion_mask = cv2.threshold(gray, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        motion_mask = (motion_mask > 0)

        # build frame
        mosaic = np.zeros((GRID_H * TILE_H, GRID_W * TILE_W, 3), np.uint8)

        for y in range(GRID_H):
            for x in range(GRID_W):

                # update tile only where motion occurs
                if motion_mask[y, x]:
                    current_idx[y, x] = random.randint(0, N - 1)

                # paste tile
                oy, ox = y * TILE_H, x * TILE_W
                mosaic[oy:oy + TILE_H, ox:ox + TILE_W] = tiles[current_idx[y, x]]

        out = cv2.resize(mosaic, (OUTPUT_W, OUTPUT_H), interpolation=OUTPUT_INTERP)
        writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    cap.release()
    writer.release()
    print("[DONE] Saved:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
