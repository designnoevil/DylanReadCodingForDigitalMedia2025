import os, glob, random
import cv2
import numpy as np


INPUT_VIDEO   = "/Users/stonesavage/Desktop/Coding for media/854100-hd_1920_1080_25fps.mp4"
OUTPUT_VIDEO  = "mosaic_basicMotion_colourCleanup_ORlogic.mp4"
DATASET_DIR   = "/Users/stonesavage/Desktop/Coding for media/Code/images"

# Mosaic grid resolution (in tiles)
GRID_W, GRID_H = 128, 64

# Tile size in pixels
TILE_W, TILE_H = 64, 64

# Output video resolution
OUTPUT_W, OUTPUT_H = 1080, 720
OUTPUT_INTERP = cv2.INTER_NEAREST   # upsampling mode

# Tile selection behaviour
TOPK = 10                           # pick randomly from the closest-K tiles
STABILITY_THRESHOLD = 18.0          # LAB threshold for colour corection updates
TILE_COOLDOWN_FRAMES = 0            # prevents rapid flipping

# Basic motion detector parameters 
BG_ALPHA         = 0.02             # background learning rate
MOTION_THRESHOLD = 32               # threshold on |frame - background|

# reproducibility
random.seed(36)
np.random.seed(36)


def list_images(folder):
    """Return sorted list of image files in a folder."""
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    out = []
    for e in exts:
        out.extend(glob.glob(os.path.join(folder, e)))
    return sorted(out)

def load_tiles(path):
    """Load dataset images, resize them, and precompute LAB means."""
    files = list_images(path)
    imgs, labs = [], []

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue

        # standardise format + size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (TILE_W, TILE_H), cv2.INTER_AREA)

        # compute mean LAB value for tile comparison
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        imgs.append(img)
        labs.append(lab.reshape(-1, 3).mean(0).astype(np.float32))

    return np.stack(imgs), np.stack(labs)

def topk_match(target_lab, tile_labs, k):
    """Return indices of the k closest LAB matches for a given pixel."""
    diff = tile_labs - target_lab
    dist = np.sum(diff * diff, axis=1)   # squared Euclidean distance
    k = min(k, len(dist))

    # partial sort to get top-k indices
    idx = np.argpartition(dist, k-1)[:k]
    idx = idx[np.argsort(dist[idx])]
    return idx

# Main Mosaic 

def main():
    tiles, tile_labs = load_tiles(DATASET_DIR)
    N = tiles.shape[0]

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (OUTPUT_W, OUTPUT_H)
    )

    # Initial random tile assignment
    current_idx = np.random.randint(0, N, (GRID_H, GRID_W))

    # Per-tile cooldown counter
    cooldown = np.zeros((GRID_H, GRID_W), np.int32)

    # background model for motion detection 
    background = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # downsample input video 
        small = cv2.resize(rgb, (GRID_W, GRID_H))
        small_lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Motion Detection 
        if background is None:
            # seed background with first frame
            background = small.astype(np.float32)

        # update moving-average background
        cv2.accumulateWeighted(small, background, BG_ALPHA)
        bg_uint8 = background.astype(np.uint8)

        # compute absolute difference from background
        diff = cv2.absdiff(small, bg_uint8)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

        # threshold into a binary motion mask
        _, motion_mask = cv2.threshold(gray, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        motion_mask = (motion_mask > 0)   # convert to boolean grid

        # Build Mosaic Frame 
        mosaic = np.zeros((GRID_H * TILE_H, GRID_W * TILE_W, 3), np.uint8)

        for y in range(GRID_H):
            for x in range(GRID_W):

                # LAB colour distance between video + current tile
                target_lab = small_lab[y, x]
                cur_lab    = tile_labs[current_idx[y, x]]
                delta      = np.linalg.norm(target_lab - cur_lab)

                # Tile update: allowed only if cooldown is over AND
                # either real motion OR significant colour deviation
                if cooldown[y, x] == 0 and (motion_mask[y, x] or delta >= STABILITY_THRESHOLD):

                    tk = topk_match(target_lab, tile_labs, TOPK)
                    # avoid reusing the same tile if possible
                    choices = [i for i in tk if i != current_idx[y, x]]
                    current_idx[y, x] = random.choice(choices) if choices else tk[0]

                    # apply cooldown to prevent tile flicker
                    cooldown[y, x] = TILE_COOLDOWN_FRAMES

                # decrement cooldown after use
                if cooldown[y, x] > 0:
                    cooldown[y, x] -= 1

                # tile into mosaic canvas
                oy, ox = y * TILE_H, x * TILE_W
                mosaic[oy:oy + TILE_H, ox:ox + TILE_W] = tiles[current_idx[y, x]]

        # upscale final mosaic to desired output resolution
        out = cv2.resize(mosaic, (OUTPUT_W, OUTPUT_H), interpolation=OUTPUT_INTERP)
        writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    # shutdown
    cap.release()
    writer.release()
    print("[DONE] Saved:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
