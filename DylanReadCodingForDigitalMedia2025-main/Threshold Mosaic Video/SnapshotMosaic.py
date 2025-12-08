import os, glob, random, cv2, numpy as np


INPUT_VIDEO  = "/Users/stonesavage/Desktop/Coding for media/videoplayback.mp4"
OUTPUT_VIDEO = "mosaic_snapshot_progressive7.mp4"
DATASET_DIR  = "/Users/stonesavage/Desktop/Coding for media/Code/images"

GRID_W, GRID_H = 128, 64
TILE_W, TILE_H = 64, 64
OUTPUT_W, OUTPUT_H = 1080, 720

TOPK = 10
COLOR_THRESHOLD = 84       # thresh flip a tile
SNAP_DURATION_SEC = 0.8        # snapshot window length in seconds

random.seed(36)
np.random.seed(36)


def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    files=[]
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

def load_tiles(path):
    imgs, labs = [], []
    for f in list_images(path):
        im = cv2.imread(f)
        if im is None: continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (TILE_W, TILE_H), cv2.INTER_AREA)
        imgs.append(im)
        lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
        labs.append(lab.reshape(-1,3).mean(0).astype(np.float32))
    return np.stack(imgs), np.stack(labs)

def topk_match(target_lab, tile_labs, k):
    diff  = tile_labs - target_lab
    dist  = np.sum(diff * diff, axis=1)
    k     = min(k, len(dist))
    idx   = np.argpartition(dist, k)[:k]
    idx   = idx[np.argsort(dist[idx])]
    return idx


def main():
    tiles, tile_labs = load_tiles(DATASET_DIR)
    N = tiles.shape[0]

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    SNAP_FRAMES = int(fps * SNAP_DURATION_SEC)

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (OUTPUT_W, OUTPUT_H)
    )

    # mosaic index grid
    current_idx = np.zeros((GRID_H, GRID_W), np.int32)

    # tile lock mask for each snapshot
    locked = np.zeros((GRID_H, GRID_W), np.bool_)

    # snapshot LAB reference frame
    snapshot_lab = None
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (GRID_W, GRID_H))
        small_lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)

        # SNAPSHOT RESET 
        if frame_count % SNAP_FRAMES == 0:
            # take snapshot LAB
            snapshot_lab = small_lab.copy()

            # reset locks
            locked[:] = False

            # build starting mosaic from snapshot
            for y in range(GRID_H):
                for x in range(GRID_W):
                    target = snapshot_lab[y, x]
                    tk = topk_match(target, tile_labs, TOPK)
                    current_idx[y, x] = random.choice(tk)

        # PER-FRAME UPDATES 
        for y in range(GRID_H):
            for x in range(GRID_W):

                if locked[y,x]:
                    continue   # tile already changed in this window

                # compare new video frame to snapshot frame
                target = small_lab[y,x]
                snap   = snapshot_lab[y,x]

                delta = np.linalg.norm(target - snap)

                if delta >= COLOR_THRESHOLD:
                    tk = topk_match(target, tile_labs, TOPK)
                    choices = [i for i in tk if i != current_idx[y,x]]
                    current_idx[y,x] = random.choice(choices) if choices else tk[0]
                    locked[y,x] = True   # cannot change again until next snapshot

        # DRAW FRAME 
        mosaic = np.zeros((GRID_H*TILE_H, GRID_W*TILE_W, 3), np.uint8)

        for y in range(GRID_H):
            for x in range(GRID_W):
                oy, ox = y*TILE_H, x*TILE_W
                mosaic[oy:oy+TILE_H, ox:ox+TILE_W] = tiles[current_idx[y,x]]

        out = cv2.resize(mosaic, (OUTPUT_W, OUTPUT_H), cv2.INTER_NEAREST)
        writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

        frame_count += 1

    cap.release()
    writer.release()
    print("[DONE] Saved:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
