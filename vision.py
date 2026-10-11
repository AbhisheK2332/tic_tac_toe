"""
Tic-tac-toe vision module (photos only, no live camera needed).
X = ORANGE pieces, O = BLACK pieces, board = tan / light.

Usage:
    python vision.py pick  empty.jpg          # click the 9 cell centers (run once)
    python vision.py check photo.jpg          # read one photo, show H/S/V per cell
    python vision.py test  tests/             # run every photo in a folder, pass/fail

Test file names must START with the expected board, 9 characters, row by row:
    X = orange piece, O = black piece, - = empty
then an underscore and anything you like. Example:
    XO-_-X-_O--_evening_lamp.jpg   -> but written WITHOUT spaces/extra underscores
    i.e.  XO--X-O--_evening.jpg
        X O .
        . X .
        O . .
"""

import sys
import os
import json
import glob
import cv2
import numpy as np

# ----------------------------------------------------------------------
# SETTINGS - tune these in Step 4 using the H/S/V numbers from `check`
# ----------------------------------------------------------------------
IMG_WIDTH = 800          # every image is resized to this width (keeps aspect ratio)
HALF = 25                # half-size of the square patch around each cell center
CENTERS_FILE = "centers.json"

# Black piece (O): it is a ring (washer) with a hole, so a tiny center square would
# only see the board through the hole. Instead we look at a bigger circle around
# the cell center and measure what FRACTION of its pixels are dark.
# A ring covers a big part of the circle; an empty cell has almost none.
# This keeps working even if your click on the center was a few pixels off.
SAMPLE_R = 85            # radius of the circle (pixels in the resized image)
DARK_V = 120             # a pixel with brightness below this counts as "dark"
DARK_FRAC_MIN = 0.25     # if at least this fraction is dark -> black piece -> O

# Orange piece (X): strongly colorful. The tan board has similar hue but much
# lower saturation (your board: S about 85-110, orange piece: S about 200).
ORANGE_SAT_MIN = 150     # saturation must be at least this -> X
ORANGE_H_MIN = 5         # orange hue range (OpenCV hue is 0-179)
ORANGE_H_MAX = 25
# ----------------------------------------------------------------------


def load(path):
    """Read an image and resize it to a fixed width so coordinates always match."""
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read image: {path}")
    h, w = img.shape[:2]
    scale = IMG_WIDTH / w
    return cv2.resize(img, (IMG_WIDTH, int(h * scale)))


def load_centers():
    if not os.path.exists(CENTERS_FILE):
        sys.exit(f"{CENTERS_FILE} not found. Run:  python vision.py pick empty.jpg")
    with open(CENTERS_FILE) as f:
        return [tuple(p) for p in json.load(f)]


def read_cell(img, cx, cy):
    """Return (label, h, s, v) for the patch around (cx, cy)."""
    patch = img[cy - HALF:cy + HALF, cx - HALF:cx + HALF]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = float(np.median(hsv[:, :, 0]))
    s = float(np.median(hsv[:, :, 1]))
    v = float(np.median(hsv[:, :, 2]))

    # Fraction of dark pixels in a circle around the center (for the black ring piece)
    r = SAMPLE_R
    y0, x0 = max(cy - r, 0), max(cx - r, 0)
    region = img[y0:cy + r, x0:cx + r]
    yy, xx = np.ogrid[y0:y0 + region.shape[0], x0:x0 + region.shape[1]]
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    vals = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)[:, :, 2][mask]
    dark = float(np.mean(vals < DARK_V))

    label = "-"
    if s >= ORANGE_SAT_MIN and ORANGE_H_MIN <= h <= ORANGE_H_MAX:
        label = "X"                                   # orange piece
    elif dark >= DARK_FRAC_MIN:
        label = "O"                                   # black piece
    return label, h, s, v, dark


def read_board(img, centers):
    return [read_cell(img, cx, cy) for cx, cy in centers]


def print_grid(cells):
    for r in range(3):
        print(" ".join(c[0] if c[0] != "-" else "." for c in cells[r * 3:r * 3 + 3]))


# ----------------------------------------------------------------------
# pick: click the 9 centers
# ----------------------------------------------------------------------
def cmd_pick(path):
    img = load(path)
    clicks = []
    win = "Click 9 cell centers (row by row). u=undo  ESC=quit"

    def redraw():
        view = img.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.rectangle(view, (x - HALF, y - HALF), (x + HALF, y + HALF), (0, 200, 0), 2)
            cv2.putText(view, str(i + 1), (x - 8, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.imshow(win, view)

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 9:
            clicks.append((x, y))
            redraw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            print("Cancelled, nothing saved.")
            break
        if key == ord("u") and clicks:
            clicks.pop()
            redraw()
        if len(clicks) == 9:
            with open(CENTERS_FILE, "w") as f:
                json.dump(clicks, f)
            print(f"Saved 9 centers to {CENTERS_FILE}:")
            for c in clicks:
                print("  ", c)
            print("Check that the 9 squares sit well inside the cells, then press any key.")
            cv2.waitKey(0)
            break
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# check: read one photo and show details
# ----------------------------------------------------------------------
def cmd_check(path):
    img = load(path)
    centers = load_centers()
    cells = read_board(img, centers)

    print("\nBoard read:")
    print_grid(cells)
    print("\nCell   label    H     S     V   dark%")
    for i, (label, h, s, v, rv) in enumerate(cells):
        print(f"  {i + 1}      {label}    {h:5.1f} {s:5.1f} {v:5.1f} {rv * 100:5.0f}%")

    view = img.copy()
    colors = {"X": (0, 0, 255), "O": (255, 0, 0), "-": (0, 200, 0)}
    for (cx, cy), (label, h, s, v, rv) in zip(centers, cells):
        cv2.rectangle(view, (cx - HALF, cy - HALF), (cx + HALF, cy + HALF), colors[label], 2)
        cv2.circle(view, (cx, cy), SAMPLE_R, (255, 255, 0), 1)     # black-piece sampling circle
        cv2.putText(view, label, (cx - 10, cy - HALF - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[label], 2)
        cv2.putText(view, f"H{h:.0f} S{s:.0f} V{v:.0f} D{rv * 100:.0f}%", (cx - HALF, cy + HALF + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    out = "check_result.png"
    cv2.imwrite(out, view)
    print(f"\nSaved overlay to {out}. Showing window - press any key to close.")
    cv2.imshow("check", view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# test: run a whole folder against expected answers in the filenames
# ----------------------------------------------------------------------
def expected_from_name(path):
    name = os.path.basename(path)
    exp = name[:9].upper()
    if len(exp) == 9 and all(ch in "XO-" for ch in exp):
        return exp
    return None


def cmd_test(folder):
    centers = load_centers()
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        files += glob.glob(os.path.join(folder, ext))
    files = sorted(set(files))
    if not files:
        sys.exit(f"No images found in {folder}")

    passed = failed = skipped = 0
    for path in files:
        exp = expected_from_name(path)
        if exp is None:
            print(f"SKIP  {os.path.basename(path)}  (name must start with 9 chars of X/O/-)")
            skipped += 1
            continue

        cells = read_board(load(path), centers)
        got = "".join(c[0] for c in cells)

        if got == exp:
            print(f"PASS  {os.path.basename(path)}")
            passed += 1
        else:
            failed += 1
            print(f"FAIL  {os.path.basename(path)}")
            print(f"      expected {exp}   got {got}")
            for i, (e, c) in enumerate(zip(exp, cells)):
                if e != c[0]:
                    print(f"      cell {i + 1}: expected {e}, got {c[0]}   "
                          f"(H={c[1]:.0f} S={c[2]:.0f} V={c[3]:.0f} dark%={c[4] * 100:.0f}%)")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped, out of {len(files)}")
    if failed == 0 and skipped == 0:
        print("All correct. Milestone reached for this test set!")


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("pick", "check", "test"):
        print(__doc__)
        sys.exit(1)
    {"pick": cmd_pick, "check": cmd_check, "test": cmd_test}[sys.argv[1]](sys.argv[2])


if __name__ == "__main__":
    main()