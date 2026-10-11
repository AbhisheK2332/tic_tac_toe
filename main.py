"""
main.py - joins the EYES (vision.py) and the BRAIN (game.py).

Keep vision.py, game.py, centers.json and this file in the same folder.

Usage:
    python main.py snap empty.jpg     # take ONE camera picture and save it (use it for `pick`)
    python main.py read photo.jpg     # read ONE photo file, print the board and the AI's best move
    python main.py read camera        # same, but takes the picture from the camera
    python main.py play               # play a full game (press Enter = camera takes the photo)

How "play" works:
    1. You make your move on the board.
    2. Press Enter. The camera takes a photo (or type a photo file name instead).
    3. The program reads the board and checks that it makes sense.
    4. If it is the AI's turn, it prints which cell to place the AI's piece in.
    5. You place that piece BY HAND (the robot arm will do this later), then make
       your own move, press Enter again, and so on.
"""

import os
import sys
import cv2
import vision
import game

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
AI = "X"              # the AI plays the black ring.  ("X" = orange piece)
HUMAN = "O"           # the human plays the orange piece
HUMAN_FIRST = True    # True = human makes the first move, False = AI goes first
EASY_CHANCE = 0.0     # 0.0 = perfect AI, 0.3 = easy mode (30% random moves)

USE_CAMERA = True     # True = press Enter to take photos. False = type file names only
CAMERA_SOURCE = 0     # 0 = first camera. Try 1 or 2 for a phone-as-webcam app.
                      # For an IP camera app use a text link, e.g. "http://192.168.1.50:8080/video"
CAMERA_BACKEND = "auto"   # "auto", "dshow" or "msmf" (Windows camera drivers). find_camera.py tells you.
CAMERA_WARMUP = 15    # frames thrown away so the camera can settle its exposure
SAVE_DIR = "captures" # every camera picture is saved here (handy for checking mistakes)
# ----------------------------------------------------------------------

game.EASY_CHANCE = EASY_CHANCE
_shot_number = 0


# ---------------------------------------------------------------- camera
def grab_frame():
    """Take one picture from the camera. Returns the full-size frame."""
    backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}.get(CAMERA_BACKEND)
    if isinstance(CAMERA_SOURCE, int) and backend is not None:
        cap = cv2.VideoCapture(CAMERA_SOURCE, backend)
    else:
        cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera. Try another CAMERA_SOURCE "
                           "(0, 1, 2 or a link) and close other apps using the camera.")
    frame = None
    for _ in range(CAMERA_WARMUP):          # first frames are often dark or blurry
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise RuntimeError("The camera opened but gave no picture.")
    return frame


def resize_frame(frame):
    """Resize like vision.load does, so the saved cell centers still match."""
    h, w = frame.shape[:2]
    scale = vision.IMG_WIDTH / w
    return cv2.resize(frame, (vision.IMG_WIDTH, int(h * scale)))


def take_picture():
    """Capture, save a copy in captures/, and return the resized image."""
    global _shot_number
    frame = grab_frame()
    os.makedirs(SAVE_DIR, exist_ok=True)
    _shot_number += 1
    name = os.path.join(SAVE_DIR, f"shot{_shot_number:03d}.jpg")
    cv2.imwrite(name, frame)
    print(f"(camera picture saved as {name})")
    return resize_frame(frame)


# ---------------------------------------------------------------- reading
def board_from_image(img, centers):
    """Image -> list of 9 cells like ['X','-','O',...]."""
    return [c[0] for c in vision.read_board(img, centers)]


def get_board(text, centers):
    """`text` is what the user typed: empty = camera, otherwise a photo file name."""
    if text == "":
        if not USE_CAMERA:
            raise ValueError("Type a photo file name (USE_CAMERA is False).")
        img = take_picture()
    else:
        img = vision.load(text)          # stops with an error message if the file is missing
    return board_from_image(img, centers)


# ---------------------------------------------------------------- game rules / safety
def whose_turn(board):
    """Return "ai", "human", or None if the piece counts make no sense."""
    ai_n = board.count(AI)
    human_n = board.count(HUMAN)
    if HUMAN_FIRST:
        if human_n == ai_n:
            return "human"
        if human_n == ai_n + 1:
            return "ai"
    else:
        if human_n == ai_n:
            return "ai"
        if ai_n == human_n + 1:
            return "human"
    return None


def check_board(board, prev):
    """
    Safety checks. Return None if the board looks right,
    or a text message explaining the problem.
    `prev` is the board we expect to see before the human's new move
    (None for the very first photo).
    """
    if whose_turn(board) is None:
        return (f"Piece counts look wrong ({board.count(HUMAN)} x {HUMAN}, "
                f"{board.count(AI)} x {AI}).")

    if prev is not None:
        for i in range(9):
            if prev[i] != "-" and board[i] != prev[i]:
                return (f"Cell {i + 1} was {prev[i]} but now reads '{board[i]}'. "
                        "A piece was moved or misread.")
        new_cells = [i for i in range(9) if prev[i] == "-" and board[i] != "-"]
        if len(new_cells) == 0:
            return "No new piece found. Did you make your move?"
        if len(new_cells) > 1:
            return f"{len(new_cells)} new pieces found, but only one move is allowed."
        if board[new_cells[0]] != HUMAN:
            return f"The new piece in cell {new_cells[0] + 1} reads as {board[new_cells[0]]}, not {HUMAN}."
    return None


def announce_if_over(board):
    """Print the result and return True if the game has ended."""
    w = game.winner(board)
    if w == AI:
        print("*** The AI wins! ***")
        return True
    if w == HUMAN:
        print("*** You win! ***")
        return True
    if game.is_full(board):
        print("*** It's a draw. ***")
        return True
    return False


# ---------------------------------------------------------------- commands
def cmd_snap(path):
    """Take one camera picture and save it (use it as the empty-board photo for `pick`)."""
    frame = grab_frame()
    cv2.imwrite(path, frame)
    h, w = frame.shape[:2]
    print(f"Saved {path} ({w}x{h}). Now run:  python vision.py pick {path}")


def cmd_read(arg):
    centers = vision.load_centers()
    board = get_board("" if arg == "camera" else arg, centers)
    print("Board read:")
    game.print_board(board)

    if announce_if_over(board):
        return
    problem = check_board(board, None)
    if problem:
        print("Warning:", problem)
    move = game.best_move(board, AI)
    print(f"Best move for {AI}: cell {move + 1}")


def cmd_play():
    centers = vision.load_centers()
    prev = None   # board we expect to see before the human's next move

    print(f"You are {HUMAN} (orange). The AI is {AI} (black ring).")
    print("Cells are numbered:\n 1 2 3\n 4 5 6\n 7 8 9\n")
    if HUMAN_FIRST:
        print("You go first. Make your move, then take a photo.")
    else:
        print("The AI goes first. Take a photo of the empty board.")

    prompt = ("\nPress Enter to take a photo (or type a file name, q to quit): "
              if USE_CAMERA else "\nPhoto file name (or q to quit): ")

    while True:
        text = input(prompt).strip().strip('"')
        if text.lower() == "q":
            print("Bye.")
            return

        try:
            board = get_board(text, centers)
        except SystemExit:               # vision.load stops when the file can't be read
            print("Could not read that file. Check the name and folder, then try again.")
            continue
        except (RuntimeError, ValueError) as e:
            print("Problem:", e)
            continue

        print("\nBoard read:")
        game.print_board(board)

        problem = check_board(board, prev)
        if problem:
            print("PROBLEM:", problem)
            print("Fix the board and take the photo again.")
            continue

        if announce_if_over(board):
            return

        if whose_turn(board) == "ai":
            move = game.best_move(board, AI)
            board[move] = AI
            print(f">>> AI move: place {AI} in cell {move + 1}")
            game.print_board(board)
            if announce_if_over(board):
                return
            print("Place the piece by hand, make YOUR move, then take a new photo.")
        else:
            print("It's your turn. Make your move, then take a new photo.")

        prev = list(board)   # includes the AI's new piece


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "snap":
        cmd_snap(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "read":
        cmd_read(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "play":
        cmd_play()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
