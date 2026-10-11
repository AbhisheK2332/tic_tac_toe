import cv2

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
found = False

for index in range(6):
    for name, backend in BACKENDS.items():
        cap = cv2.VideoCapture(index, backend)
        frame = None
        if cap.isOpened():
            for _ in range(15):              # skip the first dark frames
                ok, f = cap.read()
                if ok:
                    frame = f
        cap.release()
        if frame is not None:
            h, w = frame.shape[:2]
            out = f"cam{index}_{name}.jpg"
            cv2.imwrite(out, frame)
            print(f"WORKS: CAMERA_SOURCE = {index}, CAMERA_BACKEND = \"{name}\"  ({w}x{h}) -> saved {out}")
            found = True

if not found:
    print("No camera worked. Make sure the phone is connected and no other app is using the camera.")
else:
    print("\nOpen the saved pictures and use the numbers of the one that shows your phone's view.")