# utils_webcam.py
import cv2, threading, time


class Webcam:
    def __init__(self, cam=0, width=1280, height=720, backend=None):
        if backend is None:
            self.cap = cv2.VideoCapture(cam)
        else:
            self.cap = cv2.VideoCapture(cam, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.lock = threading.Lock()
        self.ok, self.frame = self.cap.read()
        self.alive = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        # Continuously grab the newest frame
        while self.alive:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.ok, self.frame = ok, f
            time.sleep(0.001)  # tiny yield to be nice to CPU

    def read(self):
        with self.lock:
            return self.ok, (None if self.frame is None else self.frame.copy())

    def release(self):
        self.alive = False
        self.t.join(timeout=0.2)
        self.cap.release()
