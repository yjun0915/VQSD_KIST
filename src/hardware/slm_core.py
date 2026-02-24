import time
import cv2
import screeninfo
import numpy as np
from contextlib import contextmanager


@contextmanager
def slm_session():
    start = time.time()
    slm1 = Fullscreen_CV(1)
    slm2 = Fullscreen_CV(2)
    yield [slm1, slm2]
    end = time.time()
    print("[SLM]", end - start)
    slm1.destroyWindow()
    slm2.destroyWindow()


class Fullscreen_CV():
    delay: int = 200  # internal delay time after imshow
    # screen_id 기본값 = 0
    def __init__(self, screen_id: int = 0):
        # screen_id
        self.monitor = screeninfo.get_monitors()[screen_id]
        self.width = self.monitor.width
        self.height = self.monitor.height
        self.x = self.monitor.x
        self.y = self.monitor.y
        # screen_id
        self.name = str(screen_id)

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name, self.x, self.y)
        cv2.resizeWindow(self.name, self.width, self.height)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_TOPMOST, 1)

        # set initial image
        img_gray = np.full((self.height, self.width), 127, dtype=np.uint8)
        self.imshow(img_gray)
        cv2.waitKey(1900)  # first imshow require long delay time

    @property
    def shape(self):
        return self.height, self.width, 3

    def imshow(self, image: np.ndarray):
        cv2.imshow(self.name, image)
        cv2.waitKey(self.delay)

    def destroyWindow(self):
        cv2.destroyWindow(self.name)
