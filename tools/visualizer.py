import cv2

class visualizer():
    def __init__(self) -> None:
        self.window = cv2.namedWindow("AgriSORT", cv2.WINDOW_NORMAL)

    def draw_track(self, track):
        cv2.rectangle()
