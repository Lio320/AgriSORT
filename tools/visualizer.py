import cv2
from tracker.tracker import meas_to_bbox
import sys


class Visualizer():
    def __init__(self) -> None:
        self.window = cv2.namedWindow("AgriSORT", cv2.WINDOW_NORMAL)

    def display_image(self, image, mode=0):
        cv2.imshow("AgriSORT", image)
        cv2.waitKey(mode)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit()

    def draw_track(self, track):
        bbox = meas_to_bbox(track.get_state())
        cv2.rectangle(self.window, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), track.color, 2)
        cv2.putText(self.window, str(track.id), (int(bbox[0]), int(bbox[3])), cv2.FONT_HERSHEY_COMPLEX, 2, track.color, 2)
