import cv2
from tracker.tracker import meas_to_bbox
import sys
import os


class Visualizer():
    def __init__(self) -> None:
        self.window = cv2.namedWindow("AgriSORT", cv2.WINDOW_NORMAL)

    def display_image(self, image, mode=0):
        cv2.imshow("AgriSORT", image)
        k = cv2.waitKey(mode)
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit()

    def draw_track(self, track, image):
        bbox = meas_to_bbox(track.get_state())
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), track.color, 4)
        cv2.putText(image, str(track.id), (int(bbox[0] + 15), int(bbox[3] - 20)), cv2.FONT_HERSHEY_COMPLEX, 2, track.color, 4)
        return image
    
    def save_crop(self, track, image, output_dir, frame_num):
        # Check if the input image is valid
        if image is None or image.size == 0:
            print("Error: Input image is empty")
            return
        
        # Get image dimensions
        height, width, _ = image.shape

        # Get the bounding box
        bbox = meas_to_bbox(track.get_state())
        
        # Crop the image using the bounding box
        x1, y1, x2, y2 = map(int, bbox)  # Ensure bounding box coordinates are integers
        cropped_image = image[y1:y2, x1:x2]

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # Verify the crop
        if cropped_image.size == 0:
            print(f"Error: Cropped image is empty for track ID {track.id} and frame {frame_num}")
            return
        
        # Create a folder for the specific tracking ID
        track_dir = os.path.join(output_dir, f"track_{track.id}")
        os.makedirs(track_dir, exist_ok=True)  # Create the folder if it doesn't exist
        
        # Generate a filename for the cropped image with frame number zero-padded to 5 digits
        filename = os.path.join(track_dir, f"{str(frame_num).zfill(5)}.png")
        
        # Save the cropped image
        cv2.imwrite(filename, cropped_image)
        
        # Save the cropped image
        cv2.imwrite(filename, cropped_image)
