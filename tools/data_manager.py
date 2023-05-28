import numpy as np
import cv2
import mimetypes
import os
import sys


class DataLoader():
    def __init__(self, path):
        self.path = path
        self.mode = self.check_path_type(self.path)
        if self.mode == "Folder":
            self.images_names_list = sorted(os.listdir(self.path))
        elif self.mode == "Video":
            self.video_capture = cv2.VideoCapture(self.path)

    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.mode == "Video":
            while True:
                self.index += 1
                ret, frame = self.video_capture.read()
                if not ret:
                    self.video_capture.release()
                    raise StopIteration
                return self.index, frame

        if self.mode == "Folder":
            if self.index >= len(self.images_names_list):
                raise StopIteration
            else:
                # Construct the complete file path with current index
                file_path = os.path.join(self.path, self.images_names_list[self.index])
                # Load the image using cv2
                frame = cv2.imread(file_path)
                self.index += 1
                return self.index, frame

    def check_path_type(self, path):
        if os.path.isdir(path):
            return "Folder"
        elif mimetypes.guess_type(path)[0].startswith('video'):
            return "Video"
        else:
            print("Invalid input type. Input should be eithr a video or a folder containing images.")
            print("Input type", mimetypes.guess_type(path)[0], "is not supported. Refer to documentation.")
            sys.exit()
