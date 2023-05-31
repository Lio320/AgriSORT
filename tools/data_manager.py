import numpy as np
import cv2
import mimetypes
import os
import sys


def create_experiment_folder(path='runs', prefix='track'):
    # If path doesn't exists: create it
    if not os.path.isdir(path):
        os.mkdir(path)

    # Find all existing folders with the prefix
    existing_folders = []
    for folder in os.listdir(path):
        if folder.startswith(prefix):
            existing_folders.append(folder)

    # If folders already exist
    if existing_folders:
        # Sort the folders by name
        existing_folders = sorted(existing_folders, key=lambda x: int(x.split(prefix)[-1]))
        # Get the last number and add 1
        number = int(existing_folders[-1][len(prefix):]) + 1
        folder_path = os.path.join(path, prefix + str(number))
        os.mkdir(folder_path)
    else:
        folder_path = os.path.join(path, prefix + '1')
        os.mkdir(folder_path)
    return folder_path


class DataLoader():
    def __init__(self, path):
        self.path = path
        self.mode = self.check_path_type(self.path)
        if self.mode == "Folder":
            self.images_names_list = sorted(os.listdir(self.path))
            self.len = int(len(self.images_names_list))
        elif self.mode == "Video":
            self.video_capture = cv2.VideoCapture(self.path)
            self.len = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

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
                file_path = os.path.join(self.path,
                                         self.images_names_list[self.index])
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
            print("Invalid input type. Input should be eithr a video or a\
                   folder containing images.")
            print("Input type", mimetypes.guess_type(path)[0], "is not\
                   supported. Refer to documentation.")
            sys.exit()
