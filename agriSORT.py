import cv2
import os
import torch
import rosbag
import argparse
from cv_bridge import CvBridge
from scipy.optimize import linear_sum_assignment
import time
import rospkg
import sys
from tracker.tracker import Tracker, bbox_to_meas, meas_to_bbox, meas_to_mot
from tools.visualizer import Visualizer
import tools.data_manager as DataUtils


def parse_opt():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Example script to demonstrate command line options.')
    # Add options
    parser.add_argument('-s', '--source', type=str, default='data/Grapes_002.mp4', help='Source of files (dir, file, video, ...)')
    parser.add_argument('-o', '--output', type=str, default='runs/', help='Output file path')
    parser.add_argument('-w', '--weights', type=str, default='./weights/best.pt', help='Weights of YOLOv5 detector')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--features', type=str, default="optical_flow", help='Features for camera motion compensation (ORB, optical flow, ...)')
    parser.add_argument('-v', '--visualize', type=bool, default=True, help='Enable or disable real-time visualization')
    opt = parser.parse_args()
    return opt


def main(opt):
    model = torch.hub.load('./yolov5', 'custom', path=opt.weights, source="local")
    model.conf = opt.conf_thres
    model.iou = opt.iou_thres

    # Initialize tracker
    tracker = Tracker(features=opt.features)

    # If visualizer, initialize visualizer
    if opt.visualize:
        visualizer = Visualizer()

    folder_path = DataUtils.create_experiment_folder()
    print(folder_path)
    print(folder_path + "/agriSORT.txt")

    open(folder_path + "/agriSORT.txt", 'w').close()

    dataset = DataUtils.DataLoader(opt.source)
    # TODO: Substitute OpenCV with something better to load images (faster)
    for i, frame in dataset:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pred = model(gray_image, size=1280)
        if i == 1:
            for bbox in pred.xyxy[0]:
                prev_image = gray_image
                # Generate one tracker for each detected bounding boxqq
                tracker.add_track(1, bbox_to_meas(bbox.cpu().detach().numpy()), 0.1, 0.001)
        else:
            Aff = tracker.camera_motion_computation(prev_image, gray_image)
            tracker.update_tracks(pred.xyxy[0].cpu().detach().numpy(), Aff)

            prev_image = gray_image.copy()
            for track in tracker.tracks:
                if visualizer:
                    visualizer.draw_track(track)
                with open(folder_path + "/agriSORT.txt", 'a') as f:
                    mot = meas_to_mot(track.x)
                    temp = str(mot[0]) + ', ' + str(mot[1]) + ', ' + str(mot[2]) + ', ' + str(mot[3])
                    f.write(str(i) + ', ' + str(track.id) + ', ' + temp + ', -1, -1, -1, -1' + '\n')

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Image", frame)
            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
