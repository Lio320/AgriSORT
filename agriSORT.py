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
import tools.data_manager as Utils


def parse_opt():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Example script to demonstrate command line options.')
    # Add options
    parser.add_argument('-s', '--source', type=str, default='data/Grapes_002.mp4', help='Source of files (dir, file, video, ...)')
    parser.add_argument('-o', '--output', type=str, default='runs/', help='Output file path')
    parser.add_argument('-w', '--weights', type=str, default='./weights/best.pt', help='Weights of YOLOv5 detector')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--features', type=str, default="orb", help='Features for camera motion compensation (ORB, optical flow, ...)')
    parser.add_argument('-v', '--visualize', type=bool, default=True, help='Enable or disable real-time visualization')
    opt = parser.parse_args()
    return opt


def main(opt):
    model = torch.hub.load('./yolov5', 'custom', path=opt.weights, source="local")
    model.conf = opt.conf_thres
    model.iou = opt.iou_thres

    tracker = Tracker(features=opt.features)
    # frame = cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    open("agriSORT.txt", 'w').close()

    dataset = Utils.DataLoader(opt.source)
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
            start = time.time()
            tracker.update_tracks(pred.xyxy[0].cpu().detach().numpy(), Aff)

            prev_image = gray_image.copy()
            for track in tracker.tracks:
                with open("agriSORT.txt", 'a') as f:
                    mot = meas_to_mot(track.x)
                    temp = str(mot[0]) + ', ' + str(mot[1]) + ', ' + str(mot[2]) + ', ' + str(mot[3])
                    f.write(str(i+1) + ', ' + str(track.id) + ', ' + temp + ', -1, -1, -1, -1' + '\n')
                if track.display:
                    bbox = meas_to_bbox(track.get_state())
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), track.color, 2)
                    cv2.putText(frame, str(track.id), (int(bbox[0]), int(bbox[3])), cv2.FONT_HERSHEY_COMPLEX, 2, track.color, 2)
            print("TEMPO", time.time() - start)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Image", frame)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
