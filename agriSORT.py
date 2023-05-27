import cv2
import torch
import rosbag
import argparse
from cv_bridge import CvBridge
from scipy.optimize import linear_sum_assignment
import time
import rospkg
import sys
from tracker.tracker import Tracker, convert_bbox_to_meas, convert_meas_to_bbox


def parse_opt():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Example script to demonstrate command line options.')
    # Add options
    parser.add_argument('-s', '--source', type=str, default='data/Grapes_001/', help='Source of files (dir, file, video, ...)')
    parser.add_argument('-o', '--output', type=str, default='runs/', help='Output file path')
    parser.add_argument('-w', '--weights', type=str, default='./weights/best.pt', help='Weights of YOLOv5 detector')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('-v', '--visualize', type=bool, default=True, help='Enable or disable real-time visualization')


def main():
    # path='./Weights/olives_pears.pt'
    path = './weights/best.pt'
    model = torch.hub.load('/yolov5', 'custom', path=path, source="local")
    model.conf = 0.4
    model.iou = 0.3
    # bag = rosbag.Bag('./rosbags/test1.bag')
    bag = rosbag.Bag('/home/leonardo/Documents/rosbags/Bags_test_set/20220901_142743-001.bag')
    # bag = rosbag.Bag('/home/leonardo/Documents/rosbags/Bags_test_set/20220901_142936-002.bag')
    # bag = rosbag.Bag('/home/leonardo/Documents/rosbags/Bags_test_set/20220808_122420-003.bag')

    # bag = rosbag.Bag('/home/leonardo/Documents/rosbags/Others/aligned_depth.bag')

    # image_topic = "/Aligned_color"
    # image_topic = "/camera/color/image_raw"
    image_topic = "/device_0/sensor_1/Color_0/image/data"
    tracker = Tracker(features="optical_flow")
    image = cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[image_topic])):
        print("FRAME NUMERO", i)
        image = CvBridge().imgmsg_to_cv2(msg, "passthrough")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pred = model(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size=1280)
        start = time.time()
        if i == 0:
            for bbox in pred.xyxy[0]:
                prev_image = gray_image
                # Generate one tracker for each detected bounding box
                tracker.add_track(1, convert_bbox_to_meas(bbox.cpu().detach().numpy()), 0.1, 0.005)
        else:
            motion = tracker.camera_motion_computation(prev_image, gray_image)
            tracker.update_tracks(pred.xyxy[0].cpu().detach().numpy(), motion)

            prev_image = gray_image.copy()
        for track in tracker.tracks:
            if track.display:
                bbox = convert_meas_to_bbox(track.get_state())
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), track.color, 2)
                cv2.putText(image, str(track.id), (int(bbox[0]), int(bbox[3])), cv2.FONT_HERSHEY_COMPLEX, 2, track.color, 2)
        print("TEMPO", time.time()-start)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Image", image)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    opt = parse_opt()
