import cv2
import torch
import argparse
from tracker.tracker import Tracker, bbox_to_meas, meas_to_mot
from tools.visualizer import Visualizer
import tools.data_manager as DataUtils
import time


def parse_opt():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Example script to demonstrate command line options.')
    # Add options
    parser.add_argument('-s', '--source', type=str, default='data/Grapes_001/', help='Source of files (dir, file, video, ...)')
    parser.add_argument('-o', '--output', type=str, default='runs/', help='Output file path')
    parser.add_argument('-w', '--weights', type=str, default='./weights/best.pt', help='Weights of YOLOv5 detector')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--features', type=str, default="optical_flow", help='Features for camera motion compensation (ORB, optical flow, ...)')
    parser.add_argument('--transform', type=str, default="affine", help='Tranformation for estimation of camera motion')
    parser.add_argument('-v', '--visualize', type=bool, default=True, help='Enable or disable real-time visualization')
    opt = parser.parse_args()
    return opt


def main(opt):
    model = torch.hub.load('./yolov5', 'custom', path=opt.weights, source="local")
    model.conf = opt.conf_thres
    model.iou = opt.iou_thres

    # Initialize tracker
    tracker = Tracker(features=opt.features, transform=opt.transform)

    # If visualizer, initialize visualizer
    print(opt.visualize)
    if opt.visualize:
        visualizer = Visualizer()

    # Create folder to save results
    folder_path = DataUtils.create_experiment_folder()
    open(folder_path + "/agriSORT.txt", 'w').close()

    # Load data and start tracking
    dataset = DataUtils.DataLoader(opt.source)
    for i, frame in dataset:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        d_start = time.time()
        pred = model(gray_image, size=1280)
        d_time = (time.time() - d_start) * 1000
        if i == 1:
            for bbox in pred.xyxy[0]:
                prev_image = gray_image
                # Generate one tracker for each detected bounding box
                tracker.add_track(1, bbox_to_meas(bbox.cpu().detach().numpy()), 0.05, 0.00625)
        else:
            c_start = time.time()
            Aff = tracker.camera_motion_computation(prev_image, gray_image)
            c_time = (time.time() - c_start) * 1000
            prev_image = gray_image.copy()
            t_start = time.time()
            tracker.update_tracks(pred.xyxy[0].cpu().detach().numpy(), Aff, frame)
            t_time = (time.time() - t_start) * 1000
            for track in tracker.tracks:
                if opt.visualize and track.display:
                    frame = visualizer.draw_track(track, frame)
                with open(folder_path + "/agriSORT.txt", 'a') as f:
                    mot = meas_to_mot(track.x)
                    temp = str(mot[0]) + ', ' + str(mot[1]) + ', ' + str(mot[2]) + ', ' + str(mot[3])
                    f.write(str(i-1) + ', ' + str(track.id) + ', ' + temp + ', -1, -1, -1, -1' + '\n')
            if opt.visualize:
                visualizer.display_image(frame, 0)
                cv2.imwrite("./GIF/" + str(i).zfill(5) + ".jpg", frame)
            # Terminal output
            print("Frame {}/{} || Detections {} ({:.2f} ms) || Camera Correction ({:.2f} ms) || Tracking {} ({:.2f} ms)".format(
                i, dataset.len, int(len(pred.xyxy[0])), d_time, c_time, len(tracker.tracks), t_time))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
