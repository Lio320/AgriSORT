import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from tracker.kalmanFilter import KalmanFilter


def meas_to_mot(meas):
    width = meas[2]
    height = meas[3]
    x_left = meas[0] - (width/2)
    y_top = meas[1] - (height/2)
    return (x_left, y_top, width, height)


def bbox_to_meas(det):
    height = int(det[3] - det[1])
    width = int(det[2] - det[0])
    x_c = int(det[0] + (width/2))
    y_c = int(det[1] + (height/2))
    return (x_c, y_c, width, height)


def meas_to_bbox(meas):
    x1 = int(meas[0] - (meas[2]/2))
    y1 = int(meas[1] - (meas[3]/2))
    x2 = int(meas[0] + (meas[2]/2))
    y2 = int(meas[1] + (meas[3]/2))
    return (x1, y1, x2, y2)


def compute_iou(tracks, measurements):
    iou_matrix = np.zeros((len(measurements), len(tracks)))
    for i, track in enumerate(tracks):
        track_bbox = meas_to_bbox(track.get_state())
        for j, meas in enumerate(measurements):
            xx1 = np.maximum(track_bbox[0], meas[0])
            yy1 = np.maximum(track_bbox[1], meas[1])
            xx2 = np.minimum(track_bbox[2], meas[2])
            yy2 = np.minimum(track_bbox[3], meas[3])
            if xx2 < xx1 or yy2 < yy1:
                intersection_area = 0
            else:
                intersection_area = (xx2-xx1) * (yy2-yy1)
            bb1_area = (track_bbox[2]-track_bbox[0]) * (track_bbox[3]-track_bbox[1])
            bb2_area = (meas[2]-meas[0]) * (meas[3]-meas[1])
            iou_matrix[j, i] = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou_matrix


class Tracker():
    def __init__(self, features="optical_flow", transform='affine'):
        self.tracks = []
        self.prev_id = 0
        self.color_id = 0
        self.features = features
        self.transform = transform
        if self.features == "orb":
            # Initialize the ORB detector algorithm
            self.orb = cv2.ORB_create()
            # Initialize the Brute-Force matcher for matching the keypoints
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if self.features == "optical_flow":
            self.feature_params = dict(maxCorners=200, qualityLevel=0.6, minDistance=7, blockSize=7)
            # Parameters for Lucas Kanade optical flow
            self.lk_params = dict(winSize=(21, 21),
                                  maxLevel=3,
                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def add_track(self, dt, state, state_std, meas_std):
        # The color is sequential
        color, self.color_id = self.get_colors(self.color_id)
        self.tracks.append(KalmanFilter(dt, state, state_std, meas_std, color, self.prev_id, self.transform))
        self.prev_id += 1

    def remove_track(self, track):
        self.tracks.remove(track)

    def get_colors(self, id):
        color_pans = [(204, 78, 210),
                      (0, 192, 255),
                      (0, 131, 0),
                      (240, 176, 0),
                      (254, 100, 38),
                      (0, 0, 255),
                      (182, 117, 46),
                      (185, 60, 129),
                      (204, 153, 255),
                      (80, 208, 146),
                      (0, 0, 204),
                      (17, 90, 197),
                      (0, 255, 255),
                      (102, 255, 102),
                      (255, 255, 0)]
        color = color_pans[id]
        id += 1
        if id == 14:
            id = 0
        return color, id

    def camera_motion_computation(self, prev_img, curr_img):
        if self.features == "orb":
            # Compute keypoints and descriptors
            prevKeypoints, prevDescriptors = self.orb.detectAndCompute(prev_img, None)
            currKeypoints, currDescriptors = self.orb.detectAndCompute(curr_img, None)
            matches = self.matcher.match(prevDescriptors, currDescriptors)
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            matches = matches[:50]
            prev_pts = np.float32([prevKeypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            curr_pts = np.float32([currKeypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        if self.features == "optical_flow":
            # Define the features to track using the Shi-Tomasi corner detector
            prev_pts = cv2.goodFeaturesToTrack(prev_img, maxCorners=200, qualityLevel=0.01, minDistance=10)
            # Compute the optical flow using the Lucas-Kanade method
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None, **self.lk_params)
            # Select only the points that have a good optical flow estimation
            prev_pts = prev_pts[status == 1]
            curr_pts = curr_pts[status == 1]
        if self.transform == 'affine':
            # Estimate the affine transformation matrix
            A, _ = cv2.estimateAffine2D(prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            return A
        elif self.transform == 'homography':
            # Estimate the homography
            H, _ = cv2.findHomography(prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            return H

    def ransac(kp1, kp2, good, mp1, mp2, MIN_MATCH_COUNT=2, inlier_threshold=10.0):
        pass

    def associate_tracks(self, measurements):
        associated_tracks = []
        associated_measurements = []
        non_associated_tracks = []
        non_associated_measurements = []

        temp_matrix = compute_iou(self.tracks, measurements)

        # TO DISPLAY MEASUREMENTS ######
        # for i, bbox in enumerate(measurements):
        #     cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 4)
        #     cv2.putText(image, str(i), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 2)

        # Gating phase
        temp_matrix[temp_matrix < 0.3] = 0
        iou_matrix = []

        # Remove all measurements that cannot be associated with any other
        for i, row in enumerate(temp_matrix):
            if not np.all(row == 0):
                # Best friends, verify if max is max of both row and column
                # max = np.max(row)
                # max_ind = np.argmax(row)
                # max_column = np.max(temp_matrix.T[max_ind])
                # if max != max_column:
                #     non_associated_measurements.append(i)
                # else:
                iou_matrix.append(row)
            else:
                non_associated_measurements.append(i)

        iou_matrix = np.array(iou_matrix)

        # TO DISPLAY HEURISTIC CHANGE MATRIX ######
        # print("La matrice precedente era\n", temp_matrix)
        # print("La matrice è\n", iou_matrix)

        # HUNGARIAN ALGORITHM
        if len(iou_matrix) > 0:
            associated_measurements, associated_tracks = linear_sum_assignment(-iou_matrix)

        for i, ass in enumerate(associated_measurements):
            for non_ass in non_associated_measurements:
                if non_ass <= ass:
                    associated_measurements[i] += 1

        for i, track in enumerate(self.tracks):
            if track.id not in associated_tracks:
                non_associated_tracks.append(i)

        # print("Associated tracks: {}\nAssociated measurements: {}\nNon associated tracks: {}\nNon associated measurements: {}"\
        #       .format(associated_tracks, associated_measurements, non_associated_tracks, non_associated_measurements))
        return associated_tracks, associated_measurements, non_associated_tracks, non_associated_measurements

    def update_tracks(self, detections, motion):
        # PREDICTION PHASE
        for track in self.tracks:
            track.predict(motion)
            # TO DISPLAY PREDICTION PHASE
            # if track.display:
            # bbox = convert_meas_to_bbox(track.get_state())
            # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            # cv2.putText(image, str(track.id), (int(bbox[0]), int(bbox[3])), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

        # ASSOCIATION PHASE ######
        # Associate predicted tracks with measurements, update associated ones, generates new ones and only predict not associated ones for x frames
        # Then delete the non associated if not associated for x frames. Use IOU to associate bboxes
        ass_tracks, ass_meas, non_ass_tracks, non_ass_meas = self.associate_tracks(detections)

        # UPDATE PHASE ######
        for i, j in zip(ass_tracks, ass_meas):
            # print("Track {} associated to measurement {}".format(self.tracks[i].id, j))
            self.tracks[i].update(bbox_to_meas(detections[j]))
            self.tracks[i].last_seen = 0
            self.tracks[i].display = True

        for i in non_ass_meas:
            self.add_track(1, bbox_to_meas(detections[i]), 0.1, 0.001)

        for i in non_ass_tracks:
            self.tracks[i].last_seen += 1

        for i in ass_tracks:
            if self.tracks[i].temp:
                self.tracks[i].age += 1
            if self.tracks[i].age > 2:
                self.tracks[i].temp = False

        for track in reversed(self.tracks):
            if track.temp:
                track.display = False
            if track.last_seen > 4:
                track.display = False
                # print("Hidden track", track.id)
            if track.last_seen == 10:
                self.remove_track(track)
                # print("Removed track", track.id)
