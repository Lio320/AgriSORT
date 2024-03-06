import os
import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge, CvBridgeError


def parse_opt():
    pass


def main(opt):
    bag_name = "2023-07-28-12-45-37.bag"
    path = "./tools/" + bag_name
    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    bag = rosbag.Bag(path)
    save_path = "./tools/2023-07-28-12-45-37_depth/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[depth_topic])):
        print("FRAME NUMERO", i)
        image = CvBridge().imgmsg_to_cv2(msg, "passthrough")
        image = image.astype(np.uint16)
        print(image.shape)
        print("MAX VALUE PRIMA", np.max(image))
        print("MAX VALUE", np.max(image))
        cv2.imwrite(save_path + str(i).zfill(4) + "_d.tif", image.astype(np.uint16))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
