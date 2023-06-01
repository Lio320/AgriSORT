import os
import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge, CvBridgeError


def parse_opt():
    pass


def main(opt):
    bag_name = "20220907_154427.bag"
    path = "./tools/" + bag_name
    image_topic = "/device_0/sensor_1/Color_0/image/data"
    bag = rosbag.Bag(path)
    save_path = "./tools/20220907_154427/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[image_topic])):
        # if not i % 25:
        print("FRAME NUMERO", i)
        image = CvBridge().imgmsg_to_cv2(msg, "passthrough")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("IMAGE", image)
        # cv2.waitKey(1)
        cv2.imwrite(save_path + str(i).zfill(4) + ".png", image)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
