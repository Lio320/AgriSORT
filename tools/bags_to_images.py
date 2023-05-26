import rospy
import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge, CvBridgeError

if __name__ == "__main__":
    bag_name = "Scripts_di_prova/20220808_122420-003.bag"
    path = "./" + bag_name
    path = '/home/leonardo/Documents/rosbags/Bags_test_set/20220901_142743-001.bag'
    image_topic = "/device_0/sensor_1/Color_0/image/data"
    bag = rosbag.Bag(path)
    save_path = "./Scripts_di_prova/Test_sequence_1/"
    for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[image_topic])):
        # if not i % 25:
        print("FRAME NUMERO", i)
        image = CvBridge().imgmsg_to_cv2(msg, "passthrough")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("IMAGE", image)
        # cv2.waitKey(1)
        cv2.imwrite(save_path + str(i).zfill(4) + ".png", image)
