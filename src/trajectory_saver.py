#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
import sys

file = None
file_name = "odometry_data.txt" 

def odometry_callback(msg):
    global file
    pose = msg.pose.pose
    position = pose.position
    orientation = pose.orientation
    timestamp = msg.header.stamp.to_sec()

    with open(file_name, "a") as file:
        x = position.x
        y = position.y
        z = position.z
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w

        file.write("%f %f %f %f %f %f %f %f\n" % (timestamp, x, y, z, qx, qy, qz, qw))

def main():
    global file
    rospy.init_node("odometry_saver")
    odometry_topic = rospy.get_param("~odometry_topic", "/Odometry")
    file_name = rospy.get_param("~file_name", "odometry_data.txt")

    rospy.Subscriber(odometry_topic, Odometry, odometry_callback)
    rospy.spin()

if __name__ == "__main__":
    main()
