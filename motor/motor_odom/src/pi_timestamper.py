#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry

def callback(msg):
    msg.header.stamp = rospy.Time.now()
    pub.publish(msg)

rospy.init_node('pi_timestamper')
sub = rospy.Subscriber('/wheel_odom/euler', Odometry, callback)
pub = rospy.Publisher('/wheel_odom/euler_synced', Odometry, queue_size=10)
rospy.spin()
