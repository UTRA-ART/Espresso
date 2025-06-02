#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

def odom_callback(msg):
    # Extract quaternion from Odometry pose
    q = msg.pose.pose.orientation
    quaternion = [q.x, q.y, q.z, q.w]
    
    # Convert to Euler angles
    roll, pitch, yaw = euler_from_quaternion(quaternion)
    yaw = yaw*180/3.1415

    # Log the results
    rospy.loginfo("Roll: %.3f, Pitch: %.3f, Yaw: %.3f", roll, pitch, yaw)

def quaternion_to_euler_node():
    rospy.init_node('quaternion_to_euler_node', anonymous=True)
    rospy.Subscriber("/wheel_odom/quat", Odometry, odom_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        quaternion_to_euler_node()
    except rospy.ROSInterruptException:
        pass

