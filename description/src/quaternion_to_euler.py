#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

def imu_callback(msg):
    # Extract quaternion
    q = msg.orientation
    quaternion = [q.x, q.y, q.z, q.w]
    
    # Convert to Euler angles
    roll, pitch, yaw = euler_from_quaternion(quaternion)

    # Log the results
    rospy.loginfo("Roll: %.3f, Pitch: %.3f, Yaw: %.3f", roll, pitch, yaw)

def quaternion_to_euler_node():
    rospy.init_node('quaternion_to_euler_node', anonymous=True)
    rospy.Subscriber("/imu/data", Imu, imu_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        quaternion_to_euler_node()
    except rospy.ROSInterruptException:
        pass
