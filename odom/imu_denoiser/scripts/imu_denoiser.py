#!/usr/bin/env python3
"""
imu_denoiser.py

A ROS node that denoises raw IMU data using a second-order Butterworth low-pass filter.
It subscribes to an IMU topic (e.g., /imu/data), applies the filter to each axis of the 
linear acceleration and angular velocity measurements, and publishes the denoised data to 
another topic (e.g., /imu/data_denoised).

To run node -> python3 imu_denoiser.py
"""

import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from scipy.signal import butter, lfilter, lfilter_zi

class ButterworthIMUDenoiser:
    def __init__(self, order, cutoff, fs):
        """
        Initialize the Butterworth filter for IMU data.

        Args:
            order (int): The order of the Butterworth filter.
            cutoff (float): The cutoff frequency in Hz.
            fs (float): The sampling frequency in Hz.
        """
        # Normalize cutoff frequency (Nyquist frequency is fs/2)
        Wn = cutoff / (fs / 2.0)
        self.b, self.a = butter(order, Wn, btype='low', analog=False)

        # Initialize filter states for angular velocity and linear acceleration.
        # lfilter_zi returns the steady-state initial condition.
        zi = lfilter_zi(self.b, self.a)
        self.state_ang = {'x': zi.copy(), 'y': zi.copy(), 'z': zi.copy()}
        self.state_lin = {'x': zi.copy(), 'y': zi.copy(), 'z': zi.copy()}

    def update_filter(self, sample, state):
        """
        Apply the Butterworth filter to a new sample and update the filter state.

        Args:
            sample (float): The new measurement sample.
            state (ndarray): The current filter state.

        Returns:
            filtered (float): The filtered output.
            new_state (ndarray): The updated filter state.
        """
        y, zf = lfilter(self.b, self.a, [sample], zi=state)
        return y[0], zf

    def process_imu(self, imu_msg):
        """
        Process an incoming IMU message:
          - Filters the angular velocity and linear acceleration using the Butterworth filter.
          - Copies orientation and covariance fields unchanged.

        Returns:
          A new Imu message with filtered sensor data.
        """
        filtered_msg = Imu()
        filtered_msg.header = imu_msg.header
        filtered_msg.orientation = imu_msg.orientation
        filtered_msg.orientation_covariance = imu_msg.orientation_covariance

        # Filter angular velocity for each axis.
        ang = imu_msg.angular_velocity
        f_ang = {}
        for axis in ['x', 'y', 'z']:
            current_val = getattr(ang, axis)
            filt_val, new_state = self.update_filter(current_val, self.state_ang[axis])
            self.state_ang[axis] = new_state
            f_ang[axis] = filt_val

        # Filter linear acceleration for each axis.
        lin = imu_msg.linear_acceleration
        f_lin = {}
        for axis in ['x', 'y', 'z']:
            current_val = getattr(lin, axis)
            filt_val, new_state = self.update_filter(current_val, self.state_lin[axis])
            self.state_lin[axis] = new_state
            if axis == 'x':
                filt_val+=-0.33 
                filt_val = -filt_val
            if axis == 'z':
                filt_val=0
            if axis == 'y':
                filt_val+=-0.39
            if abs(filt_val)<0.1:
                filt_val = 0
            f_lin[axis] = filt_val

        filtered_msg.angular_velocity = Vector3(**f_ang)
        filtered_msg.linear_acceleration = Vector3(**f_lin)
        filtered_msg.angular_velocity_covariance = imu_msg.angular_velocity_covariance
        filtered_msg.linear_acceleration_covariance = imu_msg.linear_acceleration_covariance

        return filtered_msg

class ImuDenoiser:
    def __init__(self):
        # Initialize the ROS node.
        rospy.init_node("imu_denoiser")

        # Retrieve parameters (overridable via the parameter server)
        # Updated default sampling frequency to 125 Hz and cutoff frequency to 10 Hz.
        self.fs = rospy.get_param('~sampling_frequency', 125.0)   # IMU sampling frequency in Hz
        self.cutoff = rospy.get_param('~cutoff_frequency', 10.0)    # Cutoff frequency in Hz for denoising
        self.order = rospy.get_param('~filter_order', 2)            # Butterworth filter order
        self.input_topic = rospy.get_param('~input_topic', '/imu/data')
        self.output_topic = rospy.get_param('~output_topic', '/imu/data_denoised')

        # Create the Butterworth filter denoiser instance.
        self.denoiser = ButterworthIMUDenoiser(self.order, self.cutoff, self.fs)

        # Set up publisher and subscriber.
        self.publisher = rospy.Publisher(self.output_topic, Imu, queue_size=10)
        rospy.Subscriber(self.input_topic, Imu, self.callback)

        rospy.loginfo("Butterworth IMU Denoiser node running with order=%d, cutoff=%.2f Hz, fs=%.2f Hz",
                      self.order, self.cutoff, self.fs)

    def callback(self, msg):
        """
        Callback for processing incoming IMU messages:
          - Filters the message and publishes the result.
        """
        filtered_msg = self.denoiser.process_imu(msg)
        self.publisher.publish(filtered_msg)

if __name__ == '__main__':
    node = ImuDenoiser()
    rospy.spin()
