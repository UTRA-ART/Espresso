#!/usr/bin/env python3
import struct
import sys 
import time
import json
import os

import cv2
import numpy as np
import onnx
import onnxruntime as ort 
import torch
import pandas as pd

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import rospkg
import rospy

from std_msgs.msg import Header
from cv.msg import FloatArray, FloatList
from geometry_msgs.msg import Point
from cv_utils import camera_projection

from line_fitting import fit_lanes
from ultralytics import YOLO

from threshold_lane.threshold import lane_detection

# import open3d as o3d
from sensor_msgs.msg import CameraInfo, LaserScan, PointCloud2, PointField
from sensor_msgs import point_cloud2


class CVModelInferencer:
    def __init__(self):
        rospy.init_node('lane_detection_model_inference')
        
        self.pub = rospy.Publisher('cv/lane_detections', FloatArray, queue_size=10)
        self.pub_raw = rospy.Publisher('cv/model_output', Image, queue_size=10)
        self.pub_pt = rospy.Publisher('cv/lane_detections_cloud', PointCloud2, queue_size=10)
        # self.pub_scan = rospy.Publisher('cv/lane_detections_scan', LaserScan, queue_size=10)

        self.bridge = CvBridge()
        self.projection = camera_projection.CameraProjection()
        
        rospack = rospkg.RosPack()
        self.model_path = rospack.get_path('lane_detection') + '/models/best.pt'
        self.depth_map_path = rospack.get_path('lane_detection') + '/config/num.npy'


        # Get the parameter to decide between deep learning and classical
        self.classical_mode = rospy.get_param('/lane_detection_inference/lane_detection_mode')
        self.Inference = None
        self.lane_detection = None
    
        if self.classical_mode == 1:
            self.lane_detection = lane_detection
            rospy.loginfo("Lane Detection node initialized with CLASSICAL... ")
        else:
            self.Inference = YOLO(self.model_path)
            # self.Inference = Inference(self.model_path, False)

            rospy.loginfo("Lane Detection node initialized with DEEP LEARNING...\nCUDA status: %s ", torch.cuda.is_available())

        # listen for transform from camera to lidar frames
        # self.listener = tf.TransformListener()
        # self.listener.waitForTransform("/left_camera_link_optical", "/base_laser", rospy.Time(), rospy.Duration(10.0))
        

        
    def run(self):
        rospy.Subscriber("/image", Image, self.process_image)
        rospy.spin()
   
    def lane_transform(self, img):
        length = img.shape[0]
        width = img.shape[1]
        new_width = int(width/8)


        input_pts = np.float32([[int(width/2-new_width),0], 
                                [int(width/2+new_width),0], 
                                [width,length],
                                [0,length] ])
        output_pts = np.float32([[new_width, 0],
                                [width-new_width, 0],
                                [int(width/2)+new_width,length],
                                [int(width/2)-new_width,length]])
        M2 = cv2.getPerspectiveTransform(input_pts,output_pts)
        out = cv2.warpPerspective(img,M2,(width, length),flags=cv2.INTER_LINEAR)
        return out


    def process_image(self, data):
        if data == []:
            return
            
        raw = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        projected_lanes = np.load(self.depth_map_path)

        
        if raw is not None:
            # Get the image
            input_img = raw.copy()
            input_img = cv2.resize(input_img, (330, 180))
            
            # Do model inference 
            output = None
            mask = None

            if self.classical_mode:
                output = self.lane_detection(input_img)

                mask = np.where(output > 0.5, 1., 0.)
                mask = mask.astype(np.uint8)

            else:
                # output = self.Inference.inference(input_img)
                # cv2.rectangle(input_img, (0,0), (input_img.shape[1],int(input_img.shape[0] / 9)), (0,0,0), -1) 
            
                output = self.Inference(input_img)
                confidence_threshold = 0.5

                output_image = np.zeros_like(input_img[:,:,0], dtype=np.uint8)
                # output_image = np.zeros_like(projected_lanes[:,:,0], dtype=np.uint8)

                if output[0].masks:
                    for k in range(len(output[0].masks)):
                        mask = np.array(output[0].masks[k].data.cpu() if torch.cuda.is_available() else output[0].masks[k].data)  # Convert tensor to numpy array
                        label = output[0].names[int(output[0].boxes[k].cls)]

                        if float(output[0].boxes[k].conf) > confidence_threshold:  # Check confidence level
                            if label == 'lane':
                                img = np.where(mask > 0.5, 255, 0).astype(np.uint8)
                                img = cv2.resize(img.squeeze(), (output_image.shape[1], output_image.shape[0]))
                                output_image = np.maximum(output_image, img)

                output = output_image

            # Publish to /cv/model_output
            img_msg = self.bridge.cv2_to_imgmsg(output, encoding='passthrough')
            img_msg.header.stamp = data.header.stamp
            # img_msg.header.stamp = data.header.stamp
            if img_msg is not None:
                self.pub_raw.publish(img_msg)
            
            # Build the message
            lane_msg = FloatList()
            pts_msg = []
            cloud = []

            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    if((output[i][j])==255):
                        pt_msg = Point()
                        pt_msg.x = projected_lanes[i][j][0]
                        pt_msg.y = projected_lanes[i][j][1]
                        pt_msg.z = projected_lanes[i][j][2]

                        pts_msg.append(pt_msg)
                        cloud.append(projected_lanes[i][j])

            lane_msg.elements = pts_msg


            msg_header = Header(frame_id='left_camera_link_optical')
            msg = FloatArray(header=msg_header, lists=[lane_msg])
            msg.header.stamp = data.header.stamp
            self.pub.publish(msg)

            pt_header = Header(frame_id='left_camera_link_optical')
            pt_header.stamp = data.header.stamp
            pt_cloud = point_cloud2.create_cloud_xyz32(header=pt_header, points=cloud)
            self.pub_pt.publish(pt_cloud)
    
                


if __name__ == '__main__':
    wrapper = CVModelInferencer()
    wrapper.run()