cmake_minimum_required(VERSION 3.0.2)
project(odom)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  robot_localization
)