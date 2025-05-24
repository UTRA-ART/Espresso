# Author: Hari Om Chadha
#!/usr/bin/env bash
# ——————————————————————————————————————————————
# Name: launch_rover.sh
# Purpose: Open each ROS command in its own GNOME Terminal window,
#          source your workspace, and apply custom delays.
# Usage:   From anywhere: ./src/Espresso/launch_rover.sh
# ——————————————————————————————————————————————


# 1) Define each entry as "command|delay_in_seconds"
entries=(
  "roslaunch description state_publisher.launch|2"
  "roslaunch nmea_navsat_driver nmea_serial_driver.launch|2"
  "roslaunch phidgets_imu imu.launch|2"
  "roslaunch sensors spatial_imu.lanch|2"
  "roslaunch sensors rplidar_dual.launch|2"
  "roslaunch filter_lidar_data filter_lidar_data.launch|2"
  "roslaunch zed_wrapper zed_no_tf.launch position_tracking:=true|2"
  "roslaunch odom_state odom_state.launch launch_state:=IGVC; roslaunch description utm.launch|2"
  "roslaunch description cartographer.launch launch_state:=IGVC|5"
  "roslaunch nav_stack move_base.launch|3"
  "roslaunch description rviz.launch|3"
)

# 3) Loop over each entry, split into cmd & delay, and launch it

for entry in "${entries[@]}"; do
  cmd="${entry%%|*}"
  delay="${entry##*|}"

  gnome-terminal -- bash -c "
  source devel/setup.bash 
  eval \"$cmd\"
  exec bash
  " &
  sleep "$delay"
done
