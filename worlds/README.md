## IGVC Basic Course Gazebo Simulation

### Steps to configure
1. Install the models into the gazebo models folder.
 - `cd ~/espresso-ws/src/Espresso/worlds/models`
 - `./install_models.sh`
2. Build src
 - `cd ~/espresso-ws`
 - `catkin_make`

### Possible Actions
1. Launch only the world in gazebo
 - `gazebo ~/espresso-ws/src/igvc_world/worlds/igvc_plain.world`
 - `gazebo ~/espresso-ws/src/igvc_world/worlds/igvc_ramp.world`
 - `gazebo ~/espresso-ws/src/igvc_world/worlds/igvc_walls.world`
 - `gazebo ~/espresso-ws/src/igvc_world/worlds/igvc_full.world`
2. Launch the world and espresso
 - ex. `roslaunch description simulate.launch world:="full"`
 - world can take on values "plain" (default), "ramp", "walls", "full"
 - plain has neither walls nor ramps
 - ramp has only ramps
 - walls has only walls
 - full has both walls and ramps
