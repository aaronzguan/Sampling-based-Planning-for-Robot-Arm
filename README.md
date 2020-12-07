# Sampling-based Planning for Robot Arm

This is the final project of the CMU-16782. It has the implementation of basic forward/inverse kinematics, and the sampling_based path planning with constraints for the Franka Robot Arm.

For Ubuntu 16.04:

1. Install ROS Kinetic on your computer following instructions here: [http://wiki.ros.org/ kinetic/Installation/Ubuntu.](http://wiki.ros.org/ kinetic/Installation/Ubuntu. ) (Make sure to install the Desktop-Full Install version of ROS Kinetic.)

2. Install libfranka and franka_ros using the following command:

   `sudo apt install ros-kinetic-libfranka ros-kinetic-franka-ros`

3. Install all requirements.

   `pip install -r requirements.txt`

4. Change the path argument in visualize.launch to the path to your working folder.

5. Run the RRT planner:

   `python3 plan.py --rrt`

   Run the CBIRRT planner:

   `python3 plan.py --rrtc`

   Run the PRM planner:

   `python3 plan.py --prm`
