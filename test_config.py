import argparse
import numpy as np
import rospy

from franka_robot import FrankaRobot
from collision_boxes_publisher import CollisionBoxesPublisher
from rrt_connect import RRTConnect


if __name__ == '__main__':
    fr = FrankaRobot()

    rospy.init_node('rrt')

    '''
    TODO: Replace obstacle box w/ the box specs in your workspace:
    [x, y, z, r, p, y, sx, sy, sz]
    '''
    boxes = np.array([

        # obstacle
        [0.4, 0, 0.25, 0, 0, 0, 0.3, 0.05, 0.5],
        # sides
        [0.15, 0.46, 0.5, 0, 0, 0, 1.2, 0.01, 1.1],
        [0.15, -0.46, 0.5, 0, 0, 0, 1.2, 0.01, 1.1],
        # back
        [-0.41, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1],
        # front
        [0.75, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1],
        # top
        [0.2, 0, 1, 0, 0, 0, 1.2, 1, 0.01],
        # bottom
        [0.2, 0, -0.05, 0, 0, 0, 1.2, 1, 0.01]
        # [0.45, -0.45, 0.7, 0, 0, 0.78, 0.6, 0.6, 0.05],
        # # obstacle
        # # # [0, 0, 0, 0, 0, 0, 0, 0, 0],
        # # [0.4, 0, 0.25, 0, 0, 0, 0.3, 0.05, 0.5],
        # # sides
        # [-0.7, 0.7, 0.75, 0, 0, 0.78, 2, 0.01, 1.6],
        # [0.7, -0.7, 0.75, 0, 0, 0.78, 2, 0.01, 1.6],
        # # back
        # [-0.7, -0.7, 0.75, 0, 0, 0.78, 0.01, 2, 1.6],
        # # front
        # [0.7, 0.7, 0.75, 0, 0, 0.78, 0.01, 2, 1.6],
        # # top
        # [0, 0, 1.5, 0, 0, 0.78, 2, 2, 0.01],
        # # bottom
        # [0, 0, -0.05, 0, 0, 0.78, 2, 2, 0.01]
    ])
    def is_in_collision(joints):
        for box in boxes:
            if fr.check_box_collision(joints, box):
                return True
        return False

    desired_ee_rp = fr.ee(fr.home_joints)[3:5]

    '''
    TODO: Fill in start and target joint positions
    '''
    # joints_start = None
    # joints_target = None
    joints_start = fr.home_joints.copy()
    #joints_start = np.array([0, 0, 0, -np.pi / 4, 0, np.pi / 4, np.pi / 4])
    joints_start[0] = np.deg2rad(45)
    joints_target = joints_start.copy()
    joints_target[0] = np.deg2rad(45)

    plan = [joints_start]
    collision_boxes_publisher = CollisionBoxesPublisher('collision_boxes')
    rate = rospy.Rate(10)
    i = 0
    while not rospy.is_shutdown():
        rate.sleep()
        joints = plan[i % len(plan)]
        fr.publish_joints(joints)
        fr.publish_collision_boxes(joints)
        collision_boxes_publisher.publish_boxes(boxes)

        i += 1
