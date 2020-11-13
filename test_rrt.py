import numpy as np
import rospy

from franka_robot import FrankaRobot 
from collision_boxes_publisher import CollisionBoxesPublisher
from rrt import RRT


if __name__ == '__main__':
    np.random.seed(0)
    rospy.init_node('rrt')
    fr = FrankaRobot()

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
    ])
    def is_in_collision(joints):
        for box in boxes:
            if fr.check_box_collision(joints, box):
                return True
        return False

    desired_ee_rp = fr.ee(fr.home_joints)[3:5]
    def ee_upright_constraint(q):
        '''
        TODO: Implement constraint function and its gradient. 
        
        This constraint should enforce the end-effector stays upright.
        Hint: Use the roll and pitch angle in desired_ee_rp. The end-effector is upright in its home state.

        Input:
            q - a joint configuration

        Output:
            err - a non-negative scalar that is 0 when the constraint is satisfied
            grad - a vector of length 6, where the ith element is the derivative of err w.r.t. the ith element of ee
        '''
        ee = fr.ee(q)
        err = np.sum((np.asarray(desired_ee_rp) - np.asarray(ee[3:5]))**2)
        grad = np.asarray([0, 0, 0, 2*(ee[3]-desired_ee_rp[0]), 2*(ee[4]-desired_ee_rp[1]), 0])
        return err, grad

    print('Desired ee roll and pitch:', desired_ee_rp)
    '''
    TODO: Fill in start and target joint positions 
    '''
    joints_start = fr.home_joints.copy()
    joints_start[0] = -np.deg2rad(45)
    joints_target = joints_start.copy()
    joints_target[0] = np.deg2rad(45)

    rrt = RRT(fr, is_in_collision)
    constraint = ee_upright_constraint
    plan = rrt.plan(joints_start, joints_target, constraint)
    
    i = 0
    collision_boxes_publisher = CollisionBoxesPublisher('collision_boxes')
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        joints = plan[i % len(plan)]
        fr.publish_joints(joints)
        fr.publish_collision_boxes(joints)
        collision_boxes_publisher.publish_boxes(boxes)

        i += 1
        rate.sleep()
