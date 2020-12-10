import argparse
import numpy as np
import rospy

from franka_robot import FrankaRobot
from collision_boxes_publisher import CollisionBoxesPublisher
from rrt import RRT
from rrt_connect import RRTConnect


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=4)
    parser.add_argument('--rrt', '-rrt', type=str2bool, const=True, nargs='?', default=False, help="Use RRT?")
    parser.add_argument('--rrtc', '-rrtc', type=str2bool, const=True, nargs='?', default=False, help="Use RRT-Connect?")
    parser.add_argument('--prm', '-prm', type=str2bool, const=True, nargs='?', default=False, help="Use PRM?")
    parser.add_argument('--reuse_graph', '-reuse_graph', type=str2bool, const=True, nargs='?', default=False, help="Reuse the graph for PRM?")
    args = parser.parse_args()

    np.random.seed(args.seed)
    fr = FrankaRobot()

    rospy.init_node('planner')

    '''
    TODO: Replace obstacle box w/ the box specs in your workspace:
    [x, y, z, r, p, y, sx, sy, sz]
    '''
    boxes = np.array([
        # obstacle
        [0.45, -0.45, 0.7, 0, 0, 0.78, 0.6, 0.6, 0.05],
        # [0.4, 0, 0.25, 0, 0, 0, 0.3, 0.05, 0.5],
        # sides
        [-0.7, 0.7, 0.75, 0, 0, 0.78, 2, 0.01, 1.6],
        [0.7, -0.7, 0.75, 0, 0, 0.78, 2, 0.01, 1.6],
        # back
        [-0.7, -0.7, 0.75, 0, 0, 0.78, 0.01, 2, 1.6],
        # front
        [0.7, 0.7, 0.75, 0, 0, 0.78, 0.01, 2, 1.6],
        # top
        [0, 0, 1.5, 0, 0, 0.78, 2, 2, 0.01],
        # bottom
        [0, 0, -0.05, 0, 0, 0.78, 2, 2, 0.01]
    ])


    def is_in_collision(joints):
        if fr.check_self_collision(joints):
            return True
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
        err = np.sum((np.asarray(desired_ee_rp) - np.asarray(ee[3:5])) ** 2)
        grad = np.asarray([0, 0, 0, 2 * (ee[3] - desired_ee_rp[0]), 2 * (ee[4] - desired_ee_rp[1]), 0])
        return err, grad


    '''
    TODO: Fill in start and target joint positions
    '''
    joints_start = fr.home_joints.copy()
    joints_start[0] = -np.deg2rad(45)
    joints_target = np.array([0, 0, 0, -np.pi / 4, 0, np.pi / 4, np.pi / 4])
    joints_target[0] = -np.deg2rad(45)

    if args.rrt:
        print("RRT: RRT planner is selected!")
        planner = RRT(fr, is_in_collision)
    elif args.rrtc:
        print("RRTC: RRT Connect planner is selected!")
        planner = RRTConnect(fr, is_in_collision)
    elif args.prm:
        print("PRM: PRM planner is selected!")

    constraint = ee_upright_constraint
    if args.prm:
        plan = planner.plan(joints_start, joints_target, constraint, reuse_graph=args.reuse_graph)
    else:
        plan = planner.plan(joints_start, joints_target, constraint)

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
