from time import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

from kdtree import KDTree
from franka_robot import FrankaRobot


class SimpleTree:

    def __init__(self, dim):
        self._parents_map = {}
        self._kd = KDTree(dim)

    def insert_new_node(self, point, parent=None):
        node_id = self._kd.insert(point)
        self._parents_map[node_id] = parent

        return node_id
        
    def get_parent(self, child_id):
        return self._parents_map[child_id]

    def get_point(self, node_id):
        return self._kd.get_node(node_id).point

    def get_nearest_node(self, point):
        return self._kd.find_nearest_point(point)

    def construct_path_to_root(self, leaf_node_id):
        path = []
        node_id = leaf_node_id
        while node_id is not None:
            path.append(self.get_point(node_id))
            node_id = self.get_parent(node_id)

        return path


class RRTConnect:

    def __init__(self, fr, is_in_collision):
        self._fr = fr
        self._is_in_collision = is_in_collision

        self._q_step_size = 0.02
        self._connect_dist = 0.8
        self._max_n_nodes = int(1e5)

    def sample_valid_joints(self):
        q = np.random.random(self._fr.num_dof) * (self._fr.joint_limits_high - self._fr.joint_limits_low) + self._fr.joint_limits_low
        return q

    def project_to_constraint(self, q0, constraint):
        def f(q):
            return constraint(q)[0]

        def df(q):
            c_grad = constraint(q)[1]
            q_grad = self._fr.jacobian(q).T @ c_grad
            return q_grad

        def c_f(q):
            diff_q = q - q0
            return diff_q @ diff_q

        def c_df(q):
            diff_q = q - q0
            return 0.5 * diff_q

        c_joint_limits = LinearConstraint(np.eye(len(q0)), self._fr.joint_limits_low, self._fr.joint_limits_high)
        c_close_to_q0 = NonlinearConstraint(c_f, 0, self._q_step_size ** 2, jac=c_df)

        res = minimize(f, q0, jac=df, method='SLSQP', tol=0.1,
                        constraints=(c_joint_limits, c_close_to_q0))

        return res.x

    def _is_seg_valid(self, q0, q1):
        qs = np.linspace(q0, q1, int(np.linalg.norm(q1 - q0) / self._q_step_size))
        for q in qs:
            if self._is_in_collision(q):
                return False
        return True

    def extend(self, tree_0, tree_1, constraint=None):
        '''
        TODO: Implement extend for RRT Connect

        - Only perform self.project_to_constraint if constraint is not None
        - Use self._is_seg_valid, self._q_step_size, self._connect_dist
        '''
        target_reached = False
        node_id_new = None
        is_collision = True

        while is_collision:
            q_sample = self.sample_valid_joints()

            node_id_near = tree_0.get_nearest_node(q_sample)[0]
            q_near = tree_0.get_point(node_id_near)
            q_new = q_near + min(self._q_step_size, np.linalg.norm(q_sample - q_near)) * (q_sample - q_near) / np.linalg.norm(q_sample - q_near)

            q_new = self.project_to_constraint(q_new, constraint)

            if self._is_in_collision(q_new):
                is_collision = True
                continue
            else:
                is_collision = False

            # Add the q_new as vertex, and the edge between q_new and q_near as edge to the tree
            node_id_new = tree_0.insert_new_node(q_new, node_id_near)
            node_id_1 = tree_1.get_nearest_node(q_new)[0]
            q_1 = tree_1.get_point(node_id_1)
            # if the new state is close to the target state, then we reached the target state
            if np.linalg.norm(q_new - q_1) < self._connect_dist and self._is_seg_valid(q_new, q_1):
                target_reached = True

        return target_reached, node_id_new, node_id_1

    def plan(self, q_start, q_target, constraint=None):
        tree_0 = SimpleTree(len(q_start))
        tree_0.insert_new_node(q_start)

        tree_1 = SimpleTree(len(q_target))
        tree_1.insert_new_node(q_target)

        q_start_is_tree_0 = True

        s = time()
        for n_nodes_sampled in range(self._max_n_nodes):
            if n_nodes_sampled > 0 and n_nodes_sampled % 100 == 0:
                print('RRT: Sampled {} nodes'.format(n_nodes_sampled))

            reached_target, node_id_new, node_id_1 = self.extend(tree_0, tree_1, constraint)

            if reached_target:
                break

            q_start_is_tree_0 = not q_start_is_tree_0
            tree_0, tree_1 = tree_1, tree_0

        print('RRT: Sampled {} nodes in {:.2f}s'.format(n_nodes_sampled, time() - s))

        if not q_start_is_tree_0:
            tree_0, tree_1 = tree_1, tree_0

        if reached_target:
            tree_0_backward_path = tree_0.construct_path_to_root(node_id_new)
            tree_1_forward_path = tree_1.construct_path_to_root(node_id_1)

            q0 = tree_0_backward_path[0]
            q1 = tree_1_forward_path[0]
            tree_01_connect_path = np.linspace(q0, q1, int(np.linalg.norm(q1 - q0) / self._q_step_size))[1:].tolist()

            path = tree_0_backward_path[::-1] + tree_01_connect_path + tree_1_forward_path
            print('RRT: Found a path! Path length is {}.'.format(len(path)))
        else:
            path = []
            print('RRT: Was not able to find a path!')
        
        return np.array(path).tolist()
