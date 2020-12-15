from time import time
import numpy as np

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


class RRT:

    def __init__(self, fr, is_in_collision):
        self._fr = fr
        self._is_in_collision = is_in_collision

        '''
        TODO: You can tune these parameters to improve RRT performance.

        However, make sure the values satisfy the following conditions:
            self._constraint_th < 2e-3
            self._q_step_size < 0.1
        '''
        self._project_step_size = 1e-1  # Default:1e-1
        self._constraint_th = 1e-3  # Default: 1e-3

        self._q_step_size = 0.02  # Default: 0.02, 0.045 for map2
        self._target_p = 0.2  # Default: 0.2
        self._max_n_nodes = int(1e5)

    def sample_valid_joints(self):
        '''
        TODO: Implement sampling a random valid configuration.

        The sampled configuration must be within the joint limits, but it does not check for collisions.

        Please use the following in your code:
            self._fr.joint_limis_low - lower joint limits
            self._fr.joint_limis_high - higher joint limits
            self._fr.num_dof - the degree of freedom of franka
        '''
        # Random Sample [1, num_dof] in the configuration space between lower joint limits and higher joint limits
        q = (self._fr.joint_limits_high - self._fr.joint_limits_low) * np.random.random(self._fr.num_dof) + self._fr.joint_limits_low
        return q

    def project_to_constraint(self, q, constraint):
        '''
        TODO: Implement projecting a configuration to satisfy a constraint function using gradient descent.

        Please use the following parameters in your code:
            self._project_step_size - learning rate for gradient descent
            self._constraint_th - a threshold lower than which the constraint is considered to be satisfied

        Input:
            q - the point to be projected
            constraint - a function of q that returns (constraint_value, constraint_gradient)
                         constraint_value is a scalar - it is 0 when the constraint is satisfied
                         constraint_gradient is a vector of length 6 - it is the gradient of the
                                constraint value w.r.t. the end-effector pose (x, y, z, r, p, y)

        Output:
            q_proj - the projected point

        You can obtain the Jacobian by calling self._fr.jacobian(q)
        '''
        q_proj = q.copy()
        err, grad = constraint(q)
        while err > self._constraint_th:
            # print('The error is: ', err)
            J = self._fr.jacobian(q_proj)
            q_proj -= self._project_step_size * J.T.dot(grad)
            # q_proj -= self._project_step_size * J.T.dot(np.linalg.inv(J.dot(J.T))).dot(grad)
            err, grad = constraint(q_proj)
        return q_proj

    def extend(self, tree, q_target, constraint=None):
        '''
        TODO: Implement the constraint extend function.

        Input: 
            tree - a SimpleTree object containing existing nodes
            q_target - configuration of the target state, in shape of [1, num_dof]
            constraint - a constraint function used by project_to_constraint
                         do not perform projection if constraint is None

        Output:
            target_reached - bool, whether or not the target has been reached
            new_node_id - node_id of the new node inserted into the tree by this extend operation
                         Note: tree.insert_new_node returns a node_id
        '''
        target_reached = False
        new_node_id = None
        is_collision = True

        while is_collision:
            if np.random.random(1) < self._target_p:
                # Make sure it will approach to the target
                q_sample = q_target
            else:
                q_sample = self.sample_valid_joints()

            # Find the nearest node (q_near) of the sampling point in current nodes tree
            # Make a step from the nearest node (q_near) to become a new node (q_new) and expand the nodes tree
            nearest_node_id = tree.get_nearest_node(q_sample)[0]
            q_near = tree.get_point(nearest_node_id)
            q_new = q_near + min(self._q_step_size, np.linalg.norm(q_sample - q_near)) * (q_sample - q_near) / np.linalg.norm(q_sample - q_near)

            # Check if the new node has collision with the constraint
            q_new = self.project_to_constraint(q_new, constraint)

            if self._is_in_collision(q_new):
                is_collision = True
                continue
            else:
                is_collision = False

            # Add the q_new as vertex, and the edge between q_new and q_near as edge to the tree
            new_node_id = tree.insert_new_node(q_new, nearest_node_id)

            # if the new state (q_new) is close to the target state, then we reached the target state
            if np.linalg.norm(q_new - q_target) < self._q_step_size:
                target_reached = True

        return target_reached, new_node_id

    def plan(self, q_start, q_target, constraint=None):
        tree = SimpleTree(len(q_start))
        tree.insert_new_node(q_start)

        s = time()
        for n_nodes_sampled in range(self._max_n_nodes):
            if n_nodes_sampled > 0 and n_nodes_sampled % 50 == 0:
                print('RRT: Sampled {} nodes'.format(n_nodes_sampled))

            reached_target, node_id_new = self.extend(tree, q_target, constraint)

            if reached_target:
                break

        print('RRT: Sampled {} nodes in {:.2f}s'.format(n_nodes_sampled, time() - s))

        path = []
        if reached_target:
            backward_path = [q_target]
            node_id = node_id_new
            while node_id is not None:
                backward_path.append(tree.get_point(node_id))
                node_id = tree.get_parent(node_id)
            path = backward_path[::-1]

            print('RRT: Found a path! Path length is {}.'.format(len(path)))
        else:
            print('RRT: Was not able to find a path!')
        
        return path
