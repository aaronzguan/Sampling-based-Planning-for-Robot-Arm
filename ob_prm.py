from time import time
import numpy as np
from kdtree import KDTree
import collections
import heapq
import pickle
import itertools
import math
import random


class SimpleGraph:

    def __init__(self, dim, capacity=100000):
        self._edges = collections.defaultdict(list)
        self._kd = KDTree(dim, capacity)
        self.start_id = None
        self.target_id = None

    def __len__(self):
        return len(self._kd)

    def insert_new_node(self, point):
        node_id = self._kd.insert(point)
        return node_id

    def add_edge(self, node_id, neighbor_id):
        self._edges[node_id].append(neighbor_id)
        self._edges[neighbor_id].append(node_id)

    def get_parent(self, child_id):
        return self._edges[child_id]

    def get_point(self, node_id):
        return self._kd.get_node(node_id).point

    def get_nearest_node(self, point):
        return self._kd.find_nearest_point(point)

    def get_neighbor_within_radius(self, point, radius):
        """
        Return a list of node_id within the radius
        """
        return self._kd.find_points_within_radius(point, radius)


class cell:
    def __init__(self):
        self.g = float('inf')
        self.parent = -1


class OBPRM:
    def __init__(self, fr, is_in_collision):
        self._fr = fr
        self._is_in_collision = is_in_collision

        self._q_step_size = 0.08
        self._joint_size = math.sqrt(0.04 ** 2 / 7) / 2
        self._radius = 0.7
        self._k = 8
        self._max_n_nodes = int(100000)

        self._project_step_size = 1e-1
        self._constraint_th = 1e-3

        self._smoothed_nodes = 0

    def sample_valid_joints(self):
        """
        The sampled configuration must be within the joint limits, but it does not check for collisions.
        """
        q = np.random.random(self._fr.num_dof) * (self._fr.joint_limits_high - self._fr.joint_limits_low) + self._fr.joint_limits_low
        return q

    def sample_near_joints(self, q):
        q_near = q
        for i in range(len(q)):
            lower = max(self._fr.joint_limits_low[i], q[i] - self._joint_size)
            upper = min(self._fr.joint_limits_high[i], q[i] + self._joint_size)
            q_near[i] = np.random.random() * (upper - lower) + lower
        return q_near

    def _is_seg_valid(self, q0, q1):
        """
        Check if the edge between q0 and q1 is collision free by interpolating the segment
        """
        qs = np.linspace(q0, q1, int(np.linalg.norm(q1 - q0) / self._q_step_size))
        for q in qs:
            if self._is_in_collision(q):
                return False
        return True

    def project_to_constraint(self, q0, constraint):
        q_proj = q0.copy()
        err, grad = constraint(q0)
        while err > self._constraint_th:
            J = self._fr.jacobian(q_proj)
            q_proj -= self._project_step_size * J.T.dot(grad)
            err, grad = constraint(q_proj)
        return q_proj

    def preprocess(self, graph, constraint=None):
        num_edges = 0
        had_collision = False
        sample_count = 0
        while len(graph) < self._max_n_nodes:
            # Sample valid joints
            if had_collision and sample_count < 80:
                q_sample = self.sample_near_joints(q_new)
            else:
                q_sample = self.sample_valid_joints()
            # Project to constraint
            q_new = self.project_to_constraint(q_sample, constraint)

            # Add the new node to the graph if it is collision free
            if not self._is_in_collision(q_new):
                sample_count = 0
                node_id = graph.insert_new_node(q_new)
                neighbor_ids = graph.get_neighbor_within_radius(q_new, self._radius)
                print('OB_PRM: Number of neighbors: {}'.format(len(neighbor_ids)))
                num_valid_neighbor = 0
                for neighbor_id in neighbor_ids:
                    q_neighbor = graph.get_point(neighbor_id)
                    if self._is_seg_valid(q_new, q_neighbor):
                        graph.add_edge(node_id, neighbor_id)
                        num_edges += 1
                        num_valid_neighbor += 1
                        if num_valid_neighbor >= self._k:
                            break
                print('OB_PRM: number of nodes: {}'.format(len(graph)))
                print('OB_PRM: number of edges {}'.format(num_edges))
                had_collision = False
            else:
                sample_count += 1
                had_collision = True
        print("OB_PRM: Graph is built successfully!")

    def smooth_path(self, path):
        print("OB_PRM: Start path smooth!")

        def getDistance(p):
            dist = 0
            prev = p[0]
            for q in p[1:]:
                dist += np.linalg.norm(q - prev)
                prev = q
            return dist

        for num_smoothed in range(self._smoothed_nodes):
            i = random.randint(0, len(path) - 2)
            j = random.randint(i + 1, len(path) - 1)
            if self._is_seg_valid(path[i], path[j]):
                if getDistance(path[i] + path[j]) < getDistance(path[i:j + 1]):
                    path = path[:i + 1] + path[j:]

        # Interpolating between two nodes
        path = [np.linspace(path[i], path[i + 1], int(np.linalg.norm(path[i + 1] - path[i]) / self._q_step_size)) for i
                in range(len(path) - 1)]
        path = list(itertools.chain.from_iterable(path))

        print("OB_PRM: Final path length after smooth is {}.".format(len(path)))

        return path

    def search(self, graph):

        def get_heuristic(graph, cur_id, target_id, use_heur=False):
            if use_heur:
                return np.linalg.norm(graph.get_point(target_id) - graph.get_point(cur_id))
            else:
                return 1

        road_map = collections.defaultdict(cell)  # stores the g_value, parent for the node
        road_map[graph.start_id].g = 0

        open_list = []
        heapq.heappush(open_list, (0, graph.start_id))  # Min-Heap, [f_value, node_id]
        close_list = set()

        found_path = False

        while open_list and not found_path:
            _, cur_id = heapq.heappop(open_list)
            if cur_id in close_list:
                continue

            close_list.add(cur_id)

            # find the neighbor
            neighbor_ids = graph.get_parent(cur_id)
            for next_id in neighbor_ids:
                if next_id == graph.target_id:
                    print("Path is Found!")
                    found_path = True
                    road_map[graph.target_id].parent = cur_id
                    break

                if road_map[next_id].g > road_map[cur_id].g + \
                        np.linalg.norm(graph.get_point(next_id) - graph.get_point(cur_id)):
                    road_map[next_id].g = road_map[cur_id].g + \
                                          np.linalg.norm(graph.get_point(next_id) - graph.get_point(cur_id))

                    f_value = road_map[next_id].g + get_heuristic(graph, next_id, graph.target_id, use_heur=False)
                    heapq.heappush(open_list, (f_value, next_id))
                    road_map[next_id].parent = cur_id

        path = []
        if found_path:
            backward_path = [graph.get_point(graph.target_id)]
            node_id = road_map[graph.target_id].parent
            while node_id != -1:
                backward_path.append(graph.get_point(node_id))
                node_id = (road_map[node_id]).parent

            path = backward_path[::-1]

            print("OB_PRM: Found a path! Path length is {}. ".format(len(path)))

        else:
            print('OB_PRM: Was not able to find a path!')

        return path

    def plan(self, q_start, q_target, constraint=None, args=None):
        if args.map3:
            graph_name = 'graph_obprm_map3.pickle'
        elif args.map2:
            graph_name = 'graph_obprm_map2.pickle'
        else:
            graph_name = 'graph_obprm_map1.pickle'

        if args.reuse_graph:
            graph = pickle.load(open(graph_name, 'rb'))
            print("OB_PRM: Reuse the graph.")
        else:
            graph = SimpleGraph(len(q_start), capacity=180000)
            s = time()
            self.preprocess(graph, constraint)
            print('OB_PRM: Build the graph in {:.2f}s'.format(time() - s))

            with open(graph_name, 'wb') as f:
                pickle.dump(graph, f, -1)
                print('OB_PRM: Graph is saved!')

        s = time()
        graph.start_id = graph.insert_new_node(q_start)
        neighbor_ids = graph.get_neighbor_within_radius(q_start, 1.0)
        print('OB_PRM: Found neighbor {} with q_start'.format(len(neighbor_ids)))
        for neighbor_id in neighbor_ids:
            q_neighbor = graph.get_point(neighbor_id)
            if self._is_seg_valid(q_start, q_neighbor):
                graph.add_edge(graph.start_id, neighbor_id)

        graph.target_id = graph.insert_new_node(q_target)
        neighbor_ids = graph.get_neighbor_within_radius(q_target, 1.0)
        print('OB_PRM: Found neighbor {} with q_target'.format(len(neighbor_ids)))
        for neighbor_id in neighbor_ids:
            q_neighbor = graph.get_point(neighbor_id)
            if self._is_seg_valid(q_target, q_neighbor):
                graph.add_edge(graph.target_id, neighbor_id)

        print('OB_PRM: Number of nodes connected with start: {}'.format(len(graph.get_parent(graph.start_id))))
        print('OB_PRM: Number of nodes connected with target: {}'.format(len(graph.get_parent(graph.target_id))))

        print('OB_PRM: Total number of nodes: {}'.format(len(graph)))
        path = self.search(graph)
        path = self.smooth_path(path)
        print('OB_PRM: Found the path in {:.2f}s'.format(time() - s))

        return path
