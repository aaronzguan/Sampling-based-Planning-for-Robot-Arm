# -*- coding: utf-8 -*-

''' From https://pypi.org/project/KdQuery/
'''

"""Kd-tree implementation.

This module defines one possible kd-tree structure implementation and a
general method to find the nearest node for any kd-tree implementation.

"""
import math
from collections import deque, namedtuple
import heapq


def interval_condition(value, inf, sup, dist):
    """Checks if value belongs to the interval [inf - dist, sup + dist].
    """
    return inf - dist < value < sup + dist


def euclidean_dist(point1, point2):
    return math.sqrt(sum([math.pow(point1[i] - point2[i], 2)
                          for i in range(len(point1))]))


Node = namedtuple('Node', 'point region axis active left right data')
"""Internal representation of a node.

The tree is represented by a list of node. Each node is associated to a
point in the k-dimensional space, is inside a region, devises this region
in two parts according to an axis, has two child nodes and stores some
data.

"""


class KDTree:
    """Kd-tree implementation.

    This class defines one implementation of a kd-tree using a python list to
    save the methods from recursion.

    Args:
        k (int, optional): The number of dimensions of the space.
        capacity (int, optional): The maximum number of nodes in the tree.
        limits (:obj:`list` of :obj:`list` of float or int, optional): A list
            of size k, where each list contains two numbers defining the limits
            of the region which all the nodes will be. If none is passed as
            argument, the region will be all the space, that is, ]-inf, inf[
            for each one of the k axis.

    Attributes:
        node_list (:obj:`list` of :obj:Node): The list of nodes.
        size (int): The number of active nodes in the list.
        next_identifier (int): The identifier of the next node to be inserted
            in the list.
        k (int): The number of dimensions of the space.
        region (:obj:`list` of :obj:`list` of float or int, optional): A list
            of size k, where each list contains two numbers defining the limits
            of the region which all the nodes belong to. If none is passed as
            argument, the region will be all the space, that is, ]-inf, inf[
            for each one of the k axis.

    """

    def __init__(self, k=2, capacity=100000, limits=None):
        self.node_list = [None] * capacity
        self.size = 0
        self.next_identifier = 0
        self.k = k

        # The region of the space where all the points are
        self.region = limits if limits is not None \
            else [[-math.inf, math.inf]] * k

    def __len__(self):
        return self.size

    def __iter__(self):
        return (node for node in self.node_list
                if node is not None and node.active)

    def get_node(self, node_id):
        return self.node_list[node_id]

    def deactivate(self, node_id):
        """Deactivate the node identified by node_id.

        Deactivates the node corresponding to node_id, which means that
        it can never be the output of a nearest_point query.

        Note:
            The node is not removed from the tree, its data is steel available.

        Args:
            node_id (int): The node identifier (given to the user after
                its insertion).

        """
        node = self.node_list[node_id]
        self.node_list[node_id] = node._replace(active=False)

    def insert(self, point, data=None):
        """Insert a new node in the tree.

        Args:
            point (:obj:`tuple` of float or int): Stores the position of the
                node.
            data (:obj, optional): The information stored by the node.

        Returns:
            int: The identifier of the new node.

        Example:
            >>> tree = Tree(4, 800)
            >>> point = (3, 7)
            >>> data = {'name': Fresnel, 'label': blue, 'speed': 98.2}
            >>> node_id = tree.insert(point, data)

        """
        assert len(point) == self.k

        if self.size == 0:
            if self.region is None:
                self.region = [[-math.inf, math.inf]] * self.k
            axis = 0
            return self.new_node(point, self.region, axis, data)

        # Iteratively descends to one leaf
        current_id = 0
        while True:
            parent_node = self.node_list[current_id]
            axis = parent_node.axis
            if point[axis] < parent_node.point[axis]:
                next_id, left = parent_node.left, True
            else:
                next_id, left = parent_node.right, False

            if next_id is None:
                break

            current_id = next_id

        # Get the region delimited by the parent node
        region = parent_node.region[:]
        region[axis] = parent_node.region[axis][:]

        # Limit to the child's region
        limit = parent_node.point[axis]

        # Update reference to the new node
        if left:
            self.node_list[current_id] = parent_node._replace(left=self.size)
            region[axis][1] = limit
        else:
            self.node_list[current_id] = parent_node._replace(right=self.size)
            region[axis][0] = limit

        return self.new_node(point, region, (axis + 1) % self.k, data)

    def new_node(self, point, region, axis, data):
        node = Node(point, region, axis, True, None, None, data)

        # Identifier to new node
        node_id = self.next_identifier
        self.node_list[node_id] = node

        self.size += 1
        self.next_identifier += 1

        return node_id

    def find_nearest_point(self, query, dist_fun=euclidean_dist):
        """Find the point in the tree that minimizes the distance to the query.

        Args:
            query (:obj:`tuple` of float or int): Stores the position of the
                node.
            dist_fun (:obj:`function`, optional): The distance function,
                euclidean distance by default.

        Returns:
            :obj:`tuple`: Tuple of length 2, where the first element is the
                identifier of the nearest node, the second is the distance
                to the query.

        Example:
            >>> tree = Tree(2, 3)
            >>> tree.insert((0, 0))
            >>> tree.insert((3, 5))
            >>> tree.insert((-1, 7))
            >>> query = (-1, 8)
            >>> nearest_node_id, dist = tree.find_nearest_point(query)
            >>> dist
            1

        """
        def get_properties(node_id):
            return self.node_list[node_id][:6]

        return nearest_point(query, 0, get_properties, dist_fun)

    def find_points_within_radius(self, query, radius, dist_fun=euclidean_dist):
        def get_properties(node_id):
            return self.node_list[node_id][:6]
        return neighbor_within_radius(query, radius, 0, get_properties, dist_fun)


def neighbor_within_radius(query, radius, root_id, get_properties, dist_fun=euclidean_dist):
    k = len(query)
    neighbors = []

    # stack_node: stack of identifiers to nodes within a region that
    # contains the query.
    # stack_look: stack of identifiers to nodes within a region that
    # does not contains the query.
    stack_node = deque([root_id])
    stack_look = deque()

    while stack_node or stack_look:

        if stack_node:
            node_id = stack_node.pop()
            look_node = False
        else:
            node_id = stack_look.pop()
            look_node = True

        point, region, axis, active, left, right = get_properties(node_id)

        # Should consider this node?
        # As it is within a region that does not contains the query, maybe
        # there is no chance to find a closer node in this region
        if look_node:
            inside_region = True
            for i in range(k):
                inside_region &= interval_condition(query[i], region[i][0],
                                                    region[i][1], radius)
            if not inside_region:
                continue

        # Update the distance only if the node is active.
        if active:
            node_distance = dist_fun(query, point)
            if node_distance <= radius and node_distance != 0:
                neighbors.append((-node_distance, node_id))

        if query[axis] < point[axis]:
            side_node = left
            side_look = right
        else:
            side_node = right
            side_look = left

        if side_node is not None:
            stack_node.append(side_node)

        if side_look is not None:
            stack_look.append(side_look)

    heapq.heapify(neighbors)
    return [item[1] for item in neighbors]


def nearest_point(query, root_id, get_properties, dist_fun=euclidean_dist):
    """Find the point in the tree that minimizes the distance to the query.

    This method implements the nearest_point query for any structure
    implementing a kd-tree. The only requirement is a function capable to
    extract the relevant properties from a node representation of the
    particular implementation.

    Args:
        query (:obj:`tuple` of float or int): Stores the position of the
            node.
        root_id (:obj): The identifier of the root in the kd-tree
            implementation.
        get_properties (:obj:`function`): The function to extract the
            relevant properties from a node, namely its point, region,
            axis, left child identifier, right child identifier and
            if it is active. If the implementation does not uses
            the active attribute the function should return always True.
        dist_fun (:obj:`function`, optional): The distance function,
            euclidean distance by default.

    Returns:
        :obj:`tuple`: Tuple of length 2, where the first element is the
            identifier of the nearest node, the second is the distance
            to the query.

    """

    k = len(query)
    dist = math.inf

    nearest_node_id = None

    # stack_node: stack of identifiers to nodes within a region that
    # contains the query.
    # stack_look: stack of identifiers to nodes within a region that
    # does not contains the query.
    stack_node = deque([root_id])
    stack_look = deque()

    while stack_node or stack_look:

        if stack_node:
            node_id = stack_node.pop()
            look_node = False
        else:
            node_id = stack_look.pop()
            look_node = True

        point, region, axis, active, left, right = get_properties(node_id)

        # Should consider this node?
        # As it is within a region that does not contains the query, maybe
        # there is no chance to find a closer node in this region
        if look_node:
            inside_region = True
            for i in range(k):
                inside_region &= interval_condition(query[i], region[i][0],
                                                    region[i][1], dist)
            if not inside_region:
                continue

        # Update the distance only if the node is active.
        if active:
            node_distance = dist_fun(query, point)
            if nearest_node_id is None or dist > node_distance:
                nearest_node_id = node_id
                dist = node_distance

        if query[axis] < point[axis]:
            side_node = left
            side_look = right
        else:
            side_node = right
            side_look = left

        if side_node is not None:
            stack_node.append(side_node)

        if side_look is not None:
            stack_look.append(side_look)

    return nearest_node_id, dist
