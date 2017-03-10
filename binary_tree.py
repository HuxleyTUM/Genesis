import copy
import shapes


class BinaryTree:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.root_node = self.divide([0, 0], [width, height], 0, depth)
        self.elements_to_leaves = {}

    def divide(self, pos, dimensions, current_depth, to_depth):
        if current_depth < to_depth:
            split_dimension = 1 if dimensions[0] > dimensions[1] else 0
            other_dimension = (split_dimension + 1) % 2
            next_dimensions = copy.copy(dimensions)
            next_dimensions[other_dimension] /= 2
            right_pos = copy.copy(pos)
            right_pos[other_dimension] += next_dimensions[other_dimension]
            left = self.divide(pos, next_dimensions, current_depth + 1, to_depth)
            right = self.divide(right_pos, next_dimensions, current_depth + 1, to_depth)
            axis = shapes.Axis(right_pos[other_dimension], split_dimension)
            node = Node(axis, left, right)
            return node
        else:
            return Leaf()

    def classify(self, element, shape):
        leaves = self.root_node.find_leaves(shape)
        self.elements_to_leaves[element] = leaves
        for leaf in leaves:
            leaf.elements.add(element)

    def reclassify(self, element, shape):
        leaves = self.root_node.find_leaves(shape)
        old_leaves = self.elements_to_leaves[element]
        update_mapping = False
        for leaf in old_leaves - leaves:
            leaf.elements.remove(element)
            update_mapping = True
        for leaf in leaves - old_leaves:
            leaf.elements.add(element)
            update_mapping = True
        if update_mapping:
            self.elements_to_leaves[element] = leaves

    def contains(self, element, shape):
        return element in self.elements_to_leaves

    def get_collision_candidates(self, shape):
        to_return = []
        for leaf in self.root_node.find_leaves(shape):
            to_return += leaf.elements
        return to_return

    def find_node(self, shape):
        return self.root_node.find_leaves(shape)

    @property
    def size(self):
        return len(self.elements_to_leaves)

    @property
    def elements(self):
        return self.elements_to_leaves.keys()

    def remove(self, element):
        old_leaves = self.elements_to_leaves[element]
        for leaf in old_leaves:
            leaf.elements.remove(element)
        del self.elements_to_leaves[element]


class Node:
    def __init__(self, axis, left, right):
        self.axis = axis
        self.left = left
        self.right = right

    @property
    def size(self):
        return self.left.size + self.right.size

    def find_leaves(self, shape, leaves=None):
        if leaves is None:
            leaves = set()
        if self.axis.collides(shape):
            leaves |= self.left.find_leaves(shape, leaves)
            leaves |= self.right.find_leaves(shape, leaves)
        else:
            pos = shape.center
            if pos[(self.axis.dimension+1) % 2] < self.axis.offset:
                leaves |= self.left.find_leaves(shape, leaves)
            else:
                leaves |= self.right.find_leaves(shape, leaves)
        return leaves

    def merge(self, merge_into=None):
        if merge_into is None:
            merge_into = set()
        merge_into |= self.left.merge()
        merge_into |= self.right.merge()
        return merge_into


class Leaf:
    def __init__(self):
        self.elements = set()

    @property
    def size(self):
        return len(self.elements)

    def find_node(self, shape=None):
        return self

    def merge(self, merge_into=None):
        if merge_into is None:
            merge_into = set()
        merge_into |= self.elements
        return self.elements

    def find_leaves(self, shape, leaves=None):
        if leaves is None:
            leaves = {self}
        else:
            leaves.add(self)
        return leaves

