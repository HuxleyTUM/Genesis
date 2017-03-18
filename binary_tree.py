import copy
import shapes


class BinaryTree:
    def __init__(self, dimension, depth):
        self.true_dimension = dimension
        dim = (2**int((depth+1)/2), 2**int(depth/2))
        if dimension[0] > dimension[1]:
            self.tree_dimension = dim
        else:
            self.tree_dimension = dim[::-1]
        self.conversion = [x/y for x, y in zip(self.true_dimension, self.tree_dimension)]
        self.leaves = [None for x in range(dim[0] * dim[1])]
        self.root_node = self.divide((0, 0), self.tree_dimension, 0, depth)
        for leaf in self.leaves:
            leaf.find_neighbours()
        self.elements_to_leaves = {}

    def tree_to_true_position(self, position):
        return [x*y for x, y in zip(position, self.conversion)]

    def divide(self, tree_pos, tree_dimension, current_depth, to_depth):
        if current_depth < to_depth:
            split_dimension = 1 if tree_dimension[0] > tree_dimension[1] else 0
            other_dimension = (split_dimension + 1) % 2
            next_dimensions = [tree_dimension[0], tree_dimension[1]]
            next_dimensions[other_dimension] /= 2
            higher_pos = [tree_pos[0], tree_pos[1]]
            higher_pos[other_dimension] += next_dimensions[other_dimension]
            lower = self.divide(tree_pos, next_dimensions, current_depth + 1, to_depth)
            higher = self.divide(higher_pos, next_dimensions, current_depth + 1, to_depth)
            true_position = self.tree_to_true_position(higher_pos)
            axis = shapes.Axis(true_position[other_dimension], split_dimension)
            node = Node(axis, lower, higher)
            return node
        else:
            true_pos = self.tree_to_true_position(tree_pos)
            true_dim = self.tree_to_true_position(tree_dimension)
            leaf = Leaf(self, tree_pos, shapes.Rectangle(true_pos[0], true_pos[1], true_dim[0], true_dim[1]))
            self.leaves[int(tree_pos[0] + tree_pos[1] * self.tree_dimension[0])] = leaf
            return leaf

    def classify(self, element, shape):
        leaves = self.root_node.find_leaves(shape)
        self.elements_to_leaves[element] = leaves
        for leaf in leaves:
            leaf.elements.add(element)

    def reclassify(self, element, shape):
        # leaf_set = self.elements_to_leaves[element]
        # candidates = []
        # leafs_to_remove = set()
        # for leaf in leaf_set:
        #     h_contained = 0
        #     v_contained = 0
        #     if shape.left > leaf.bounding_rectangle.right:
        #         leafs_to_remove.add(leaf)
        #         continue
        #     if shape.right > leaf.bounding_rectangle.left:
        #         leafs_to_remove.add(leaf)
        #         continue
        #
        #
        #     if False:
        #         self.__add_leaf(leaf.left, leaf_list, leaf_set)
        #         h_contained += 1
        #     if shape.right > leaf.bounding_rectangle.right:
        #         self.__add_leaf(leaf.right, leaf_list, leaf_set)
        #         h_contained += 1
        #     if shape.down < leaf.bounding_rectangle.down:
        #         self.__add_leaf(leaf.down, leaf_list, leaf_set)
        #         v_contained += 1
        #     if shape.up > leaf.bounding_rectangle.up:
        #         self.__add_leaf(leaf.up, leaf_list, leaf_set)
        #         v_contained += 1

        leaves = set()
        self.root_node._find_leaves(shape, leaves)
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
    
    def __add_leaf(self, leaf, leaf_list, leaf_set):
        if leaf is not None and leaf not in leaf_set:
            leaf_set.add(leaf)
            leaf_list.append(leaf)

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
    def __init__(self, axis, lower, higher):
        self.axis = axis
        self.lower = lower
        self.higher = higher

        self.parent = None

    @property
    def size(self):
        return self.lower.size + self.higher.size

    def find_leaves(self, shape, leaves=None):
        if leaves is None:
            leaves = set()
        self._find_leaves(shape, leaves)
        return leaves

    def _find_leaves(self, shape, leaves):
        if self.axis.dimension == 0:
            lower = shape.down < self.axis.offset
            higher = shape.up > self.axis.offset
        else:
            lower = shape.left < self.axis.offset
            higher = shape.right > self.axis.offset
        if lower:
            self.lower._find_leaves(shape, leaves)
        if higher:
            self.higher._find_leaves(shape, leaves)

    def merge(self, merge_into=None):
        if merge_into is None:
            merge_into = set()
        self._merge(merge_into)
        return merge_into

    def _merge(self, merge_into):
        self.lower._merge(merge_into)
        self.higher._merge(merge_into)


class Leaf:
    def __init__(self, tree, tree_pos, bounding_rectangle):
        self.bounding_rectangle = bounding_rectangle
        self.x = int(tree_pos[0])
        self.y = int(tree_pos[1])
        self.tree = tree
        self.elements = set()
        self.left = None
        self.right = None
        self.down = None
        self.up = None

    @property
    def size(self):
        return len(self.elements)

    def find_node(self, shape=None):
        return self

    def _merge(self, merge_into):
        merge_into |= self.elements

    def merge(self, merge_into=None):
        if merge_into is None:
            merge_into = set(self.elements)
        else:
            merge_into |= self.elements
        return merge_into

    def _find_leaves(self, shape, leaves):
        leaves.add(self)

    def find_leaves(self, shape, leaves=None):
        if leaves is None:
            leaves = {self}
        else:
            leaves.add(self)
        return leaves
    
    def find_neighbours(self):
        self.left = self.find_neighbour(self.x-1, self.y)
        self.right = self.find_neighbour(self.x+1, self.y)
        self.down = self.find_neighbour(self.x, self.y-1)
        self.up = self.find_neighbour(self.x, self.y+1)

    def find_neighbour(self, x, y):
        if 0 > x or x >= self.tree.tree_dimension[0] or 0 > y or y >= self.tree.tree_dimension[1]:
            return None
        return self.tree.leaves[int(x + y * self.tree.tree_dimension[0])]
    

