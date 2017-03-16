import math
import numpy


class Orientation:
    def __init__(self, direction, opposite=None):
        self.__direction = direction
        if opposite is None:
            opposite = Orientation((-direction[0], -direction[1]), self)
        self._opposite = opposite

    @property
    def reversed(self):
        return self._opposite

LEFT = Orientation((-1, 0))
DOWN = Orientation((0, -1))
RIGHT = LEFT.reversed
UP = DOWN.reversed

FOUR_WAY_ORIENTATIONS = [LEFT, DOWN, RIGHT, UP]


class Shape:
    @property
    def center_x(self):
        raise Exception("Not implemented!")

    @center_x.setter
    def center_x(self, x):
        raise Exception("Not implemented!")

    @property
    def center_y(self):
        raise Exception("Not implemented!")

    @center_y.setter
    def center_y(self, y):
        raise Exception("Not implemented!")

    @property
    def center(self):
        return self.center_x, self.center_y

    @center.setter
    def center(self, pos):
        self.center_x = pos[0]
        self.center_y = pos[1]

    @property
    def dimensions(self):
        return self.width, self.height

    @dimensions.setter
    def dimensions(self, dimensions):
        self.width = dimensions[0]
        self.height = dimensions[1]

    @property
    def width(self):
        raise Exception("Not implemented!")

    @width.setter
    def width(self, width):
        raise Exception("Not implemented!")

    @property
    def height(self):
        raise Exception("Not implemented!")

    @height.setter
    def height(self, height):
        raise Exception("Not implemented!")

    @property
    def left(self):
        return self.center_x - self.width/2

    @left.setter
    def left(self, left):
        self.translate_x(left - self.left)

    @property
    def down(self):
        return self.center_y - self.height/2

    @down.setter
    def down(self, down):
        self.translate_y(down - self.down)

    @property
    def right(self):
        return self.center_x + self.width/2

    @property
    def up(self):
        return self.center_y + self.height/2

    def has_area(self):
        raise Exception("Not implemented!")

    def to_bounding_box(self):
        return self.left, self.down, self.width, self.height

    def to_int_bounding_box(self):
        return int(self.left), int(self.down), int(self.width+0.5), int(self.height+0.5)

    def to_generous_int_bounding_box(self):
        return int(self.left-1), int(self.down-1), int(self.width+2), int(self.height+2)

    def to_bounding_rectangle(self):
        return Rectangle(self.left, self.down, self.width, self.height)

    def translate(self, delta):
        self.center = (self.center_x + delta[0], self.center_y + delta[1])

    def translate_x(self, delta_x):
        self.center = (self.center_x + delta_x, self.center_y)

    def translate_y(self, delta_y):
        self.center = (self.center_x, self.center_y + delta_y)

    def scale(self, scalar):
        self.width = self.width * scalar[0]
        self.height = self.height * scalar[1]

    def point_lies_within(self, point):
        raise Exception("Not implemented!")

    def collides(self, other):
        raise Exception("Not implemented!")
    
    def collides_with_axis(self, axis):
        raise Exception("Not implemented!")

    def collides_with_circle(self, circle):
        raise Exception("Not implemented!")

    def collides_with_line_segment(self, line_segment):
        raise Exception("Not implemented!")

    def collides_with_rectangle(self, rectangle):
        raise Exception("Not implemented!")

    def bounding_boxes_collide(self, other):
        return self.left < other.right and self.right > other.left and \
               self.down < other.up and self.up > other.down


class Axis(Shape):
    def __init__(self, offset, dimension):
        self.offset = offset
        self.dimension = dimension

    def has_area(self):
        return False

    @property
    def other_dimension(self):
        return (self.dimension + 1) % 2

    @property
    def center(self):
        to_return = [0, 0]
        to_return[self.other_dimension] = self.offset
        return to_return

    @property
    def center_x(self):
        return self.offset if self.dimension == 0 else 0

    @property
    def center_y(self):
        return self.offset if self.dimension == 1 else 0
    
    @property
    def width(self):
        return math.inf if self.dimension == 0 else 0

    @property
    def height(self):
        return math.inf if self.dimension == 1 else 0

    def collides(self, other):
        return other.collides_with_axis(self)

    def collides_with_axis(self, axis):
        return axis.dimension != self.dimension or axis.offset == self.offset

    def collides_with_circle(self, circle):
        return circle.collides_with_axis(self)

    def collides_with_line_segment(self, line_segment):
        return line_segment.collides_with_axis(self)

    def collides_with_rectangle(self, rectangle):
        return rectangle.collides_with_axis(self)


# class Oval(Shape):
#     def __init__(self, center, radii):
#         self.__center = center
#         self.__radii = radii


class Circle(Shape):
    def __init__(self, center, radius):
        self.__center = center
        self.__radius = radius

    def has_area(self):
        return True

    @property
    def center_x(self):
        return self.__center[0]

    @center_x.setter
    def center_x(self, x):
        self.__center[0] = x

    @property
    def center_y(self):
        return self.__center[1]

    @center_y.setter
    def center_y(self, y):
        self.__center[1] = y

    @property
    def center(self):
        return self.__center

    @center.setter
    def center(self, pos):
        self.__center = pos

    @property
    def dimensions(self):
        return self.__radius, self.__radius

    @property
    def width(self):
        return self.__radius * 2

    @property
    def height(self):
        return self.__radius * 2

    @property
    def radius(self):
        return self.__radius
        
    @radius.setter
    def radius(self, radius):
        self.__radius = radius

    def scale(self, scalar):
        if scalar[0] == scalar[1]:
            self.translate((-self.__radius, -self.__radius))
            self.__radius *= scalar[0]
            self.translate((self.__radius, self.__radius))
        else:
            print(scalar)
            raise Exception("Not implemented!")

    def point_lies_within(self, point):
        distance = numpy.linalg.norm([self.__center[0] - point[0], self.__center[1] - point[1]])
        return distance < self.__radius

    def collides(self, other):
        return other.collides_with_circle(self)

    def collides_with_axis(self, axis):
        circle_offset = self.center[axis.other_dimension]
        return circle_offset + self.__radius > axis.offset > circle_offset - self.__radius

    def collides_with_circle(self, circle):
        distance = numpy.linalg.norm([self.__center[0] - circle.center_x, self.__center[1] - circle.center_y])
        return distance < self.__radius + circle.radius

    def collides_with_line_segment(self, line_segment):
        return line_segment.collides_with_circle(self)

    def collides_with_rectangle(self, rectangle):
        return rectangle.collides_with_circle(self)


class LineSegment(Shape):
    def __init__(self, start_point, end_point):
        self.__start = start_point
        self.__end = end_point

    def has_area(self):
        return False

    @property
    def start_point(self):
        return self.__start

    @property
    def start_point_x(self):
        return self.__start[0]

    @property
    def start_point_y(self):
        return self.__start[1]

    @property
    def end_point(self):
        return self.__end

    @property
    def end_point_x(self):
        return self.__end[0]

    @property
    def end_point_y(self):
        return self.__end[1]

    @property
    def center_x(self):
        return (self.__start[0] + self.__end[0])/2

    @center_x.setter
    def center_x(self, x):
        current_center = self.center_x
        dx = x - current_center
        self.__start = (self.__start[0] + dx, self.__start[1])
        self.__end = (self.__end[0] + dx, self.__end[1])

    @property
    def center_y(self):
        return (self.__start[1] + self.__end[1])/2

    @center_y.setter
    def center_y(self, y):
        current_center = self.center_y
        dy = y - current_center
        self.__start = (self.__start[0], self.__start[1] + dy)
        self.__end = (self.__end[0], self.__end[1] + dy)

    @property
    def width(self):
        return abs(self.__start[0] - self.__end[0])

    # @property
    # def center(self):
    #     return self.center_x, self.center_y
    #
    # @center.setter
    # def center(self, pos):
    #     self.center_x = pos[0]
    #     self.center_y = pos[1]

    @property
    def height(self):
        return abs(self.__start[1] - self.__end[1])

    # @height.setter
    # def height(self, height):
    #     raise Exception("Not implemented!")

    @property
    def dimensions(self):
        return self.width, self.height

    @dimensions.setter
    def dimensions(self, dimensions):
        x_scale = dimensions[0] / self.width if self.width > 0 else 1
        if self.__start[0] > self.__end[0]:
            start_x = self.__start[0] * x_scale
            end_x = self.__end[0]
        else:
            end_x = self.__end[0] * x_scale
            start_x = self.__start[0]
        y_scale = dimensions[1] / self.height if self.height > 0 else 1
        if self.__start[1] > self.__end[1]:
            start_y = self.__start[1] * y_scale
            end_y = self.__end[1]
        else:
            end_y = self.__end[1] * y_scale
            start_y = self.__start[1]
        self.__start = (start_x, start_y)
        self.__end = (end_x, end_y)

    def to_vector(self):
        return self.width, self.height

    def intersect_with_line_segment(self, line_segment):
        width = self.width
        height = self.height
        other_width = line_segment.width
        other_height = line_segment.height

        dx = self.start_point_x - line_segment.start_point_x
        dy = self.start_point_y - line_segment.start_point_y
        denominator = -other_width * height + width * other_height
        s = (-height * dx + width * dy) / denominator
        t = (other_width * dy - other_height * dx) / denominator

        if 0 <= s <= 1 and 0 <= t <= 1:
            return self.start_point_x + (t * width), self.start_point_y + (t * height)
        return None

    def scale(self, scalar):
        self.dimensions = (self.dimensions[0] * scalar[0], self.dimensions[1] * scalar[1])

    def point_lies_within(self, point):
        return False

    def collides(self, other):
        return other.collides_with_line(self)

    def collides_with_axis(self, axis):
        return axis.collides_with_line_segment(self)

    def collides_with_circle(self, circle):
        vector = flip_vector(self.to_vector())
        circle_center = circle.center
        line_from_circle = LineSegment(circle_center, (circle_center[0] + vector[0], circle_center[1] + vector[1]))
        intersection = line_from_circle.intersect_with_line_segment(self)
        return circle.point_lies_within(intersection)

    def collides_with_line_segment(self, line):
        return self.intersect_with_line_segment(line) is not None

    def collides_with_rectangle(self, rectangle):
        return rectangle.collides_with_line_segment(self)


def rect(dimensions):
    return Rectangle(0, 0, dimensions[0], dimensions[1])


class Rectangle(Shape):
    def __init__(self, left, down, width, height):
        self.__left = left
        self.__down = down
        self.__width = width
        self.__height = height

    def __str__(self):
        return str((self.left, self.down, self.width, self.height))

    def has_area(self):
        return True

    @property
    def center_x(self):
        return self.__left + self.__width / 2

    @center_x.setter
    def center_x(self, x):
        self.__left = x - self.__width / 2

    @property
    def center_y(self):
        return self.__down + self.__height / 2

    @center_y.setter
    def center_y(self, y):
        self.__down = y - self.__height / 2

    @property
    def center(self):
        return self.__left + self.__width / 2, self.__down + self.__height / 2

    @center.setter
    def center(self, pos):
        self.center_x = pos[0]
        self.center_y = pos[1]

    @property
    def left(self):
        return self.__left

    @left.setter
    def left(self, left):
        self.translate_x(left - self.__left)

    @property
    def down(self):
        return self.__down

    @down.setter
    def down(self, down):
        self.translate_y(down - self.__down)

    @property
    def right(self):
        return self.__left + self.__width

    @property
    def up(self):
        return self.__down + self.__height

    @property
    def dimensions(self):
        return self.__width, self.__height

    @dimensions.setter
    def dimensions(self, dimensions):
        self.__width = dimensions[0]
        self.__height = dimensions[1]

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, width):
        self.__width = width

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, height):
        self.__height = height

    def left_bottom(self):
        return self.__left, self.__down

    def bottom_right(self):
        return self.__down, self.right

    def right_top(self):
        return self.right, self.up

    def top_left(self):
        return self.up, self.__left

    def left_segment(self):
        return LineSegment(self.top_left(), self.left_bottom())

    def down_segment(self):
        return LineSegment(self.left_bottom(), self.bottom_right())

    def right_segment(self):
        return LineSegment(self.bottom_right(), self.right_top())

    def up_segment(self):
        return LineSegment(self.right_top(), self.top_left())

    _RECTANGLE_ORIENTATION_SEGMENT_MAPPING = {LEFT: left_segment, DOWN: down_segment,
                                              RIGHT: right_segment, UP: up_segment}

    def get_segment(self, orientation):
        return Rectangle._RECTANGLE_ORIENTATION_SEGMENT_MAPPING[orientation](self)

    def scale(self, scalar):
        self.__width *= scalar[0]
        self.__height *= scalar[1]

    def point_lies_within(self, point):
        return self.__left < point[0] < self.right and self.down < point[1] < self.up

    def collides(self, other):
        return other.collides_with_rectangle(self)

    def collides_with_axis(self, axis):
        self.up > axis.offset > self.__down if axis.dimension == 0 else \
            self.right > axis.offset > self.__left

    def collides_with_circle(self, circle):
        if self.__left > circle.right or self.right < circle.left or \
                        self.up < circle.down or self.down > circle.up:
            return False
        if self.point_lies_within(circle.center):
            return True
        for orientation in FOUR_WAY_ORIENTATIONS:
            segment = self.get_segment(orientation)
            if segment.collides_with_circle(circle):
                return True
        return False

    def collides_with_line_segment(self, line_segment):
        if not self.bounding_boxes_collide(line_segment):
            return False
        if self.point_lies_within(line_segment.start_point):
            return True
        for orientation in FOUR_WAY_ORIENTATIONS[0:2]:
            segment = self.get_segment(orientation)
            if segment.collides_with_line_segment(line_segment):
                return True
        return False

    def collides_with_rectangle(self, rectangle):
        return self.bounding_boxes_collide(rectangle)


class Polygon(Shape):
    def __init__(self, points):
        self.__points = points
        self.__left_down = (0, 0)
        self.__width = 0
        self.__height = 0
        self.__recalc_values()

    def has_area(self):
        return True

    def __recalc_values(self):
        min_x = math.inf
        max_x = -math.inf
        min_y = math.inf
        max_y = -math.inf
        for point in self.__points:
            min_x = min(point[0], min_x)
            max_x = max(point[0], max_x)
            min_y = min(point[1], min_y)
            max_y = max(point[1], max_y)
        self.__width = max_x - min_x
        self.__height = max_y - min_y
        self.__left_down = (min_x, min_y)

    @property
    def center_x(self):
        return self.__left_down[0] + self.__width/2

    @center_x.setter
    def center_x(self, x):
        dx = x - self.center_x
        self.translate((dx, 0))

    @property
    def center_y(self):
        return self.__left_down[1] + self.__height/2

    @center_y.setter
    def center_y(self, y):
        dy = y - self.center_y
        self.translate((0, dy))

    @property
    def center(self):
        return self.center_x, self.center_y

    @center.setter
    def center(self, pos):
        dx = pos[0] - self.center_x
        dy = pos[1] - self.center_y
        self.translate((dx, dy))

    @property
    def dimensions(self):
        return self.__width, self.__height

    @dimensions.setter
    def dimensions(self, dimensions):
        scalar_x = dimensions[0] / self.__width
        scalar_y = dimensions[1] / self.__height
        new_points = []
        for point in self.__points:
            dx = point[0] - self.__left_down[0]
            dy = point[1] - self.__left_down[1]
            new_points.append((self.__left_down[0] + dx * scalar_x, self.__left_down[1] + dy * scalar_y))
        self.__points = new_points
        self.__width = dimensions[0]
        self.__height = dimensions[1]

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, width):
        self.dimensions = (width, self.__height)

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, height):
        self.dimensions = (self.__width, height)

    @property
    def left(self):
        return self.__left_down[0]

    @property
    def down(self):
        return self.__left_down[1]

    @property
    def right(self):
        return self.__left_down[0] + self.__width

    @property
    def up(self):
        return self.__left_down[1] + self.__height

    @property
    def points(self):
        return self.__points

    def translate(self, delta):
        new_points = []
        for point in self.__points:
            new_points.append((point[0] + delta[0], point[1] + delta[1]))
        self.__left_down = (self.__left_down[0] + delta[0], self.__left_down[1] + delta[1])
        self.__points = new_points

    def scale(self, scalar):
        self.dimensions = (self.width * scalar[0], self.height * scalar[1])


def flip_vector(vector):
    return -vector[1], vector[0]


X_AXIS = Axis(0, 0)
Y_AXIS = Axis(0, 1)
