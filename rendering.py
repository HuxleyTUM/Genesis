import threading
import copy
import operator
import shapes
import numpy as np
import time
import pygame
import functools
import events
import colours
import numbers
import genesis

render_lock = threading.Lock()


def listen_for_key_console(self):
    def w(op):
        self._render_width = op(self._render_width, 10)

    def h(op):
        self._render_height = op(self._render_height, 10)

    def p(op):
        self._physics_delay = op(self._physics_delay, 0.1)

    def r(op):
        self._render_delay = op(self._render_delay, 5)

    commands = {"w": w, "h": h, "p": p, "r": r}
    operators = {"+": operator.add, "-": operator.sub}
    while True:
        print("listening..")
        value_input = input("command:")
        print("read: \"" + value_input + "\"")
        if len(value_input) == 2:
            f = commands[value_input[0]]
            o = operators[value_input[1]]
            if f is not None and o is not None:
                f(o)


class Camera:
    @property
    def is_identity(self):
        return self.position is (0, 0) and self.zoom is (1, 1)

    def zoom_by_factor(self, zoom_factor):
        raise Exception("Not implemented in "+str(type(self)))

    def pan(self, delta):
        raise Exception("Not implemented in "+str(type(self)))

    @property
    def position(self):
        raise Exception("Not implemented in "+str(type(self)))

    @property
    def zoom(self):
        raise Exception("Not implemented in "+str(type(self)))

    def transform_point_to_parent(self, point):
        raise Exception("Not implemented in "+str(type(self)))

    def transform_point_from_parent(self, point):
        raise Exception("Not implemented in "+str(type(self)))

    def transform_x_to_parent(self, point):
        raise Exception("Not implemented in "+str(type(self)))

    def transform_y_to_parent(self, point):
        raise Exception("Not implemented in "+str(type(self)))

    def transform_shape_to_parent(self, shape, transform_shape=False):
        raise Exception("Not implemented in "+str(type(self)))


class RelativeCamera(Camera):
    def __init__(self, position=(0, 0), zoom=(1, 1)):
        self.__zoom = zoom
        self.__position = position

    def __str__(self):
        return "(position="+str(self.position)+", zoom="+str(self.zoom)+")"

    @property
    def position(self):
        return self.__position

    @property
    def zoom(self):
        return self.__zoom

    def zoom_by_factor(self, zoom_factor):
        self.__zoom = [a * b for a, b in zip(self.zoom, zoom_factor)]

    def pan(self, delta):
        self.__position = [x + d for x, d in zip(self.position, delta)]

    def transform_point_to_parent(self, point):
        if not self.is_identity:
            return self.transform_x_to_parent(point), self.transform_y_to_parent(point)
        else:
            return point

    def transform_point_from_parent(self, point):
        return point[0] / self.zoom[0] + self.position[0], point[1] / self.zoom[1] + self.position[1]

    def transform_x_to_parent(self, point):
        return (point[0] - self.position[0]) * self.zoom[0]

    def transform_y_to_parent(self, point):
        return (point[1] - self.position[1]) * self.zoom[1]

    def transform_shape_to_parent(self, shape, transform_shape=False):
        if not self.is_identity:
            if not transform_shape:
                shape = copy.deepcopy(shape)
            left_down = self.transform_point_to_parent((shape.left, shape.down))
            shape.translate((left_down[0] - shape.left, left_down[1] - shape.down))
            shape.scale(self.zoom)
        return shape


class AbsoluteCamera(Camera):
    def __init__(self, canvas_offset, canvas_dimensions, screen_offset, screen_dimensions):
        self._canvas_dimensions = canvas_dimensions
        self.canvas_offset = canvas_offset
        self.screen_offset = screen_offset
        self._screen_dimensions = screen_dimensions
        self.__update_scaling()

    def __update_scaling(self):
        self.scaling = np.divide(self._screen_dimensions, self._canvas_dimensions)

    @property
    def canvas_dimensions(self):
        return self._canvas_dimensions

    @property
    def screen_bounding(self):
        return self.screen_offset[0], self.screen_offset[1], self._screen_dimensions[0], self._screen_dimensions[1]

    @property
    def canvas_rectangle(self):
        return shapes.Rectangle(self.canvas_offset[0], self.canvas_offset[1],
                                self._canvas_dimensions[0], self._canvas_dimensions[1])

    @property
    def screen_rectangle(self):
        return shapes.Rectangle(self.screen_offset[0], self.screen_offset[1],
                                self._screen_dimensions[0], self._screen_dimensions[1])

    @canvas_dimensions.setter
    def canvas_dimensions(self, canvas_dimensions):
        self._canvas_dimensions = canvas_dimensions
        self.__update_scaling()

    @property
    def screen_dimensions(self):
        return self._screen_dimensions

    @screen_dimensions.setter
    def screen_dimensions(self, screen_dimensions):
        self._screen_dimensions = screen_dimensions
        self.__update_scaling()

    def transform_point_from_parent(self, point):
        return (point[0] - self.screen_offset[0]) / self.scaling[0] + self.canvas_offset[0], \
               (point[1] - self.screen_offset[1]) / self.scaling[1] + self.canvas_offset[1]

    def transform_point_to_parent(self, point):
        return self.transform_x_to_parent(point), self.transform_y_to_parent(point)

    def transform_x_to_parent(self, point):
        return (point[0] - self.canvas_offset[0]) * self.scaling[0] + self.screen_offset[0]

    def transform_y_to_parent(self, point):
        return (point[1] - self.canvas_offset[1]) * self.scaling[1] + self.screen_offset[1]

    def transform_shape_to_parent(self, shape, transform_shape=False):
        if not self.is_identity:
            if not transform_shape:
                shape = copy.deepcopy(shape)
            left_down = self.transform_point_to_parent((shape.left, shape.down))
            shape.translate((left_down[0] - shape.left, left_down[1] - shape.down))
            shape.scale(self.scaling)
        return shape


class Canvas:
    def __init__(self, back_ground_colour=colours.BLACK):
        self.has_changed_listeners = []

        self.mouse_pressed_event_listeners = []
        self.mouse_released_event_listeners = []
        self.mouse_canceled_event_listeners = []
        self.mouse_wheel_up_event_listeners = []
        self.mouse_wheel_down_event_listeners = []
        self.pressed_key_left_event_listeners = []
        self.pressed_key_right_event_listeners = []
        self.pressed_key_up_event_listeners = []
        self.pressed_key_down_event_listeners = []

        self.graphics_changed_listener = None
        self.last_drawn_screen_box = None
        self.__back_ground_colour = back_ground_colour
        self.screen_boxes_to_update = []
        self.__reblit = True

    @property
    def reblit(self):
        return self.__reblit

    @reblit.setter
    def reblit(self, reblit):
        self.__reblit = reblit
        if self.parent_canvas is not None:
            self.parent_canvas.redraw_children = True

    @property
    def redraw_surface(self):
        raise Exception("Not implemented!")

    @redraw_surface.setter
    def redraw_surface(self, redraw_surface):
        raise Exception("Not implemented!")

    def reset_surface(self, global_bounding):
        bounding_dimensions = global_bounding.dimensions
        dimensions = tuple(int(round(x)) for x in bounding_dimensions)
        if self.surface is None or all(x != y for x, y in zip(self.surface.get_size(), dimensions)):
            self.surface = pygame.Surface(dimensions, pygame.SRCALPHA, 32).convert_alpha()
            if self.back_ground_colour is not None:
                self.surface.fill(self.back_ground_colour)
        else:
            if self.back_ground_colour is not None:
                self.surface.fill(self.back_ground_colour)
            else:
                self.surface.fill(colours.TRANSPARENT)

    def paint_to_parent(self, screen_boxes_to_update, outlines, parent_bounding, update_any=True):
        draw = self.redraw_surface or (isinstance(self, Container) and self.redraw_children) and self.is_visible
        blit = self.reblit or draw and self.is_visible
        update_old = self.reblit and update_any
        update_new = (self.redraw_surface or self.reblit) and update_any
        global_bounding = None
        if draw or blit or update_new or len(self.screen_boxes_to_update) > 0:
            global_bounding = self.global_bounding_rectangle
        if update_new:
            drawn_screen_bounding = global_bounding.to_generous_int_bounding_box()
            clipped_box = parent_bounding.clip_with_box(drawn_screen_bounding)
            if clipped_box is not None:
                screen_boxes_to_update.append(clipped_box)
                outlines.append(clipped_box)
        else:
            drawn_screen_bounding = None
        if update_old and self.last_drawn_screen_box is not None:
            self.__clip_and_append(parent_bounding, self.last_drawn_screen_box, screen_boxes_to_update)
        if draw:
            update_children = update_any and not update_new
            self.paint_to_surface(screen_boxes_to_update, outlines, global_bounding, parent_bounding, update_children)
        if blit:
            if drawn_screen_bounding is None:
                drawn_screen_bounding = global_bounding.to_generous_int_bounding_box()
            self.last_drawn_screen_box = drawn_screen_bounding
            self.blit_to_parent(outlines, global_bounding, parent_bounding)
        if len(self.screen_boxes_to_update) > 0:
            for box in self.screen_boxes_to_update:
                self.__clip_and_append(global_bounding, box, screen_boxes_to_update)
            self.screen_boxes_to_update = []

        self.redraw_surface = False
        self.reblit = False

    def paint_to_surface(self, screen_boxes_to_update, outlines, global_bounding, parent_bounding, update_children=True):
        self.reset_surface(global_bounding)

    def __clip_and_append(self, clipping, box, append_to):
        clipped = clipping.clip_with_box(box)
        if clipped is not None:
            append_to.append(clipped)

    def blit_to_parent(self, outlines, global_bounding, parent_global):
        if self.parent_canvas is not None:
            pos = (global_bounding.left - parent_global.left, global_bounding.down - parent_global.down)
            if type(self) is TextGraphic:
                outlines.append(global_bounding.to_bounding_box())
            self.parent_canvas.surface.blit(self.surface, pos)

    # def paint_text(self, label, bounding_rectangle, screen_delta, transform_shape=False):
    #     if not transform_shape:
    #         bounding_rectangle = copy.deepcopy(bounding_rectangle)
    #     bounding_rectangle.translate((-self.position[0], -self.position[1]))
    #     return self.parent_canvas.paint_text(label, bounding_rectangle, screen_delta, False)

    def recalc_point(self, point, delta_x, delta_y, scale_x, scale_y):
        return (point[0] - delta_x) * scale_x, (point[1] - delta_y) * scale_y

    def paint_shape(self, shape, colour, border_width, dimensions):
        if type(shape) is shapes.Circle:
            bounding_box = (0, 0, int(round(dimensions[0])), int(round(dimensions[1])))
            max_border_width = max(min(bounding_box[2], bounding_box[3]) - 1, 0)
            border_width = min(border_width, max_border_width)
            pygame.draw.ellipse(self.surface, colour, bounding_box, border_width)
        elif type(shape) is shapes.Axis:
            axis = shape
            screen_pos_from = axis.center
            screen_pos_to = copy.copy(screen_pos_from)
            screen_pos_from[axis.dimension] = 0
            screen = self.screen
            screen_pos_to[axis.dimension] = screen.dimensions[axis.dimension]
            pygame.draw.line(self.surface, colour, screen_pos_from, screen_pos_to, max(border_width, 1))
        elif type(shape) is shapes.LineSegment:
            scale_x = dimensions[0] / shape.width if shape.width > 0 else 0
            scale_y = dimensions[1] / shape.height if shape.height > 0 else 0
            first = ((shape.start_point[0] - shape.left) * scale_x, (shape.start_point[1] - shape.down) * scale_y)
            second = ((shape.end_point[0] - shape.left) * scale_x, (shape.end_point[1] - shape.down) * scale_y)
            pygame.draw.line(self.surface, colour, first, second, border_width)
        elif type(shape) is shapes.Rectangle:
            bounding_box = (0, 0, dimensions[0], dimensions[1])
            pygame.draw.rect(self.surface, colour, bounding_box, border_width)
        elif isinstance(shape, shapes.PointShape):
            new_points = []
            scale_x = dimensions[0] / shape.width
            scale_y = dimensions[1] / shape.height
            for point in shape.points:
                new_points.append(self.recalc_point(point, shape.left, shape.down, scale_x, scale_y))
            if type(shape) is shapes.Polygon:
                pygame.draw.polygon(self.surface, colour, new_points, border_width)
            elif type(shape) is shapes.PointLine:
                pygame.draw.lines(self.surface, colour, False, new_points, border_width)
        else:
            raise "Unknown shape: " + str(type(shape))

    @property
    def surface(self):
        raise Exception("Not implemented!")

    @surface.setter
    def surface(self, surface):
        raise Exception("Not implemented!")

    def scale(self, scalar):
        raise Exception("Not implemented in " + str(type(self)))

    def add_has_changed_listener(self, listener):
        self.has_changed_listeners.append(listener)

    def remove_has_changed_listener(self, listener):
        self.has_changed_listeners.remove(listener)

    def notify_listeners_of_change(self, *args):
        for listener in self.has_changed_listeners:
            listener()

    @property
    def children(self):
        return ()

    @property
    def is_visible(self):
        raise Exception("Not implemented in " + str(type(self)))

    @is_visible.setter
    def is_visible(self, value):
        raise Exception("Not implemented in " + str(type(self)))

    @property
    def screen(self):
        return self.parent_canvas.screen if self.parent_canvas is not None else None

    @property
    def parent_canvas(self):
        return None

    @parent_canvas.setter
    def parent_canvas(self, parent_canvas):
        raise Exception("Can't set parent canvas!")

    @property
    def position(self):
        return None

    @position.setter
    def position(self, position):
        raise Exception("Can't set position!")

    def translate(self, delta):
        if delta[0] != 0 or delta[1] != 0:
            self.reblit = True
            pos = self.position
            self.position = (pos[0] + delta[0], pos[1] + delta[1])

    @property
    def global_bounding_rectangle(self):
        return self.transform_shape_to_screen(self.local_bounding_rectangle)

    @property
    def local_bounding_rectangle(self):
        raise Exception("Not implemented!")

    @property
    def local_bounding_box(self):
        raise Exception("Not implemented!")

    @property
    def back_ground_colour(self):
        return self.__back_ground_colour

    @back_ground_colour.setter
    def back_ground_colour(self, value):
        if not all(x == y for x, y in zip(self.back_ground_colour, value)):
            self.__back_ground_colour = value
            self.redraw_surface = True

    def mouse_wheel_scrolled_up(self, event):
        events.fire_listeners(self.mouse_wheel_up_event_listeners, event)

    def mouse_wheel_scrolled_down(self, event):
        events.fire_listeners(self.mouse_wheel_down_event_listeners, event)

    def scroll_up(self):
        self.translate((0, 10))

    def scroll_down(self):
        self.translate((0, -10))

    def scroll_left(self):
        self.translate((10, 0))

    def scroll_right(self):
        self.translate((-10, 0))

    def mouse_pressed(self, event):
        events.fire_listeners(self.mouse_pressed_event_listeners, event)

    def mouse_released(self, event):
        events.fire_listeners(self.mouse_released_event_listeners, event)

    def mouse_canceled(self, event):
        events.fire_listeners(self.mouse_canceled_event_listeners, event)

    def up_key_pressed(self, event):
        events.fire_listeners(self.pressed_key_up_event_listeners, event)

    def down_key_pressed(self, event):
        events.fire_listeners(self.pressed_key_down_event_listeners, event)

    def right_key_pressed(self, event):
        events.fire_listeners(self.pressed_key_right_event_listeners, event)

    def left_key_pressed(self, event):
        events.fire_listeners(self.pressed_key_left_event_listeners, event)

    @property
    def local_bounding_rectangle(self):
        return shapes.Rectangle(0, 0, self.local_bounding_rectangle.width, self.local_bounding_rectangle.height)

    def transform_point_from_screen(self, point):
        if self.parent_canvas is None:
            raise Exception("This canvas has no parent canvas")
        else:
            point_from_parent = self.parent_canvas.transform_point_from_screen(point)
            point_in_parent = (point_from_parent[0] - self.position[0],
                               point_from_parent[1] - self.position[1])
            return point_in_parent

    def transform_point_to_screen(self, point):
        current_canvas = self
        while current_canvas is not None:
            point = current_canvas.transform_point_to_parent(point)
            current_canvas = current_canvas.parent_canvas
        return point

    def transform_shape_to_screen(self, shape, transform_shape=False):
        current_canvas = self
        while current_canvas is not None:
            shape = current_canvas.transform_shape_to_parent(shape, transform_shape)
            current_canvas = current_canvas.parent_canvas
            transform_shape = True
        return shape

    def transform_shape_to_surface(self, shape, transform_shape=False):
        if not transform_shape:
            shape = copy.copy(shape)
        return self.parent_canvas.transform_shape_to_surface(shape)

    def transform_point_to_parent(self, point):
        return point[0] + self.position[0], point[1] + self.position[1]

    def transform_shape_to_parent(self, shape, transform_shape=False):
        if not transform_shape:
            shape = copy.deepcopy(shape)
        shape.translate(self.position)
        return shape

    # def paint_text(self, label, canvas_rectangle, screen_delta, transform_shape=False, canvas=None):
    #     raise Exception("Not implemented in " + str(type(self)))
    #
    # def paint_shape(self, shape, colour, border_width, screen_delta, transform_shape=False, canvas=None):
    #     raise Exception("Not implemented in " + str(type(self)))


class Container(Canvas):
    def __init__(self, local_bounding_rectangle, back_ground_colour=colours.BLACK, children_overlap=False):
        super().__init__(back_ground_colour)
        self.__local_bounding_rectangle = local_bounding_rectangle
        self.__children = []
        self.__is_visible = True
        self.__redraw_children = True
        self.__redraw_surface = True
        self.__surface = None
        self.__children_overlap = children_overlap

    def paint_to_surface(self, screen_boxes_to_update, outlines, global_bounding, parent_bounding, update_children=True):
        self.reset_surface(global_bounding)
        for child in self.children:
            if self.__children_overlap:
                child.reblit = True
            child.paint_to_parent(screen_boxes_to_update, outlines, global_bounding, update_children)
        self.__redraw_children = False

    @property
    def redraw_children(self):
        return self.__redraw_children

    @redraw_children.setter
    def redraw_children(self, redraw_children):
        self.__redraw_children = redraw_children
        if redraw_children and self.parent_canvas is not None and self.parent_canvas.redraw_children is False:
            self.parent_canvas.redraw_children = True

    @property
    def surface(self):
        return self.__surface

    @surface.setter
    def surface(self, surface):
        self.__surface = surface

    @property
    def redraw_surface(self):
        return self.__redraw_surface

    @redraw_surface.setter
    def redraw_surface(self, redraw_surface):
        self.__redraw_surface = redraw_surface
        if redraw_surface:
            self.redraw_children = True
            for canvas in self.children:
                canvas.reblit = True

    @property
    def is_visible(self):
        return self.__is_visible

    @is_visible.setter
    def is_visible(self, value):
        self.__is_visible = value

    @property
    def local_bounding_rectangle(self):
        return self.__local_bounding_rectangle

    @property
    def local_bounding_box(self):
        raise self.__local_bounding_rectangle.to_bounding_box()

    def transform_point_from_screen(self, point):
        return self.camera.transform_point_from_parent(Canvas.transform_point_from_screen(self, point))

    def transform_shape_to_surface(self, shape, transform_shape=False):
        new_shape = self.camera.transform_shape_to_parent(shape, transform_shape)
        return self.parent_canvas.transform_shape_to_surface(new_shape, new_shape != shape)

    def transform_point_to_parent(self, point):
        point = self.camera.transform_point_to_parent(point)
        return point[0] + self.position[0], point[1] + self.position[1]

    def transform_shape_to_parent(self, shape, transform_shape=False):
        if not transform_shape:
            shape = copy.copy(shape)
        shape = self.camera.transform_shape_to_parent(shape, True)
        shape.translate(self.position)
        return shape

    def translate(self, delta):
        if delta[0] != 0 or delta[1] != 0:
            self.reblit = True
            pos = self.position
            self.position = (pos[0] + delta[0], pos[1] + delta[1])

    def add_and_center_canvas(self, canvas):
        self.add_canvas(canvas)
        graphic_rectangle = canvas.transform_shape_to_parent(canvas.local_bounding_rectangle)
        old_center = self.local_bounding_rectangle.center
        graphic_center = graphic_rectangle.center
        canvas.translate((old_center[0] - graphic_center[0], old_center[1] - graphic_center[1]))

    @property
    def camera(self):
        return None

    def add_canvases(self, canvases, position=(0, 0), canvas_index=None):
        for canvas in canvases:
            self.add_canvas(canvas, position, canvas_index)

    def add_canvas(self, canvas, position=None, canvas_index=None):
        canvas.parent_canvas = self
        # if canvas.redraw_surface or (isinstance(canvas, Container) and canvas.redraw_children):
        self.redraw_children = True
        if position is not None:
            canvas.position = position
        if canvas_index is None:
            self.__children.append(canvas)
        else:
            self.__children.insert(canvas_index, canvas)

    def remove_canvases(self, canvases):
        for canvas in canvases:
            self.remove_canvas(canvas)

    def remove_canvas(self, canvas):
        if canvas.last_drawn_screen_box is not None:
            self.screen_boxes_to_update.append(canvas.last_drawn_screen_box)
        self.screen_boxes_to_update.append(canvas.global_bounding_rectangle.to_bounding_box()) # todo: do we need this?
        self.screen_boxes_to_update += canvas.screen_boxes_to_update
        canvas.screen_boxes_to_update = []

        canvas.parent_canvas = None
        self.__children.remove(canvas)
        self.redraw_children = True

    @property
    def children(self):
        return self.__children

    @property
    def parent_canvas(self):
        return None

    @parent_canvas.setter
    def parent_canvas(self, parent_canvas):
        raise Exception("Can't set parent canvas!")

    @property
    def position(self):
        return None

    @position.setter
    def position(self, position):
        raise Exception("Can't set position!")


class SimpleContainer(Container):
    def __init__(self, local_bounding_rectangle, camera=RelativeCamera(), back_ground_area=None,
                 border_width=1, border_colour=None, back_ground_colour=colours.BLACK,
                 children_overlap=False):
        super().__init__(local_bounding_rectangle, back_ground_colour, children_overlap=children_overlap)
        self.__border_colour = border_colour
        self.__border_width = border_width
        self.__parent_canvas = None
        self.__position = (0, 0)
        self.__camera = copy.copy(camera)
        self.__back_ground_colour = back_ground_colour
        # self.__back_ground_graphic = None
        self.__border_graphic = None

        if back_ground_area is None:
            self.__back_ground_area = local_bounding_rectangle
        else:
            self.__back_ground_area = back_ground_area
        self.back_ground_colour = back_ground_colour
        self.border_colour = border_colour
        self.border_width = border_width

    @property
    def border_colour(self):
        return self.__border_colour
    
    @border_colour.setter
    def border_colour(self, border_colour):
        self.__set_border(border_colour, self.__border_width)

    @property
    def border_width(self):
        return self.__border_width

    @border_width.setter
    def border_width(self, border_width):
        self.__set_border(self.__border_colour, border_width)

    def __set_border(self, border_colour, border_width):
        if border_colour is not None and border_width > 0:
            if self.__border_graphic is None:
                self.__border_graphic = SimpleOutlineGraphic(self.__back_ground_area, border_colour, border_width)
                self.add_canvas(self.__border_graphic)
            else:
                self.__border_graphic.border_colour = border_colour
                self.__border_graphic.border_width = border_width
        else:
            if self.__border_graphic is not None:
                self.remove_canvas(self.__border_graphic)
                self.__border_graphic = None
        self.__border_colour = border_colour
        self.__border_width = border_width

    @property
    def parent_canvas(self):
        return self.__parent_canvas

    @parent_canvas.setter
    def parent_canvas(self, parent_canvas):
        self.__parent_canvas = parent_canvas

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, position):
        self.__position = position

    @property
    def camera(self):
        return self.__camera

    # def paint_text(self, label, label_bounding, screen_delta=(0, 0), transform_shape=False, canvas=None):
    #     if canvas is None:
    #         canvas = self
    #     if not transform_shape:
    #         label_bounding = copy.copy(label_bounding)
    #     if label_bounding.bounding_boxes_collide(self.local_bounding_rectangle):
    #         if not self.camera.is_identity:
    #             new_left_bottom = self.camera.transform_point_to_parent((label_bounding.left, label_bounding.down))
    #             delta = new_left_bottom[0] - label_bounding.left, new_left_bottom[1] - label_bounding.down
    #             label_bounding.translate(delta)
    # 
    #         screen_delta = self.camera.transform_point_to_parent(screen_delta)
    #         screen_delta = (screen_delta[0] + self.position[0], screen_delta[1] + self.position[1])
    #         return self.parent_canvas.paint_text(label, label_bounding, screen_delta, True, canvas)
    #     else:
    #         return None
    # 
    # def paint_shape(self, shape, colour, border_width, screen_delta, transform_shape=False, canvas=None):
    #     if canvas is None:
    #         canvas = self
    #     if not transform_shape:
    #         shape = copy.deepcopy(shape)
    #     if shape.bounding_boxes_collide(self.local_bounding_rectangle):
    #         shape = self.camera.transform_shape_to_parent(shape, True)
    #         screen_delta = self.camera.transform_point_to_parent(screen_delta)
    #         screen_delta = (screen_delta[0] + self.position[0], screen_delta[1] + self.position[1])
    #         # shape.translate(self.position)
    #         return self.parent_canvas.paint_shape(shape, colour, border_width, screen_delta, True, canvas)
    #     else:
    #         return None


class AtomicGraphic(Canvas):
    def __init__(self, possibly_obstructed=False):
        super().__init__(back_ground_colour=None)
        self.__parent_canvas = None
        self.__redraw_surface = True

    @property
    def redraw_surface(self):
        return self.__redraw_surface

    @redraw_surface.setter
    def redraw_surface(self, redraw_surface):
        self.__redraw_surface = redraw_surface
        if redraw_surface and self.parent_canvas is not None:
            self.parent_canvas.redraw_children = True

    @property
    def surface(self):
        raise Exception("Not implemented!")

    @surface.setter
    def surface(self, surface):
        raise Exception("Not implemented!")

    # @properedraw

    # def paint_shape(self, shape, colour, border_width, screen_delta, transform_shape=False, canvas=None):
    #     if not transform_shape:
    #         shape = copy.deepcopy(shape)
    #     shape.translate((-self.position[0], -self.position[1]))
    #     return self.parent_canvas.paint_shape(shape, colour, border_width, screen_delta, False, self)

    @property
    def local_bounding_box(self):
        raise Exception("Not implemented in "+str(type(self)))

    @property
    def local_bounding_rectangle(self):
        raise Exception("Not implemented in "+str(type(self)))

    def translate(self, delta):
        raise Exception("Not implemented in "+str(type(self)))

    def scale(self, scalar):
        raise Exception("Not implemented in " + str(type(self)))

    @property
    def is_visible(self):
        raise Exception("Not implemented in " + str(type(self)))

    @is_visible.setter
    def is_visible(self, value):
        raise Exception("Not implemented in " + str(type(self)))

    @property
    def parent_canvas(self):
        return self.__parent_canvas

    @parent_canvas.setter
    def parent_canvas(self, parent_canvas):
        self.__parent_canvas = parent_canvas

    @property
    def position(self):
        bounding_rectangle = self.local_bounding_rectangle
        return bounding_rectangle.left, bounding_rectangle.down

    @position.setter
    def position(self, position):
        current = self.position
        delta = (position[0] - current[0], position[1] - current[1])
        self.translate(delta)


class TextGraphic(AtomicGraphic):
    def __init__(self, text, __font_type, font_size, text_colour=colours.WHITE, is_visible=True, cache_values=False):
        super().__init__()
        self.__font_size = font_size
        self.__font_type = __font_type
        self.cache_values = cache_values
        self.__text_colour = text_colour
        self.__pyg_label = None
        self.__local_bounding_rectangle = shapes.Rectangle(0, 0, 1, 1)
        self.__text = text
        self.__is_visible = is_visible
        self.__rerender_label()
    
    @property
    def font_size(self):
        return self.__font_size

    @font_size.setter
    def font_size(self, font_size):
        if self.__font_size != font_size:
            self.__font_size = font_size
            self.redraw_surface = True

    @property
    def font_type(self):
        return self.font_type

    @font_type.setter
    def font_type(self, font_type):
        if self.__font_type != font_type:
            self.font_type = font_type
            self.redraw_surface = True

    @property
    def surface(self):
        return self.__pyg_label

    @surface.setter
    def surface(self, surface):
        raise Exception("Can't set surface")

    @property
    def position(self):
        shape = self.__local_bounding_rectangle
        return shape.left, shape.down

    @position.setter
    def position(self, position):
        current = self.position
        delta = (position[0] - current[0], position[1] - current[1])
        self.translate(delta)

    @property
    def local_bounding_box(self):
        return self.local_bounding_rectangle.to_bounding_box()

    @property
    def local_bounding_rectangle(self):
        dim = self.__local_bounding_rectangle.dimensions
        return shapes.Rectangle(0, 0, dim[0], dim[1])

    @property
    def global_bounding_rectangle(self):
        global_pos = self.transform_point_to_screen((0, 0))
        rectangle = self.local_bounding_rectangle
        return shapes.Rectangle(global_pos[0], global_pos[1], rectangle.width, rectangle.height)

    @property
    def text_colour(self):
        return self.__text_colour

    @text_colour.setter
    def text_colour(self, text_colour):
        if any(x != y for x, y in zip(self.__text_colour, text_colour)):
            self.__text_colour = text_colour
            self.redraw_surface = True

    def translate(self, delta):
        if delta[0] != 0 or delta[1] != 0:
            self.__local_bounding_rectangle.translate(delta)
            self.reblit = True

    @property
    def text(self):
        return self.__text

    @text.setter
    def text(self, text):
        if text != self.__text:
            self.__text = text
            self.redraw_surface = True

    def __rerender_label(self):
        if self.cache_values:
            self.__pyg_label = Fonts.cached_arial_label(self.__font_type, self.__font_size, self.__text)
        else:
            font = Fonts.font(self.__font_type, self.__font_size)
            self.__pyg_label = font.render(self.__text, 1, self.__text_colour)
        self.__local_bounding_rectangle.dimensions = self.__pyg_label.get_size()
        self.reblit = True

    @property
    def is_visible(self):
        return self.__is_visible

    @is_visible.setter
    def is_visible(self, is_visible):
        if is_visible != self.__is_visible:
            self.__is_visible = is_visible
            self.redraw_surface = True

    def paint_to_surface(self, screen_boxes_to_update, outlines, global_bounding, parent_bounding, update_children=True):
        self.__rerender_label()


class ShapedGraphic(AtomicGraphic):
    def __init__(self, possibly_obstructed=False):
        super().__init__()
        self.__surface = None

    @property
    def surface(self):
        return self.__surface

    @surface.setter
    def surface(self, surface):
        self.__surface = surface

    @property
    def position(self):
        shape = self.shape
        return shape.left, shape.down

    @position.setter
    def position(self, position):
        current = self.position
        delta = (position[0] - current[0], position[1] - current[1])
        self.translate(delta)

    @property
    def local_bounding_box(self):
        return self.local_bounding_rectangle.to_bounding_box()

    @property
    def local_bounding_rectangle(self):
        dim = self.shape.dimensions
        return shapes.Rectangle(0, 0, dim[0], dim[1])

    @property
    def shape(self):
        raise Exception("Not implemented in "+str(type(self)))

    @shape.setter
    def shape(self, shape):
        raise Exception("Not implemented in "+str(type(self)))

    def translate(self, delta):
        if delta[0] != 0 or delta[1] != 0:
            self.shape.translate(delta)
            self.reblit = True

    @property
    def is_visible(self):
        raise Exception("Not implemented in "+str(type(self)))

    @is_visible.setter
    def is_visible(self, value):
        raise Exception("Not implemented in "+str(type(self)))


class OutlineGraphic(ShapedGraphic):
    def __init__(self, possibly_obstructed=False):
        super().__init__()

    @property
    def border_colour(self):
        raise Exception("Not implemented in "+str(type(self)))

    @border_colour.setter
    def border_colour(self, value):
        raise Exception("Not implemented in "+str(type(self)))

    @property
    def border_width(self):
        raise Exception("Not implemented in "+str(type(self)))

    @border_width.setter
    def border_width(self, value):
        raise Exception("Not implemented in "+str(type(self)))

    @property
    def shape(self):
        raise Exception("Not implemented in "+str(type(self)))

    @shape.setter
    def shape(self, shape):
        raise Exception("Not implemented in "+str(type(self)))

    @property
    def is_visible(self):
        raise Exception("Not implemented in "+str(type(self)))

    @is_visible.setter
    def is_visible(self, value):
        raise Exception("Not implemented in "+str(type(self)))
    
    def paint_to_surface(self, screen_boxes_to_update, outlines, global_bounding, parent_bounding, update_children=True):
        self.reset_surface(global_bounding)
        self.paint_shape(self.shape, self.border_colour, self.border_width, global_bounding.dimensions)


class SimpleOutlineGraphic(OutlineGraphic):
    def __init__(self, shape, border_colour, border_width=1, is_visible=True):
        super().__init__()
        self.__is_visible = is_visible
        self._border_width = border_width
        self._border_colour = border_colour
        self._shape = shape

    @property
    def border_colour(self):
        return self._border_colour

    @border_colour.setter
    def border_colour(self, border_colour):
        self._border_colour = border_colour
        self.redraw_surface = True

    @property
    def border_width(self):
        return self._border_width

    @border_width.setter
    def border_width(self, border_width):
        self._border_width = border_width
        self.redraw_surface = True

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self.redraw_surface = True

    @property
    def is_visible(self):
        return self.__is_visible

    @is_visible.setter
    def is_visible(self, is_visible):
        if is_visible != self.__is_visible:
            self.__is_visible = is_visible
            self.reblit = True


class MonoColouredGraphic(ShapedGraphic):
    def __init__(self):
        super().__init__()

    @property
    def fill_colour(self):
        raise Exception("Not implemented in "+str(type(self)))

    @fill_colour.setter
    def fill_colour(self, fill_colour):
        raise Exception("Not implemented in "+str(type(self)))

    @property
    def shape(self):
        raise Exception("Not implemented in "+str(type(self)))

    @shape.setter
    def shape(self, shape):
        raise Exception("Not implemented in "+str(type(self)))

    @property
    def is_visible(self):
        raise Exception("Not implemented in "+str(type(self)))

    @is_visible.setter
    def is_visible(self, value):
        raise Exception("Not implemented in "+str(type(self)))
    
    def paint_to_surface(self, screen_boxes_to_update, outlines, global_bounding, parent_bounding, update_children=True):
        self.reset_surface(global_bounding)
        self.paint_shape(self.shape, self.fill_colour, 0, global_bounding.dimensions)


class SimpleMonoColouredGraphic(MonoColouredGraphic):
    def __init__(self, shape, fill_colour, is_visible=True):
        super().__init__()
        self._fill_colour = fill_colour
        self._shape = shape
        self.__is_visible = is_visible

    @property
    def fill_colour(self):
        return self._fill_colour

    @fill_colour.setter
    def fill_colour(self, fill_colour):
        self._fill_colour = fill_colour
        self.redraw_surface = True

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self.redraw_surface = True

    @property
    def is_visible(self):
        return self.__is_visible

    @is_visible.setter
    def is_visible(self, is_visible):
        if is_visible != self.__is_visible:
            self.__is_visible = is_visible


class Table(SimpleContainer):
    def __init__(self, font_type, font_size, text_colour, column_count, padding=5, cache_values=False):
        super().__init__(shapes.rect((1, 1)))
        self.font_type = font_type
        self.padding = padding
        self.column_count = column_count
        self.text_colour = text_colour
        self.font_size = font_size
        self.row_graphics = []
        self.row_width = [0] * column_count
        self.cache_values = cache_values

    def add_rows(self, rows):
        for row in rows:
            self.add_row(row)

    def add_row(self, row_text):
        row = []
        for column_index in range(self.column_count):
            text_graphic = TextGraphic(str(row_text[column_index]), self.font_type, self.font_size, self.text_colour,
                                       cache_values=self.cache_values)
            x = 0 if column_index == 0 else sum(self.row_width[0:column_index])
            row_height = text_graphic.local_bounding_rectangle.height
            y = len(self.row_graphics) * text_graphic.local_bounding_rectangle.height
            text_graphic.translate((x, y))
            self.local_bounding_rectangle.height = max(self.local_bounding_rectangle.height, y + row_height)
            row.append(text_graphic)
            self.add_canvas(text_graphic)
            self.__check_column_width(column_index, text_graphic.local_bounding_rectangle.width + self.padding)
        self.row_graphics.append(row)

    def __check_column_width(self, column_index, width):
        dx = width - self.row_width[column_index]
        if dx > 0:
            self.local_bounding_rectangle.width += dx
            for row_to_fix in self.row_graphics:
                for graphic in row_to_fix[column_index + 1:]:
                    graphic.translate((dx, 0))
            self.row_width[column_index] = width

    def set(self, x, y, text):
        text_graphic = self.row_graphics[y][x]
        text_graphic.text = str(text)
        self.__check_column_width(x, text_graphic.local_bounding_rectangle.width + self.padding)

    @property
    def row_count(self):
        return len(self.row_graphics)


class ValueDisplay(Table):
    def __init__(self, font_type, font_size, text_colour, precision=2):
        super().__init__(font_type, font_size, text_colour, 2, cache_values=True)
        self.precision = precision
        self.value_updaters = []

    def add_row(self, row_text):
        val = row_text[1]
        if callable(val):
            self.value_updaters.append((self.row_count, row_text[1]))
            val = row_text[1]()
        Table.add_row(self, (row_text[0], self.__clean_input(val)))

    def refresh_values(self):
        for value_updater in self.value_updaters:
            self.set(1, value_updater[0], self.__clean_input(value_updater[1]()))

    def set(self, x, y, text):
        Table.set(self, x, y, self.__clean_input(text))

    def __clean_input(self, val_input):
        val_input = self.__clean_value(val_input)
        if not isinstance(val_input, str) and hasattr(val_input, "__getitem__"):
            val_input = [self.__clean_value(x) for x in val_input]
        return str(val_input)

    def __clean_value(self, value):
        if isinstance(value, numbers.Number):
            return round(value, self.precision)
        return value


class ScrollingPane(SimpleContainer):
    def __init__(self, local_bounding_rectangle, scroll_vertically, scroll_horizontally, camera=RelativeCamera()):
        super().__init__(local_bounding_rectangle, camera)
        pane_area = copy.copy(local_bounding_rectangle)
        pane_area.translate((3, 0))
        pane_area.width -= 6
        self.pane = SimpleContainer(pane_area)

        self.add_canvas(self.pane)
        self.scroll_horizontally = scroll_horizontally
        self.scroll_vertically = scroll_vertically

        pane = self.pane
        def scroll_down_function(x):
            if self.scroll_vertically and pane.position[1] <= 0:
                pane.scroll_up()
        def scroll_up_function(x):
            pane_bottom = pane.local_bounding_rectangle.height + pane.position[1]
            if self.scroll_vertically and pane_bottom > self.local_bounding_rectangle.height:
                pane.scroll_down()
        def scroll_right_function(x):
            if self.scroll_horizontally: pane.scroll_right()
        def scroll_left_function(x):
            if self.scroll_horizontally: pane.scroll_left()

        self.mouse_wheel_up_event_listeners.append(scroll_down_function)
        self.mouse_wheel_down_event_listeners.append(scroll_up_function)
        self.pressed_key_left_event_listeners.append(scroll_left_function)
        self.pressed_key_right_event_listeners.append(scroll_right_function)
        self.pressed_key_down_event_listeners.append(scroll_down_function)
        self.pressed_key_up_event_listeners.append(scroll_up_function)


class Button(SimpleContainer):
    def __init__(self, button_area, border_width=1, back_ground_colour=colours.grey(200), border_colour=None,
                 darken_on_press=True):
        super().__init__(button_area, border_width=border_width, border_colour=border_colour,
                         back_ground_colour=back_ground_colour)
        self.darken_on_press = darken_on_press
        self.action_listeners = []
        self.mouse_pressed_event_listeners.append(self.__mouse_pressed)
        self.mouse_released_event_listeners.append(self.__mouse_released)
        self.mouse_canceled_event_listeners.append(self.__mouse_canceled)
        self.is_pressed = False

    def __mouse_pressed(self, event):
        if not self.is_pressed:
            self.is_pressed = True
            self._adjust_colour()
            event.consume()

    def __mouse_released(self, event):
        if self.is_pressed:
            self.is_pressed = False
            self._adjust_colour()
            events.fire_listeners(self.action_listeners, event)
            event.consume()

    def _adjust_colour(self):
        if self.darken_on_press and self.back_ground_colour is not None:
            colour_factor = 0.5 if self.is_pressed else 2
            self.back_ground_colour = colours.adjust_colour(self.back_ground_colour, colour_factor)

    def __mouse_canceled(self, event):
        if self.is_pressed:
            self.is_pressed = False
            self._adjust_colour()
            event.consume()


class RectangularButton(Button):
    def __init__(self, dimensions):
        super().__init__(shapes.rect(dimensions))


class TextButton(RectangularButton):
    def __init__(self, text, font_type, font_size=12):
        super().__init__((1, 1))
        self.font_type = font_type
        self.label = TextGraphic(text, self.font_type, font_size, colours.BLACK)
        self.add_canvas(self.label)
        self.local_bounding_rectangle.dimensions = self.label.local_bounding_rectangle.dimensions


class ButtonBar(SimpleContainer):
    def __init__(self, dimensions, padding=5):
        super().__init__(shapes.rect(dimensions), border_width=1, border_colour=colours.WHITE,
                         back_ground_colour=colours.BLACK)
        self.padding = padding
        self.last_x = 0
        self.button_height = dimensions[1] - self.padding * 2

    def add_button(self, button):
        self.add_canvas(button, (self.last_x + self.padding, self.padding))
        scaling = self.button_height / button.local_bounding_rectangle.height
        self.last_x += button.local_bounding_rectangle.width*scaling + self.padding


class Fonts:
    _monospaced_fonts = {}
    _arial_fonts = {}
    _fonts = {}
    _labels = {}

    @staticmethod
    def cached_arial_label(font_str, size, text):
        sizes = Fonts.__put_and_get(Fonts._labels, font_str, lambda: {})
        labels = Fonts.__put_and_get(sizes, size, lambda: {})
        label = Fonts.__put_and_get(labels, text, lambda: Fonts.font(font_str, size).render(text, 1, colours.WHITE))
        return label

    @staticmethod
    def __put_and_get(dict, key, f):
        if key not in dict:
            val = f()
            dict[key] = val
            return val
        else:
            return dict[key]

    @staticmethod
    def font(font_str, size):
        sizes = Fonts.__put_and_get(Fonts._fonts, font_str, lambda: {})
        return Fonts.__put_and_get(sizes, size, lambda: pygame.font.SysFont(font_str, size))

    @staticmethod
    def monospaced_font(size):
        return Fonts.font("monospaced", size)

    @staticmethod
    def arial_font(size):
        return Fonts.font("arial", size)


class Screen(Container):
    def __init__(self, dimensions):
        super().__init__(shapes.rect(dimensions))
        pygame.init()
        flags = pygame.FULLSCREEN | pygame.DOUBLEBUF
        self.py_screen = pygame.display.set_mode(dimensions)
        self.surface = pygame.Surface(self.py_screen.get_size())
        # self.py_screen.set_alpha(None)
        self.dimensions = dimensions
        self.redraw_surface = False
        self.visualize_boundings = False

    def paint(self):
        screen_boxes_to_update = []
        outlines = []
        rectangle = self.global_bounding_rectangle
        self.paint_to_parent(screen_boxes_to_update, outlines, rectangle)
        if self.visualize_boundings:
            for outline in outlines:
                pygame.draw.rect(self.surface, colours.RED, outline, 1)
        self.py_screen.blit(self.surface, (0, 0))
        return screen_boxes_to_update

    @property
    def global_bounding_rectangle(self):
        return self.local_bounding_rectangle

    @property
    def screen(self):
        return self

    def transform_point_to_screen(self, point):
        return point

    def transform_shape_to_screen(self, shape, transform_shape=False):
        return shape

    def transform_point_from_screen(self, point):
        return point

    def transform_shape_to_surface(self, shape, transform_shape=False):
        return shape

    def transform_point_to_parent(self, point):
        return point

    def transform_shape_to_parent(self, shape, transform_shape=False):
        return shape

    # def paint_text(self, label, bounding_rectangle, screen_delta, transform_shape=False, canvas=None):
    #     # if canvas is None:
    #     #     canvas = self
    #     dirty_rect = bounding_rectangle.to_bounding_box()
    #     # dirty_rect = canvas.surface.blit(label, (bounding_rectangle.left, bounding_rectangle.down))
    #     # print("blitted to "+str(dirty_rect))
    #     return dirty_rect[0] + screen_delta[0], dirty_rect[1] + screen_delta[1], dirty_rect[2], dirty_rect[3]

    # def paint_shape(self, shape, colour, border_width, screen_delta, transform_shape=False, canvas=None):
    #     if canvas is None:
    #         canvas = self
    #     bounding_box = shape.to_int_bounding_box()
    #     max_border_width = max(min(bounding_box[2], bounding_box[3]) - 1, 0)
    #     border_width = min(border_width, max_border_width)
    # 
    #     if type(shape) is shapes.Circle:
    #         dirty_rect = pygame.draw.ellipse(canvas.surface, colour, bounding_box, border_width)
    #     elif type(shape) is shapes.Axis:
    #         axis = shape
    #         screen_pos_from = axis.center
    #         screen_pos_to = copy.copy(screen_pos_from)
    #         screen_pos_from[axis.dimension] = 0
    #         screen_pos_to[axis.dimension] = self.dimensions[axis.dimension]
    #         dirty_rect = pygame.draw.line(canvas.surface, colour, screen_pos_from, screen_pos_to, max(border_width, 1))
    #     elif type(shape) is shapes.LineSegment:
    #         line_segment = shape
    #         dirty_rect = pygame.draw.line(canvas.surface, colour, line_segment.start_point, line_segment.end_point)
    #     elif type(shape) is shapes.Rectangle:
    #         dirty_rect = pygame.draw.rect(canvas.surface, colour, bounding_box, border_width)
    #     elif type(shape) is shapes.Polygon:
    #         dirty_rect = pygame.draw.polygon(canvas.surface, colour, shape.points, border_width)
    #     elif type(shape) is shapes.PointLine:
    #         dirty_rect = pygame.draw.lines(canvas.surface, colour, False, shape.points, border_width)
    #     else:
    #         raise "Unknown shape: " + str(type(shape))
    #     return dirty_rect[0] + screen_delta[0] - 1, dirty_rect[1] + screen_delta[1] - 1,\
    #            dirty_rect[2] + 2, dirty_rect[3] + 2

    @property
    def camera(self):
        return None


class Frame:
    def __init__(self, frame_counter, rects_to_update):
        self.rects_to_update = rects_to_update
        self.frame_counter = frame_counter
        self.rects_rendered = None


class GraphicsSource:
    @property
    def graphics(self):
        raise Exception("Not implemented in "+str(type(self)))


class Renderer:
    def __init__(self):
        pass

    def render(self):
        pass


class PyGameRenderer(Renderer):
    def __init__(self, screen, render_clock=None, thread_render_clock=None):
        super().__init__()
        self.screen = screen
        self.thread_render_clock = thread_render_clock
        self.render_clock = render_clock
        self._last_render_time = -1
        self._current_frame_counter = 0
        self._last_frame = None
        self.screen_boxes_to_update = []
        self.__visualize_boundings = False

    def visualize_boundings(self, *args):
        self.__visualize_boundings = not self.__visualize_boundings
        # if not self.__visualize_boundings:
        #     self.screen.redraw_surface = True

    def render(self):
        if self.render_clock is not None:
            render_clock = self.render_clock
            render_clock.tick()
        else:
            render_clock = None

        self._last_render_time = time.time()

        frame = Frame(self._current_frame_counter, self.screen_boxes_to_update)
        self.screen_boxes_to_update = []
        self._last_frame = frame
        self._current_frame_counter += 1
        self.render_with_pygame(frame)
        if render_clock is not None:
            render_clock.tock()

    def render_with_pygame(self, frame):
        render_lock.acquire()
        self.screen.visualize_boundings = self.__visualize_boundings
        clock = self.thread_render_clock
        if clock is not None:
            clock.tick()
        boxes = frame.rects_to_update
        self.screen.visualize_boundings = self.__visualize_boundings
        boxes += self.screen.paint()

        pygame.display.update(boxes)

        if clock is not None:
            clock.tock()
        render_lock.release()

        # def render_canvas(self, canvas, clipping_rectangle, update_boxes, boxes_to_outline, update_any=True):
        #
        #
        # def render_atomic(self, atomic_graphic, drawn_screen_bounding, global_bounding):
        #     pos = atomic_graphic.position
        #     if isinstance(atomic_graphic, ShapedGraphic):
        #         shape = atomic_graphic.shape
        #         if isinstance(atomic_graphic, OutlineGraphic):
        #             colour = atomic_graphic.border_colour
        #             border_width = atomic_graphic.border_width
        #         elif isinstance(atomic_graphic, MonoColouredGraphic):
        #             colour = atomic_graphic.fill_colour
        #             border_width = 0
        #         else:
        #             raise Exception("unknown graphic: "+str(type(atomic_graphic)))
        #         drawn_bounding = atomic_graphic.paint_shape_x(shape, colour, border_width, global_bounding)
        #     elif isinstance(atomic_graphic, TextGraphic):
        #         bounding = atomic_graphic.local_bounding_rectangle
        #         drawn_bounding = atomic_graphic.paint_text(None, bounding, pos)
        #     else:
        #         raise Exception("unknown graphic: "+str(type(atomic_graphic)))
        #     if drawn_bounding is None:
        #         drawn_bounding = drawn_screen_bounding
        #     return drawn_bounding

# class AsciiRenderer(Renderer):
#     def __init__(self, canvas_dimensions, screen_dimensions=(120, 20)):
#         self.screen_dimensions = screen_dimensions
#         self._source_dimensions = canvas_dimensions
#         self._render_left = 0
#         self._render_top = 0
#         self._last_render_time = -1
#
#     def render(self, graphics):
#         shapes_to_render = []
#         pixels = []
#         for creature in self._environment.living_creatures:
#             for organ in creature.organs:
#                 if organ is not creature.body and organ.shape is not None:
#                     shapes_to_render.append(copy.deepcopy(organ.shape))
#                     pixels.append("X")
#             shapes_to_render.append(copy.deepcopy(creature.body.shape))
#             pixels.append(creature.name[len(creature.name) - 1])
#         for food in self._environment.food_pellets:
#             shapes_to_render.append(copy.deepcopy(food.shape))
#             pixels.append(".")
#         additional_1 = ["width(w): " + str(self._render_width), "height(h): " + str(self._render_height),
#                         "ticks: " + str(self._environment.tick_count),
#                         "frames/s: " + str(1 / (time.time() - self._last_render_time)),
#                         "physics/s: " + str(1 / self._environment.last_tick_delta)]
#         side_info = []
#         for creature in sorted(self._environment.living_creatures, key=lambda x: x.get_energy()):
#             side_info.append(creature.name + ": " + str(creature.get_energy()))
#
#         t = threading.Thread(target=render_to_ascii, args=(shapes_to_render, pixels, [additional_1], side_info,
#                                                            self._render_width, self._render_height,
#                                                            self._render_left, self._render_top,
#                                                            self._environment.width, self._environment.height))
#         self._last_render_time = time.time()
#         t.start()
#
#
# def render_to_ascii(shapes_to_render, pixels, additionals, side_infos, output_width, output_height, x_init, y_init,
#                     width, height):
#     x_step_size = width / output_width
#     y_step_size = height / output_height
#     line = "#" + "-" * output_width + "#\n"
#     text = ""
#     for additional in additionals:
#         text_line = ""
#         first = True
#         for value in additional:
#             if not first:
#                 text_line += ", "
#             first = False
#             text_line += value
#         text += "|" + text_line[0:output_width] + " " * (output_width - len(text_line)) + "|\n"
#     stage = ""
#     for i in range(len(side_infos), output_height):
#         side_infos.append("")
#     for y, side_info in zip(np.arange(y_init, y_init + height, y_step_size), side_infos):
#         stage += "|"
#         for x in np.arange(x_init, x_init + width, x_step_size):
#             pixel = " "
#             point = shapes.Circle(x, y, 0)
#             for shape, pixel2 in zip(shapes_to_render, pixels):
#                 if point.collides(shape):
#                     pixel = pixel2
#                     break
#             stage += pixel
#         stage += "|" + side_info + "\n"
#     print("\n\n\n" + line + text + line + stage + line)
