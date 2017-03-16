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


render_lock = threading.Lock()
graphics_counter = 0


class Graphic:
    def __init__(self):
        self.has_changed_listeners = []

    def add_has_changed_listener(self, listener):
        self.has_changed_listeners.append(listener)

    def remove_has_changed_listener(self, listener):
        self.has_changed_listeners.remove(listener)

    def notify_listeners_of_change(self, *args):
        for listener in self.has_changed_listeners:
            listener()

    @property
    def bounding_box(self):
        raise Exception("Not implemented in "+str(type(self)))

    @property
    def bounding_rectangle(self):
        raise Exception("Not implemented in "+str(type(self)))

    def translate(self, delta):
        raise Exception("Not implemented in "+str(type(self)))

    def scale(self, scalar):
        raise Exception("Not implemented in "+str(type(self)))

    @property
    def is_visible(self):
        raise Exception("Not implemented in "+str(type(self)))

    @is_visible.setter
    def is_visible(self, value):
        raise Exception("Not implemented in "+str(type(self)))


class TextGraphic(Graphic):
    def __init__(self, text, font, text_colour=(255, 255, 255, 0), is_visible=True):
        super().__init__()
        self.__text_colour = text_colour
        self.font = font
        self.pyg_label = None
        self.__text = None
        self.__bounding_rectangle = shapes.Rectangle(0, 0, 1, 1)
        self.text = text
        self.__is_visible = is_visible

    @property
    def text_colour(self):
        return self.__text_colour

    @text_colour.setter
    def text_colour(self, text_colour):
        self.__text_colour = text_colour
        self.notify_listeners_of_change()

    @property
    def position(self):
        return self.__bounding_rectangle.left, self.__bounding_rectangle.down

    @position.setter
    def position(self, position):
        self.__bounding_rectangle.left = position[0]
        self.__bounding_rectangle.down = position[1]

    def translate(self, delta):
        self.position = [x+dx for x, dx in zip(self.position, delta)]
        self.notify_listeners_of_change()

    @property
    def bounding_box(self):
        return (self.__bounding_rectangle.left, self.__bounding_rectangle.down,
                self.__bounding_rectangle.width, self.__bounding_rectangle.height)

    @property
    def bounding_rectangle(self):
        return self.__bounding_rectangle

    @property
    def text(self):
        return self.__text

    @text.setter
    def text(self, text):
        self.__text = text
        self.__bounding_rectangle.dimensions = self.font.size(text)
        self.pyg_label = self.font.render(self.__text, 1, self.__text_colour)
        self.notify_listeners_of_change()

    @property
    def is_visible(self):
        return self.__is_visible

    @is_visible.setter
    def is_visible(self, is_visible):
        if is_visible is not self.__is_visible:
            self.__is_visible = is_visible
            self.notify_listeners_of_change()



class ShapedGraphic(Graphic):
    def __init__(self):
        super().__init__()

    @property
    def bounding_box(self):
        return self.shape.to_bounding_box()

    @property
    def bounding_rectangle(self):
        return self.shape.to_bounding_rectangle()

    @property
    def shape(self):
        raise Exception("Not implemented in "+str(type(self)))

    @shape.setter
    def shape(self, shape):
        raise Exception("Not implemented in "+str(type(self)))

    def translate(self, delta):
        self.shape.translate(delta)
        self.notify_listeners_of_change()

    @property
    def is_visible(self):
        raise Exception("Not implemented in "+str(type(self)))

    @is_visible.setter
    def is_visible(self, value):
        raise Exception("Not implemented in "+str(type(self)))


class OutlineGraphic(ShapedGraphic):
    def __init__(self):
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
        self.notify_listeners_of_change()

    @property
    def border_width(self):
        return self._border_width

    @border_width.setter
    def border_width(self, border_width):
        self._border_width = border_width
        self.notify_listeners_of_change()

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self.notify_listeners_of_change()

    @property
    def is_visible(self):
        return self.__is_visible

    @is_visible.setter
    def is_visible(self, is_visible):
        if is_visible is not self.__is_visible:
            self.__is_visible = is_visible
            self.notify_listeners_of_change()


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
        self.notify_listeners_of_change()

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self.notify_listeners_of_change()

    @property
    def is_visible(self):
        return self.__is_visible

    @is_visible.setter
    def is_visible(self, is_visible):
        if is_visible is not self.__is_visible:
            self.__is_visible = is_visible
            self.notify_listeners_of_change()


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

    def transform_shape_to_parent(self, shape):
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
        return self.transform_x_to_parent(point), self.transform_y_to_parent(point)

    def transform_point_from_parent(self, point):
        return point[0] / self.zoom[0] + self.position[0], point[1] / self.zoom[1] + self.position[1]

    def transform_x_to_parent(self, point):
        return (point[0] - self.position[0]) * self.zoom[0]

    def transform_y_to_parent(self, point):
        return (point[1] - self.position[1]) * self.zoom[1]

    def transform_shape_to_parent(self, shape):
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

    def transform_shape_to_parent(self, shape):
        left_down = self.transform_point_to_parent((shape.left, shape.down))
        shape.translate((left_down[0] - shape.left, left_down[1] - shape.down))
        shape.scale(self.scaling)
        return shape


class Canvas:
    def __init__(self, local_canvas_area, back_ground_colour=(0, 0, 0)):
        self.canvases_to_remove = []
        self.redraw_whole_screen = False
        self.__redraw_all_graphics = False
        self.__back_ground_colour = back_ground_colour
        self.local_canvas_area = local_canvas_area
        self.graphic_infos_listed = []
        self.__graphic_infos_mapped = {}
        self._graphics_changed_since_last_render = {}
        self._graphics_listeners = {}
        self.graphics_registered_listeners = []
        self.graphics_un_registered_listeners = []
        self.__canvases = []
        self.mouse_pressed_event_listeners = []
        self.mouse_released_event_listeners = []
        self.mouse_canceled_event_listeners = []
        self.mouse_wheel_up_event_listeners = []
        self.mouse_wheel_down_event_listeners = []
        self.pressed_key_left_event_listeners = []
        self.pressed_key_right_event_listeners = []
        self.pressed_key_up_event_listeners = []
        self.pressed_key_down_event_listeners = []

        # event_condition
        # self.buttons = []

    @property
    def global_canvas_area(self):
        return self.transform_shape_to_screen(self.local_canvas_area)

    @property
    def back_ground_colour(self):
        return self.__back_ground_colour

    @back_ground_colour.setter
    def back_ground_colour(self, value):
        self.__back_ground_colour = value

    @property
    def redraw_all_graphics(self):
        return self.__redraw_all_graphics

    @redraw_all_graphics.setter
    def redraw_all_graphics(self, value):
        self.__redraw_all_graphics = value
        if value:
            for canvas in self.canvases:
                canvas.redraw_all_graphics = value

    def mouse_wheel_scrolled_up(self, event):
        events.fire_listeners(self.mouse_wheel_up_event_listeners, event)

    def mouse_wheel_scrolled_down(self, event):
        events.fire_listeners(self.mouse_wheel_down_event_listeners, event)

    def scroll_up(self):
        self.move_camera((0, -10))

    def scroll_down(self):
        self.move_camera((0, 10))

    def scroll_left(self):
        self.move_camera((-10, 0))

    def scroll_right(self):
        self.move_camera((10, 0))

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

    def move_camera(self, delta):
        if self.camera is not None:
            self.camera.pan(delta)
            self.redraw_all_graphics = True

    def register_and_center_graphic(self, graphic):
        graphic_rectangle = graphic.bounding_rectangle
        clicking_center = self.local_canvas_area.center
        graphic_center = graphic_rectangle.center
        graphic.translate((clicking_center[0] - graphic_center[0], clicking_center[1] - graphic_center[1]))
        self.register_graphic(graphic)

    @property
    def local_bounding_rectangle(self):
        return shapes.Rectangle(0, 0, self.local_canvas_area.left, self.local_canvas_area.down)

    def transform_point_from_screen(self, point):
        if self.parent_canvas is None:
            return point
        else:
            point_from_parent = self.parent_canvas.transform_point_from_screen(point)
            point_in_parent = (point_from_parent[0] - self.position_in_parent[0],
                               point_from_parent[1] - self.position_in_parent[1])
            local_point = self.camera.transform_point_from_parent(point_in_parent)
            return local_point

    # def transform_point_to_screen

    @property
    def parent_canvas(self):
        return None

    @parent_canvas.setter
    def parent_canvas(self, parent_canvas):
        raise Exception("Can't set parent canvas!")

    @property
    def position_in_parent(self):
        return None

    @position_in_parent.setter
    def position_in_parent(self, position_in_parent):
        raise Exception("Can't set position!")

    def translate(self, delta):
        self.position_in_parent = [a+b for a, b in zip(self.position_in_parent, delta)]

    @property
    def camera(self):
        return None

    def transform_shape_to_screen(self, shape, transform_shape=False):
        raise Exception("Not implemented in "+str(type(self)))

    def transform_shape_to_parent(self, shape, transform_shape=False):
        raise Exception("Not implemented in "+str(type(self)))

    def paint_text(self, label, canvas_rectangle, transform_shape=False):
        raise Exception("Not implemented in "+str(type(self)))

    def paint_shape(self, shape, colour, border_width, transform_shape=False):
        raise Exception("Not implemented in "+str(type(self)))

    def add_canvas(self, canvas, position=(0, 0)):
        canvas.parent_canvas = self
        canvas.position_in_parent = position
        self.__canvases.append(canvas)

    def queue_canvas_for_removal(self, canvas):
        self.canvases_to_remove.append(canvas)

    def _remove_canvas(self, canvas):
        canvas.parent_canvas = None
        canvas.position_in_parent = None
        self.__canvases.remove(canvas)
        self.redraw_whole_screen = True

    def register_graphics(self, graphics):
        for graphic in graphics:
            self.register_graphic(graphic)

    def register_graphic(self, graphic):
        graphic_info = GraphicInfo(graphic, len(self.__graphic_infos_mapped))

        def graphic_changed():
            graphic_info.changed_since_render = True

        graphic_info.graphics_changed_listener = graphic_changed
        self.__graphic_infos_mapped[graphic] = graphic_info
        graphic.add_has_changed_listener(graphic_info.graphics_changed_listener)
        graphic_changed()
        self.graphic_infos_listed.append(graphic_info)
        for listener in self.graphics_registered_listeners:
            listener(graphic_info)

    def un_register_graphics(self, graphics):
        for graphic in graphics:
            self.un_register_graphic(graphic)

    def un_register_graphic(self, graphic):
        graphic_info = self.__graphic_infos_mapped[graphic]
        graphic.remove_has_changed_listener(graphic_info.graphics_changed_listener)
        del self.__graphic_infos_mapped[graphic]
        self.graphic_infos_listed.remove(graphic_info)
        for listener in self.graphics_un_registered_listeners:
            listener(graphic_info)

    @property
    def graphic_infos(self):
        return self.graphic_infos_listed

    @property
    def graphics(self):
        to_return = []
        for graphic_info in self.graphic_infos_listed:
            to_return.append(graphic_info.graphic)
        return to_return

    @property
    def canvases(self):
        return self.__canvases


class SimpleCanvas(Canvas):
    def __init__(self, local_canvas_area, camera=RelativeCamera(), back_ground_area=None,
                 border_width=1, border_colour=None, back_ground_colour=None):
        super().__init__(local_canvas_area, back_ground_colour)
        self.__border_colour = border_colour
        self.__border_width = border_width
        self.__parent_canvas = None
        self.__position_in_parent = None
        self.__camera = copy.copy(camera)
        self.__back_ground_colour = back_ground_colour
        self.__back_ground_graphic = None
        self.__border_graphic = None

        if back_ground_area is None:
            self.__back_ground_area = local_canvas_area
        else:
            self.__back_ground_area = back_ground_area
        self.back_ground_colour = back_ground_colour
        self.border_colour = border_colour
        self.border_width = border_width
        # if back_ground_colour is not None:
        #     self.back_ground_graphic = SimpleMonoColouredGraphic(self.__back_ground_area, back_ground_colour)
        #     self.register_graphic(self.back_ground_graphic)
        # else:
            
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
                self.register_graphic(self.__border_graphic)
            else:
                self.__border_graphic.border_colour = border_colour
                self.__border_graphic.border_width = border_width
        else:
            if self.__border_graphic is not None:
                self.un_register_graphic(self.__border_graphic)
                self.__border_graphic = None
        self.__border_colour = border_colour
        self.__border_width = border_width

    @property
    def back_ground_colour(self):
        return self.__back_ground_colour

    @back_ground_colour.setter
    def back_ground_colour(self, back_ground_colour):
        self.__back_ground_colour = back_ground_colour
        if back_ground_colour is not None:
            if self.__back_ground_graphic is None:
                self.__back_ground_graphic = SimpleMonoColouredGraphic(self.__back_ground_area, back_ground_colour)
                self.register_graphic(self.__back_ground_graphic)
            else:
                self.__back_ground_graphic.fill_colour = back_ground_colour
        else:
            if self.__back_ground_graphic is not None:
                self.un_register_graphic(self.__back_ground_graphic)
                self.__back_ground_graphic = None

    @property
    def parent_canvas(self):
        return self.__parent_canvas

    @parent_canvas.setter
    def parent_canvas(self, parent_canvas):
        self.__parent_canvas = parent_canvas

    @property
    def position_in_parent(self):
        return self.__position_in_parent

    @position_in_parent.setter
    def position_in_parent(self, position_in_parent):
        self.__position_in_parent = position_in_parent

    @property
    def camera(self):
        return self.__camera

    def paint_text(self, label, label_bounding, transform_shape=False):
        if not transform_shape:
            label_bounding = copy.copy(label_bounding)
        new_left_bottom = self.camera.transform_point_to_parent((label_bounding.left, label_bounding.down))
        label_bounding.translate((new_left_bottom[0] - label_bounding.left, new_left_bottom[1] - label_bounding.down))
        label_bounding.translate(self.position_in_parent)
        return self.parent_canvas.paint_text(label, label_bounding, True)

    def transform_shape_to_screen(self, shape, transform_shape=False):
        shape = self.transform_shape_to_parent(shape, transform_shape)
        return self.parent_canvas.transform_shape_to_screen(shape, True)

    def transform_shape_to_parent(self, shape, transform_shape=False):
        if not transform_shape:
            shape = copy.copy(shape)
        self.camera.transform_shape_to_parent(shape)
        shape.translate(self.position_in_parent)
        return shape

    def paint_shape(self, shape, colour, border_width, transform_shape=False):
        if not transform_shape:
            shape = copy.copy(shape)
        if shape.left < self.local_canvas_area.right and \
                shape.right > self.local_canvas_area.left and \
                shape.down < self.local_canvas_area.up and \
                shape.up > self.local_canvas_area.down:
            self.camera.transform_shape_to_parent(shape)
            shape.translate(self.position_in_parent)
            return self.parent_canvas.paint_shape(shape, colour, border_width, True)
        else:
            return None


class Table(SimpleCanvas):
    def __init__(self, font_size, text_colour, column_count, padding=5):
        super().__init__(shapes.rect((1, 1)))
        self.padding = padding
        self.column_count = column_count
        self.text_colour = text_colour
        self.font_size = font_size
        self.row_graphics = []
        self.row_width = [0] * column_count

    def add_rows(self, rows):
        for row in rows:
            self.add_row(row)

    def add_row(self, row_text):
        row = []
        for column_index in range(self.column_count):
            text_graphic = TextGraphic(str(row_text[column_index]), Fonts.arial_font(self.font_size), self.text_colour)
            x = 0 if column_index == 0 else sum(self.row_width[0:column_index])
            row_height = text_graphic.bounding_rectangle.height
            y = len(self.row_graphics) * row_height
            self.local_canvas_area.height = max(self.local_canvas_area.height, y + row_height)
            text_graphic.translate((x, y))
            row.append(text_graphic)
            self.register_graphic(text_graphic)
            self.__check_column_width(column_index, text_graphic.bounding_rectangle.width + self.padding)
        self.row_graphics.append(row)

    def __check_column_width(self, column_index, width):
        dx = width - self.row_width[column_index]
        if dx > 0:
            self.local_canvas_area.width += dx
            for row_to_fix in self.row_graphics:
                for graphic in row_to_fix[column_index + 1:]:
                    graphic.translate((dx, 0))
            self.row_width[column_index] = width

    def set(self, x, y, text):
        text_graphic = self.row_graphics[y][x]
        text_graphic.text = str(text)
        self.__check_column_width(x, text_graphic.bounding_rectangle.width + self.padding)

    @property
    def row_count(self):
        return len(self.row_graphics)


class ScrollingPane(SimpleCanvas):
    def __init__(self, local_canvas_area, scroll_vertically, scroll_horizontally, camera=RelativeCamera()):
        super().__init__(local_canvas_area, camera)
        self.pane = SimpleCanvas(copy.copy(local_canvas_area))
        self.add_canvas(self.pane)
        self.scroll_horizontally = scroll_horizontally
        self.scroll_vertically = scroll_vertically

        pane = self.pane
        def scroll_down_function(x):
            if self.scroll_vertically and pane.camera.position[1] > 0:
                pane.scroll_up()
        def scroll_up_function(x):
            pane_bottom = pane.local_canvas_area.height - pane.camera.position[1]
            if self.scroll_vertically and pane_bottom > self.local_canvas_area.height:
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


class Button(SimpleCanvas):
    def __init__(self, button_area, border_width=1, back_ground_colour=(200, 200, 200, 0), border_colour=None,
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
        print("canceled!")
        if self.is_pressed:
            self.is_pressed = False
            self._adjust_colour()
            event.consume()


class RectangularButton(Button):
    def __init__(self, dimensions):
        super().__init__(shapes.rect(dimensions))


class TextButton(RectangularButton):
    def __init__(self, text, font_size=12):
        super().__init__((1, 1))
        self.label = TextGraphic(text, Fonts.arial_font(font_size), (0, 0, 0, 0))
        self.register_graphic(self.label)
        self.local_canvas_area.dimensions = self.label.bounding_rectangle.dimensions


class ButtonBar(SimpleCanvas):
    def __init__(self, dimensions, padding=5):
        super().__init__(shapes.rect(dimensions), border_width=1, border_colour=(255, 255, 255, 0),
                         back_ground_colour=(0, 0, 0, 0))
        self.padding = padding
        self.last_x = 0
        self.button_height = dimensions[1] - self.padding * 2

    def add_button(self, button):
        self.add_canvas(button, (self.last_x + self.padding, self.padding))
        scaling = self.button_height / button.local_canvas_area.height
        # button.camera.zoom_by_factor((scaling, scaling))
        self.last_x += button.local_canvas_area.width*scaling + self.padding


class Fonts:
    _monospaced_fonts = {}
    _arial_fonts = {}

    @staticmethod
    def monospaced_font(size):
        if size not in Fonts._arial_fonts:
            Fonts._monospaced_fonts[size] = pygame.font.SysFont("monospaced", size)
        return Fonts._monospaced_fonts[size]

    @staticmethod
    def arial_font(size):
        if size not in Fonts._arial_fonts:
            Fonts._arial_fonts[size] = pygame.font.SysFont("arial", size)
        return Fonts._arial_fonts[size]


class Screen(Canvas):
    def __init__(self, dimensions):
        super().__init__(shapes.rect(dimensions))
        pygame.init()
        # self._monospaced_fonts = {}
        # self._arial_fonts = {}
        self.py_screen = pygame.display.set_mode(dimensions)
        self.dimensions = dimensions
        # self._static_camera = RelativeCamera()

    def transform_shape_to_screen(self, shape, transform_shape=False):
        return shape

    def transform_shape_to_parent(self, shape, transform_shape=False):
        return shape

    def paint_text(self, label, bounding_rectangle, transform_shape=False):
        self.py_screen.blit(label, (bounding_rectangle.left, bounding_rectangle.down))
        return bounding_rectangle.to_bounding_box()

    def paint_shape(self, shape, colour, border_width, transform_shape=False):
        bounding_box = shape.to_int_bounding_box()
        max_border_width = max(min(bounding_box[2], bounding_box[3]) - 1, 0)
        border_width = min(border_width, max_border_width)
        if type(shape) is shapes.Circle:
            pygame.draw.ellipse(self.py_screen, colour, bounding_box, border_width)
        elif type(shape) is shapes.Axis:
            axis = shape
            screen_pos_from = axis.center
            screen_pos_to = copy.copy(screen_pos_from)
            screen_pos_from[axis.dimension] = 0
            screen_pos_to[axis.dimension] = self.dimensions[axis.dimension]
            pygame.draw.line(self.py_screen, colour, screen_pos_from, screen_pos_to, max(border_width, 1))
        elif type(shape) is shapes.LineSegment:
            line_segment = shape
            pygame.draw.line(self.py_screen, colour, line_segment.start_point, line_segment.end_point)
        elif type(shape) is shapes.Rectangle:
            pygame.draw.rect(self.py_screen, colour, bounding_box, border_width)
        elif type(shape) is shapes.Polygon:
            pygame.draw.polygon(self.py_screen, colour, shape.points, border_width)
        else:
            raise "Unknown shape: " + str(type(shape))
        # pygame.draw.rect(self.py_screen, (255, 0, 0, 0), bounding_box, 1)
        return shape.to_generous_int_bounding_box()

    @property
    def camera(self):
        return None

    # def monospaced_font(self, size):
    #     if size not in self._arial_fonts:
    #         self._monospaced_fonts[size] = pygame.font.SysFont("monospaced", size)
    #     return self._monospaced_fonts[size]
    #
    # def arial_font(self, size):
    #     if size not in self._arial_fonts:
    #         self._arial_fonts[size] = pygame.font.SysFont("arial", size)
    #     return self._arial_fonts[size]


class Frame:
    def __init__(self, frame_counter, canvas, rects_to_update):
        self.canvas = canvas
        self.rects_to_update = rects_to_update
        self.frame_counter = frame_counter
        self.rects_rendered = None


class GraphicsSource:
    @property
    def graphics(self):
        raise Exception("Not implemented in "+str(type(self)))


# rects_previously_rendered = []


class GraphicInfo:
    def __init__(self, graphic, graphics_id):
        self.graphics_id = graphics_id
        self.graphic = graphic
        self.changed_since_render = True
        self.graphics_changed_listener = None
        self.last_rect_rendered = None


class Renderer:
    def __init__(self):
        pass

    def render(self):
        pass


class PyGameRenderer(Renderer):
    #        clock = self._environment.time[gen.RENDER_KEY]
    def __init__(self, screen, render_clock=None, thread_render_clock=None):
        super().__init__()
        self.screen = screen
        for canvas in self.screen.canvases:
            canvas.graphics_un_registered_listeners.append(self.un_registered_graphic)
        self.thread_render_clock = thread_render_clock
        self.render_clock = render_clock
        self._last_render_time = -1
        self._current_frame_counter = 0
        self._last_frame = None
        self.screen_boxes_to_update = []
        self.__visualise_boundings = False

    def visualise_boundings(self, *args):
        self.__visualise_boundings = not self.__visualise_boundings
        self.redraw_canvas(self.screen)

    def un_registered_graphic(self, graphic_info):
        self.screen_boxes_to_update.append(graphic_info.last_rect_rendered)

    def redraw_canvas(self, canvas):
        self.screen_boxes_to_update.append(canvas.global_canvas_area.to_bounding_box())

    def render(self):
        if self.render_clock is not None:
            render_clock = self.render_clock
            render_clock.tick()
        else:
            render_clock = None

        # ## #
        # G the following is a visualisation of the binary tree. uncomment to see how the food pellets are classified
        # ###
        # root_axis = self._environment.food_tree.root_node.axis
        # shapes_to_render.append(root_axis)
        # pixels.append((255, 0, 0))
        # collision_circle = shapes.Circle(self._environment.tick_count, self._environment.tick_count, 6)
        # shapes_to_render.append(collision_circle)
        # pixels.append((0, 255, 255))
        # for food_pellet in self._environment.food_tree.get_collision_candidates(collision_circle):
        #     shapes_to_render.append(food_pellet.shape)
        #     pixels.append((255, 0, 255))



        # for creature in self._environment.living_creatures:
        #     for organ in creature.organs:
        #         shape = organ.shape
        #         if organ is not creature.body and shape is not None:
        #             shapes_to_render.append(copy.deepcopy(shape))
        #             pixels.append((0, 255, 255))
        #     shapes_to_render.append(copy.deepcopy(creature.body.shape))
        #     pixels.append((0, 0, 255 - 200 * (creature.age / gen.MAX_AGE)))
        # for food in self._environment.food_tree.elements:
        #     shapes_to_render.append(copy.deepcopy(food.shape))
        #     pixels.append((0, 255, 0))
        # rects_to_render = []
        # for shape in shapes_to_render:
        #     rects_to_render.append(shape.to_bounding_box())
        # additional = ["width(w): " + str(self.screen.dimensions[0]), "height(h): " + str(self.screen.dimensions[1]),
        #               # , "ticks: " + str(self._environment.tick_count)
        #               "frames/s: " + str(1 / (time.time() - self._last_render_time))
        #               # , "physics/s: " + str(1 / max(self._environment.last_tick_delta, 0.00001))
        #               ]
        self._last_render_time = time.time()
        # side_info = []
        # for creature in sorted(self._environment.living_creatures, key=lambda x: x.mass):
        #     side_info.append(creature.name + ": " + str(creature.mass))

        # if self._last_frame is None:
        #     rects_previously_rendered = []
        # else:
        #     rects_previously_rendered = self._last_frame.rects_rendered

        frame = Frame(self._current_frame_counter, self.screen, self.screen_boxes_to_update)
        self.screen_boxes_to_update = []
        self._last_frame = frame
        self._current_frame_counter += 1
        # t = threading.Thread(target=render_with_pygame, args=(frame, self))
        # t.start()
        self.render_with_pygame(frame)
        if render_clock is not None:
            render_clock.tock()

    def render_canvas(self, canvas, dirty_rectangles):
        redrawing_whole_canvas = canvas.redraw_whole_screen
        if redrawing_whole_canvas:
            dirty_rectangles.append(canvas.global_canvas_area.to_bounding_box())
            canvas.redraw_whole_screen = False
        for graphic_info in list(canvas.graphic_infos_listed):
            graphic = graphic_info.graphic
            if graphic.is_visible:
                if not redrawing_whole_canvas and canvas.redraw_all_graphics:
                    graphic_info.changed_since_render = True
                if isinstance(graphic, ShapedGraphic):
                    shape = graphic.shape
                    if isinstance(graphic, OutlineGraphic):
                        colour = graphic.border_colour
                        border_width = graphic.border_width
                    elif isinstance(graphic, MonoColouredGraphic):
                        colour = graphic.fill_colour
                        border_width = 0
                    else:
                        continue
                    drawn_bounding = canvas.paint_shape(shape, colour, border_width)

                elif isinstance(graphic, TextGraphic):
                    drawn_bounding = canvas.paint_text(graphic.pyg_label, graphic.bounding_rectangle)
                if drawn_bounding is not None:
                    if self.__visualise_boundings:
                        # canvas.paint_shape(shape.to_bounding_rectangle(), (255, 0, 0, 0), 1)
                        pygame.draw.rect(self.screen.py_screen, (255, 0, 0, 0), drawn_bounding, 1)
            if graphic_info.changed_since_render:
                if not redrawing_whole_canvas:
                    if graphic_info.last_rect_rendered is not None:
                        dirty_rectangles.append(graphic_info.last_rect_rendered)
                    if drawn_bounding is not None:
                        dirty_rectangles.append(drawn_bounding)
                # pygame.draw.rect(frame.camera.screen, (255, 0, 255, 0), drawn_bounding, 1)
                if drawn_bounding is not None:
                    graphic_info.last_rect_rendered = drawn_bounding
                graphic_info.changed_since_render = False
        for sub_canvas in canvas.canvases:
            self.render_canvas(sub_canvas, dirty_rectangles)
        for canvas_to_remove in canvas.canvases_to_remove:
            canvas._remove_canvas(canvas_to_remove)
        canvas.redraw_all_graphics = False
        canvas.canvases_to_remove = []

    def render_with_pygame(self, frame):
        render_lock.acquire()
        clock = self.thread_render_clock
        if clock is not None:
            clock.tick()
        self.screen.py_screen.fill((0, 0, 0))
        # self.boxes_to_update = None
        dirty_rectangles = frame.rects_to_update
        self.render_canvas(frame.canvas, dirty_rectangles)
        pygame.display.update(dirty_rectangles)
        if clock is not None:
            clock.tock()
        render_lock.release()

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
