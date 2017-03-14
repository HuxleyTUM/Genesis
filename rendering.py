import threading
import copy
import operator
import shapes
import numpy as np
import time
import pygame
import sys
from pygame.locals import *
import binary_tree as bt

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
        raise Exception("Not implemented!")

    @property
    def bounding_rectangle(self):
        raise Exception("Not implemented!")


class TextGraphic(Graphic):
    def __init__(self, text, font, position):
        super().__init__()
        (width, height) = font.size(text)
        self.__bounding_rectangle = shapes.Rectangle(position[0], position[1], width, height)
        self.text = text
        self.label = font.render(text, 1, (255, 255, 255))
        # screen.blit(label, (100, 100))

    @property
    def bounding_box(self):
        return (self.__bounding_rectangle.left, self.__bounding_rectangle.down,
                self.__bounding_rectangle.width, self.__bounding_rectangle.height)

    @property
    def bounding_rectangle(self):
        return self.__bounding_rectangle


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
        raise Exception("Not implemented!")

    @shape.setter
    def shape(self, shape):
        raise Exception("Not implemented!")


class OutlineGraphic(ShapedGraphic):
    def __init__(self):
        super().__init__()

    @property
    def border_colour(self):
        raise Exception("Not implemented!")

    @border_colour.setter
    def border_colour(self, value):
        raise Exception("Not implemented!")

    @property
    def border_width(self):
        raise Exception("Not implemented!")

    @border_width.setter
    def border_width(self, value):
        raise Exception("Not implemented!")

    @property
    def shape(self):
        raise Exception("Not implemented!")

    @shape.setter
    def shape(self, shape):
        raise Exception("Not implemented!")


class SimpleOutlineGraphic(OutlineGraphic):
    def __init__(self, shape, border_colour, border_width=1):
        super().__init__()
        self._border_width = border_width
        self._border_colour = border_colour
        self._shape = shape

    @property
    def border_colour(self):
        return self._border_colour

    @property
    def border_width(self):
        return self._border_width

    @border_width.setter
    def border_width(self, value):
        self._border_width = value

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape


class MonoColouredGraphic(ShapedGraphic):
    def __init__(self):
        super().__init__()

    @property
    def fill_colour(self):
        raise Exception("Not implemented!")

    @fill_colour.setter
    def fill_colour(self, fill_colour):
        raise Exception("Not implemented!")

    @property
    def shape(self):
        raise Exception("Not implemented!")

    @shape.setter
    def shape(self, shape):
        raise Exception("Not implemented!")


class SimpleMonoColouredGraphic(MonoColouredGraphic):
    def __init__(self, shape, fill_colour):
        super().__init__()
        self._fill_colour = fill_colour
        self._shape = shape

    @property
    def fill_colour(self):
        return self._fill_colour

    @fill_colour.setter
    def fill_colour(self, fill_colour):
        self._fill_colour = fill_colour

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape


def listen_for_key(self):
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
    def transform_point_to_parent(self, point):
        raise Exception("Not implemented!")

    def transform_point_from_parent(self, point):
        raise Exception("Not implemented!")

    def transform_x_to_parent(self, point):
        raise Exception("Not implemented!")

    def transform_y_to_parent(self, point):
        raise Exception("Not implemented!")

    def transform_shape_to_parent(self, shape):
        raise Exception("Not implemented!")


class RelativeCamera(Camera):
    def __init__(self, position=(0, 0), zoom=(1, 1)):
        self.zoom = zoom
        self.position = position

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


class Button:
    def __init__(self, graphics, click_area):
        self.click_area = click_area
        self.graphics = graphics
        self.listeners = []

    def notify_listeners(self, point):
        for listener in self.listeners:
            listener()


class Canvas:
    def __init__(self, dimensions, back_ground_colour=(0, 0, 0)):
        self.back_ground_colour = back_ground_colour
        self.dimensions = dimensions
        self.graphic_infos_listed = []
        self.__graphic_infos_mapped = {}
        self._graphics_changed_since_last_render = {}
        self._graphics_listeners = {}
        self.graphics_registered_listeners = []
        self.graphics_un_registered_listeners = []
        self.__canvases = []
        self.clicked_listeners = []
        self.buttons = []

    def clicked(self, point):
        for listener in self.clicked_listeners:
            listener(point)

    def register_button(self, button):
        self.register_graphics(button.graphics)

        def check_click(point):
            if button.click_area.point_lies_within(point):
                button.notify_listeners(point)
        self.clicked_listeners.append(check_click)

    @property
    def local_bounding(self):
        return shapes.Rectangle(0, 0, self.dimensions[0], self.dimensions[1])

    def transform_point_from_screen(self, point):
        if self.parent_canvas is None:
            return point
        else:
            point_from_parent = self.parent_canvas.transform_point_from_screen(point)
            point_in_parent = (point_from_parent[0] - self.position_in_parent[0],
                               point_from_parent[1] - self.position_in_parent[1])
            point_in_parent = self.camera.transform_point_from_parent(point_in_parent)
            return point_in_parent

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

    @property
    def camera(self):
        return None

    def paint_text(self, label, canvas_rectangle, transform_shape=False):
        raise Exception("Not implemented!")

    def paint_shape(self, shape, colour, border_width, transform_shape=False):
        raise Exception("Not implemented!")

    def add_canvas(self, canvas, position=(0, 0)):
        canvas.parent_canvas = self
        canvas.position_in_parent = position
        self.__canvases.append(canvas)

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

    @property
    def monospaced_font(self):
        raise Exception("Not implemented!")


class SimpleCanvas(Canvas):
    def __init__(self, dimensions, camera, border_thickness=0, border_colour=None, back_ground_colour=None):
        super().__init__(dimensions, back_ground_colour)
        self.border_colour = border_colour
        self.border_thickness = border_thickness
        self.__parent_canvas = None
        self.__position_in_parent = None
        self.__camera = camera
        self.__canvas_area = shapes.Rectangle(0, 0, dimensions[0], dimensions[1])
        if back_ground_colour is not None:
            self.back_ground_graphics = SimpleMonoColouredGraphic(self.__canvas_area, back_ground_colour)
            self.register_graphic(self.back_ground_graphics)
        if border_colour is not None and border_thickness > 0:
            self.border_graphics = SimpleOutlineGraphic(self.__canvas_area, border_colour, border_thickness)
            self.register_graphic(self.border_graphics)

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

    def paint_shape(self, shape, colour, border_width, transform_shape=False):
        if not transform_shape:
            shape = copy.copy(shape)
        self.camera.transform_shape_to_parent(shape)
        if shape.left < self.__canvas_area.right and \
                shape.right > self.__canvas_area.left and \
                shape.down < self.__canvas_area.up and \
                shape.up > self.__canvas_area.down:
            shape.translate(self.position_in_parent)
            return self.parent_canvas.paint_shape(shape, colour, border_width, True)
        else:
            return None

    @property
    def monospaced_font(self):
        return self.parent_canvas.monospaced_font

    def arial_font(self, size):
        return self.parent_canvas.arial_font(size)


class Screen(Canvas):
    def __init__(self, dimensions):
        super().__init__(dimensions)
        pygame.init()
        self._monospaced_font = pygame.font.SysFont("monospaced", 10)
        self._arial_fonts = {}
        self.py_screen = pygame.display.set_mode(dimensions)
        self.dimensions = dimensions
        self._static_camera = Camera()

    def paint_text(self, label, bounding_rectangle, transform_shape=False):
        self.py_screen.blit(label, (bounding_rectangle.left, bounding_rectangle.down))
        return bounding_rectangle.to_bounding_box()

    def paint_shape(self, shape, colour, border_width, transform_shape=False):
        bounding_box = shape.to_int_bounding_box()
        # pygame.draw.ellipse(screen, colour, (50, 50, 0, 20), 1)
        max_border_width = max(min(bounding_box[2], bounding_box[3]) - 1, 0)
        # try:
        border_width = min(border_width, max_border_width)
        # except TypeError:
        #     print("type: "+str(type(shape)))
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
        else:
            raise "Unknown shape: " + str(type(shape))
        return bounding_box

    @property
    def camera(self):
        return self._static_camera

    @property
    def monospaced_font(self):
        return self._monospaced_font

    def arial_font(self, size):
        if size not in self._arial_fonts:
            self._arial_fonts[size] = pygame.font.SysFont("arial", size)
        return self._arial_fonts[size]


class Frame:
    def __init__(self, frame_counter, canvases, rects_to_update):
        self.canvases = canvases
        self.rects_to_update = rects_to_update
        self.frame_counter = frame_counter
        self.rects_rendered = None


class GraphicsSource:
    @property
    def graphics(self):
        raise Exception("Not implemented!")


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
        # self._last_rendered_rects = []
        self._current_frame_counter = 0
        self._last_frame = None
        self.rects_to_update = []

    def un_registered_graphic(self, graphic_info):
        self.rects_to_update.append(graphic_info.last_rect_rendered)

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
        additional = ["width(w): " + str(self.screen.dimensions[0]), "height(h): " + str(self.screen.dimensions[1]),
                      # , "ticks: " + str(self._environment.tick_count)
                      "frames/s: " + str(1 / (time.time() - self._last_render_time))
                      # , "physics/s: " + str(1 / max(self._environment.last_tick_delta, 0.00001))
                      ]
        self._last_render_time = time.time()
        side_info = []
        # for creature in sorted(self._environment.living_creatures, key=lambda x: x.mass):
        #     side_info.append(creature.name + ": " + str(creature.mass))

        if self._last_frame is None:
            rects_previously_rendered = []
        else:
            rects_previously_rendered = self._last_frame.rects_rendered

        frame = Frame(self._current_frame_counter, self.screen.canvases, self.rects_to_update)
        self.rects_to_update = []
        self._last_frame = frame
        self._current_frame_counter += 1
        # t = threading.Thread(target=render_with_pygame, args=(frame, self))
        # t.start()
        render_with_pygame(frame, self)
        if render_clock is not None:
            render_clock.tock()


def render_with_pygame(frame, renderer):
    render_lock.acquire()
    clock = renderer.thread_render_clock
    if clock is not None:
        clock.tick()

    renderer.screen.py_screen.fill((0, 0, 0))
    dirty_rectangles = frame.rects_to_update
    # for previous_rendered_rect in frame.rects_previously_rendered:
    #     pygame.draw.rect(frame.camera.screen, (255, 0, 0), previous_rendered_rect, 1)
    for canvas in frame.canvases:
        # pygame.draw.rect(canvas.screen.py_screen, canvas.back_ground_colour, canvas.camera.screen_bounding, 0)
        # if canvas.border_thickness > 0:
        #     pygame.draw.rect(canvas.screen.py_screen, canvas.border_colour,
        #                      canvas.camera.screen_bounding, canvas.border_thickness)
        
        for graphic_info in list(canvas.graphic_infos):
            graphic = graphic_info.graphic
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
                drawn_bounding = canvas.paint_text(graphic.label, graphic.bounding_rectangle)

            if drawn_bounding is not None and graphic_info.changed_since_render:
                if graphic_info.last_rect_rendered is not None:
                    # pass
                    dirty_rectangles.append(graphic_info.last_rect_rendered)
                    # pygame.draw.rect(frame.camera.screen, (255, 255, 255, 0), graphic_info.last_rect_rendered, 1)

                dirty_rectangles.append(drawn_bounding)
                # pygame.draw.rect(frame.camera.screen, (255, 0, 255, 0), drawn_bounding, 1)

                graphic_info.last_rect_rendered = drawn_bounding
                graphic_info.changed_since_render = False


    # frame.rects_rendered = dirty_rectangles
    # pygame.draw.circle()
    # i = 0
    # for additional in additionals[3:4]:
    #     font = pygame.font.Font(None, 36)
    #     text = font.render(additional, 0, (255, 255, 255))
    #     #textpos = text.get_rect()
    #     # textpos. = ().centerx
    #     screen.blit(text, (0, 36/2+26*i))
    #     i += 1
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
#         print("+")
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
