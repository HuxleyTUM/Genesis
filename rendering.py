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


class Graphics:
    def __init__(self):
        self.has_changed_listeners = []

    def add_has_changed_listener(self, listener):
        self.has_changed_listeners.append(listener)

    def remove_has_changed_listener(self, listener):
        self.has_changed_listeners.remove(listener)

    def notify_listeners_of_change(self, *args):
        for listener in self.has_changed_listeners:
            listener()


class ShapedGraphics(Graphics):
    def __init__(self):
        super().__init__()

    def get_bounding_box(self):
        return self.shape.to_bounding_box()

    @property
    def shape(self):
        raise Exception("Not implemented!")

    @shape.setter
    def shape(self, shape):
        raise Exception("Not implemented!")


class OutlineGraphics(ShapedGraphics):
    def __init__(self):
        super().__init__()

    @property
    def border_colour(self):
        raise Exception("Not implemented!")

    @property
    def shape(self):
        raise Exception("Not implemented!")

    @shape.setter
    def shape(self, shape):
        raise Exception("Not implemented!")


class SimpleOutlineGraphics(OutlineGraphics):
    def __init__(self, shape, border_colour):
        super().__init__()
        self._border_colour = border_colour
        self._shape = shape

    @property
    def border_colour(self):
        return self._border_colour

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape


class MonoColouredGraphics(ShapedGraphics):
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


class SimpleMonoColouredGraphics(MonoColouredGraphics):
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
    def __init__(self, screen, render_dimensions, pos, source_dimensions):
        self.source_dimensions = source_dimensions
        self.pos = pos
        self.render_dimensions = render_dimensions
        self.screen = screen


class Frame:
    def __init__(self, frame_counter, graphic_infos, additionals, side_infos, rects_to_update, camera):
        self.rects_to_update = rects_to_update
        self.frame_counter = frame_counter
        self.camera = camera
        self.side_infos = side_infos
        self.additionals = additionals
        self.graphic_infos = graphic_infos
        # self.shapes_to_render = shapes_to_render
        # self.env = env
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
        self.graphic_infos_listed = []
        self.graphic_infos = {}
        self._graphics_changed_since_last_render = {}
        self._graphics_listeners = {}

    def render(self):
        pass

    def register_graphics(self, graphics):
        for graphic in graphics:
            self.register_graphic(graphic)

    def register_graphic(self, graphic):
        graphic_info = GraphicInfo(graphic, len(self.graphic_infos))

        def graphic_changed():
            graphic_info.changed_since_render = True

        graphic_info.graphics_changed_listener = graphic_changed
        self.graphic_infos[graphic] = graphic_info
        graphic.add_has_changed_listener(graphic_info.graphics_changed_listener)
        graphic_changed()
        self.graphic_infos_listed.append(graphic_info)

    def un_register_graphics(self, graphics):
        for graphic in graphics:
            self.un_register_graphic(graphic)

    def un_register_graphic(self, graphic):
        graphic_info = self.graphic_infos[graphic]
        graphic.remove_has_changed_listener(graphic_info.graphics_changed_listener)
        del self.graphic_infos[graphic]
        self.graphic_infos_listed.remove(graphic_info)


class PyGameRenderer(Renderer):
    #        clock = self._environment.time[gen.RENDER_KEY]
    def __init__(self, source_dimensions, render_dimensions=(900, 600), render_clock=None,
                 thread_render_clock=None):
        super().__init__()
        self.thread_render_clock = thread_render_clock
        self.render_clock = render_clock
        pygame.init()
        self.screen = pygame.display.set_mode(render_dimensions)
        self._render_dimensions = render_dimensions
        self._render_pos = [0, 0]
        self._last_render_time = -1
        # self._last_rendered_rects = []
        self._current_frame_counter = 0
        self._last_frame = None
        self.camera = Camera(self.screen, self._render_dimensions, self._render_pos, source_dimensions)
        self.rects_to_update = []

    def un_register_graphic(self, graphic):
        graphic_info = self.graphic_infos[graphic]
        super(PyGameRenderer, self).un_register_graphic(graphic)
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
        additional = ["width(w): " + str(self._render_dimensions[0]), "height(h): " + str(self._render_dimensions[1])
                      # , "ticks: " + str(self._environment.tick_count)
            , "frames/s: " + str(1 / (time.time() - self._last_render_time))
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
        frame = Frame(self._current_frame_counter, self.graphic_infos_listed, additional, side_info,
                      self.rects_to_update, self.camera)
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

    scaling = np.divide(frame.camera.render_dimensions, frame.camera.source_dimensions)

    frame.camera.screen.fill((0, 0, 0))
    dirty_rectangles = frame.rects_to_update
    # for previous_rendered_rect in frame.rects_previously_rendered:
    #     pygame.draw.rect(frame.camera.screen, (255, 0, 0), previous_rendered_rect, 1)
    for graphic_info in frame.graphic_infos:
        graphic = graphic_info.graphic
        if isinstance(graphic, ShapedGraphics):
            shape = graphic.shape
            if isinstance(graphic, OutlineGraphics):
                colour = graphic.border_colour
                border_width = 1
            elif isinstance(graphic, MonoColouredGraphics):
                colour = graphic.fill_colour
                border_width = 0
            else:
                continue
            drawn_bounding = draw_shape(shape, frame.camera.screen, colour, scaling,
                                        frame.camera.render_dimensions, border_width)
            if graphic_info.changed_since_render:
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


def transform_to_bounding(shape, scaling, render_dimensions):
    top_left = (shape.left * scaling[0], shape.down * scaling[1])
    shape_dimensions = (int(round(shape.width * scaling[0])),
                        int(round(shape.height * scaling[1])))
    rect = (top_left[0], top_left[1], shape_dimensions[0], shape_dimensions[1])
    return rect


def draw_shape(shape, screen, colour, scaling, render_dimensions, border_width):
    bounding_rectangle = transform_to_bounding(shape, scaling, render_dimensions)
    # pygame.draw.ellipse(screen, colour, (50, 50, 0, 20), 1)
    max_border_width = max(min(bounding_rectangle[2], bounding_rectangle[3]) - 1, 0)
    border_width = min(border_width, max_border_width)
    if type(shape) is shapes.Circle:
        pygame.draw.ellipse(screen, colour, bounding_rectangle, border_width)
    elif type(shape) is shapes.Axis:
        screen_pos_from = np.multiply(shape.center, scaling)
        screen_pos_to = copy.copy(screen_pos_from)
        screen_pos_to[shape.dimension] = render_dimensions[shape.dimension]
        pygame.draw.line(screen, colour, screen_pos_from, screen_pos_to)
    elif type(shape) is shapes.Rectangle:
        pygame.draw.rect(screen, colour, bounding_rectangle, border_width)
    return bounding_rectangle

# class AsciiRenderer(Renderer):
#     def __init__(self, source_dimensions, render_dimensions=(120, 20)):
#         self.render_dimensions = render_dimensions
#         self._source_dimensions = source_dimensions
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
