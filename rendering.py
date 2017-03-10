import threading
import copy
import operator
import shapes
import numpy as np
import time
import pygame
import sys
from pygame.locals import *
import genesis as gen
import binary_tree as bt

lock = threading.Lock()

def listen_for_key(self):
    def w(op): self._render_width = op(self._render_width, 10)
    def h(op): self._render_height = op(self._render_height, 10)
    def p(op): self._physics_delay = op(self._physics_delay, 0.1)
    def r(op): self._render_delay = op(self._render_delay, 5)

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


class Renderer:
    def __init__(self, environment):
        self._environment = environment

    def render(self):
        pass

rects_previously_rendered = []


class PyGameRenderer(Renderer):
    def __init__(self, environment, render_width=900, render_height=600):
        pygame.init()
        self.screen = pygame.display.set_mode((render_width, render_height))
        super().__init__(environment)
        self._render_dimensions = [render_width, render_height]
        self._render_pos = [0, 0]
        self._last_render_time = -1
        self._last_rendered_rects = []

    def render(self):
        clock = self._environment.time[gen.RENDER_KEY]
        clock.tick()
        render_time = time.time()

        shapes_to_render = []
        pixels = []

        # ## #
        # the following is a visualisation of the binary tree. uncomment to see how the food pellets are classified
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

        for creature in self._environment.living_creatures:
            for organ in creature.organs:
                shape = organ.shape
                if organ is not creature.body and shape is not None:
                    shapes_to_render.append(copy.deepcopy(shape))
                    pixels.append((0, 255, 0))
            shapes_to_render.append(copy.deepcopy(creature.body.shape))
            pixels.append((0, 0, 255 - 200*(creature.age / gen.MAX_AGE)))
        for food in self._environment.food_tree.elements:
            shapes_to_render.append(copy.deepcopy(food.shape))
            pixels.append((255, 0, 0))
        rects_to_render = []
        for shape in shapes_to_render:
            rects_to_render.append(shape.to_bounding_box())
        additional = ["width(w): " + str(self._render_dimensions[0]), "height(h): " + str(self._render_dimensions[1]),
                      "ticks: " + str(self._environment.tick_count),
                      "frames/s: " + str(1 / (time.time() - self._last_render_time)),
                      "physics/s: " + str(1 / max(self._environment.last_tick_delta, 0.00001))
                      ]
        self._last_render_time = time.time()
        side_info = []
        for creature in sorted(self._environment.living_creatures, key=lambda x: x.mass):
            side_info.append(creature.name + ": " + str(creature.mass))

        t = threading.Thread(target=render_with_pygame, args=(self._environment,
                                                              self.screen, shapes_to_render, pixels, additional,
                                                              side_info, rects_previously_rendered,
                                                              self._render_dimensions, self._render_pos,
                                                              [self._environment.width, self._environment.height]))
        t.start()
        clock.tock()


def render_with_pygame(env, screen, shapes_to_render, pixels, additionals, side_infos, rects_previously_rendered,
                       render_dimensions, pos, source_dimensions):
    lock.acquire()
    clock = env.time[gen.RENDER_THREAD_KEY]
    clock.tick()

    scaling = np.divide(render_dimensions, source_dimensions)

    screen.fill((0, 0, 0))

    # for shape in rects_to_render:
    #     draw_shape(shape, screen, (255, 0, 255, 0), scaling, render_dimensions)
    rects = []
    for shape, colour in zip(reversed(shapes_to_render), reversed(pixels)):
        c = colour+(0,)
        drawn = draw_shape(shape, screen, c, scaling, render_dimensions)
        rects.append(drawn)

    # pygame.draw.circle()
    # i = 0
    # for additional in additionals[3:4]:
    #     font = pygame.font.Font(None, 36)
    #     text = font.render(additional, 0, (255, 255, 255))
    #     #textpos = text.get_rect()
    #     # textpos. = ().centerx
    #     screen.blit(text, (0, 36/2+26*i))
    #     i += 1
    pygame.display.update(rects + rects_previously_rendered)
    del rects_previously_rendered[:]
    rects_previously_rendered += rects
    clock.tock()
    lock.release()


def transform_to_bounding(shape, scaling, render_dimensions):
    top_left = (shape.left * scaling[0], shape.down * scaling[1])
    shape_dimensions = (int(round(shape.width * scaling[0])),
                        int(round(shape.height * scaling[1])))
    rect = (top_left[0], top_left[1], shape_dimensions[0], shape_dimensions[1])
    return rect


def draw_shape(shape, screen, colour, scaling, render_dimensions):
    rect = transform_to_bounding(shape, scaling, render_dimensions)
    if type(shape) is shapes.Circle:
        pygame.draw.ellipse(screen, colour, rect, 0)
    elif type(shape) is shapes.Axis:
        screen_pos_from = np.multiply(shape.center, scaling)
        screen_pos_to = copy.copy(screen_pos_from)
        screen_pos_to[shape.dimension] = render_dimensions[shape.dimension]
        pygame.draw.line(screen, colour, screen_pos_from, screen_pos_to)
    elif type(shape) is shapes.Rectangle:
        pygame.draw.rect(screen, colour, rect, 0)
    return rect


class AsciiRenderer(Renderer):
    def __init__(self, environment, render_width=120, render_height=20):
        super().__init__(environment)
        self._render_width = render_width
        self._render_height = render_height
        self._render_left = 0
        self._render_top = 0
        self._last_render_time = -1

    def render(self):
        shapes_to_render = []
        pixels = []
        for creature in self._environment.living_creatures:
            for organ in creature.organs:
                if organ is not creature.body and organ.shape is not None:
                    shapes_to_render.append(copy.deepcopy(organ.shape))
                    pixels.append("X")
            shapes_to_render.append(copy.deepcopy(creature.body.shape))
            pixels.append(creature.name[len(creature.name) - 1])
        for food in self._environment.food_pellets:
            shapes_to_render.append(copy.deepcopy(food.shape))
            pixels.append(".")
        additional_1 = ["width(w): " + str(self._render_width), "height(h): " + str(self._render_height),
                        "ticks: " + str(self._environment.tick_count),
                        "frames/s: "+str(1/(time.time()-self._last_render_time)),
                        "physics/s: "+str(1/self._environment.last_tick_delta)]
        side_info = []
        for creature in sorted(self._environment.living_creatures, key=lambda x: x.get_energy()):
            side_info.append(creature.name + ": " + str(creature.get_energy()))

        t = threading.Thread(target=render_to_ascii, args=(shapes_to_render, pixels, [additional_1], side_info,
                                                           self._render_width, self._render_height,
                                                           self._render_left, self._render_top,
                                                           self._environment.width, self._environment.height))
        self._last_render_time = time.time()
        t.start()
        print("+")


def render_to_ascii(shapes_to_render, pixels, additionals, side_infos, output_width, output_height, x_init, y_init, width, height):
    x_step_size = width / output_width
    y_step_size = height / output_height
    line = "#"+"-"*output_width+"#\n"
    text = ""
    for additional in additionals:
        text_line = ""
        first = True
        for value in additional:
            if not first:
                text_line += ", "
            first = False
            text_line += value
        text += "|"+text_line[0:output_width]+" "*(output_width-len(text_line))+"|\n"
    stage = ""
    for i in range(len(side_infos), output_height):
        side_infos.append("")
    for y, side_info in zip(np.arange(y_init, y_init + height, y_step_size), side_infos):
        stage += "|"
        for x in np.arange(x_init, x_init + width, x_step_size):
            pixel = " "
            point = shapes.Circle(x, y, 0)
            for shape, pixel2 in zip(shapes_to_render, pixels):
                if point.collides(shape):
                    pixel = pixel2
                    break
            stage += pixel
        stage += "|"+side_info+"\n"
    print("\n\n\n" + line + text + line + stage + line)