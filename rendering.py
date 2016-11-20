import threading
import copy
import operator
import shapes
import numpy as np
import pygame
import sys


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

class PyGame(Renderer):
    def __init__(self, environment, render_delay = 100, physics_delay = 0.00, render_width = 900, render_height = 600):
        pygame.init()
        self.screen = pygame.display.set_mode((render_width, render_height))
        super().__init__(environment)
        self._render_delay = render_delay
        self._physics_delay = physics_delay
        self._render_width = render_width
        self._render_height = render_height
        self._render_left = 0
        self._render_top = 0

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        shapes_to_render = []
        pixels = []
        for creature in self._environment._living_creatures:
            for organ in creature._organs:
                if organ is not creature.get_body() and organ.get_shape() is not None:
                    shapes_to_render.append(copy.deepcopy(organ.get_shape()))
                    pixels.append((0,255,0))
            shapes_to_render.append(copy.deepcopy(creature.get_body().get_shape()))
            pixels.append((0,0,255))
        for food in self._environment._food:
            shapes_to_render.append(copy.deepcopy(food.get_shape()))
            pixels.append((255,0,0))
        additional_1 = ["width(w): " + str(self._render_width), "height(h): " + str(self._render_height),
                        "physics delay(p): " + str(self._physics_delay),
                        "render delay(r): " + str(self._render_delay)]
        side_info = []
        for creature in sorted(self._environment._living_creatures, key=lambda x: x.get_energy()):
            side_info.append(creature.get_name() + ": " + str(creature.get_energy()))

        t = threading.Thread(target=PyGameRender, args=(self.screen, shapes_to_render, pixels, [additional_1], side_info,
                                                           self._render_width, self._render_height,
                                                           self._render_left,
                                                           self._render_top, self._environment._width,
                                                           self._environment._height))
        t.start()
        print("+")

def PyGameRender(screen, shapes_to_render, pixels, additionals, side_infos, output_width, output_height, x_init, y_init, width, height):
    print("PyGameRenderer")
    screen.fill((0,0,0))
    for shape, colour in zip(shapes_to_render, pixels):
        pygame.draw.circle(screen,colour+(0,),(int(shape.get_x()*10),int(shape.get_y()*10)),int(shape.get_radius()*10),0)
    pygame.display.flip()

class AsciiRenderer(Renderer):
    def __init__(self, environment, render_delay = 100, physics_delay = 0.00, render_width = 120, render_height = 20):
        super().__init__(environment)
        self._render_delay = render_delay
        self._physics_delay = physics_delay
        self._render_width = render_width
        self._render_height = render_height
        self._render_left = 0
        self._render_top = 0

    def render(self):
        shapes_to_render = []
        pixels = []
        for creature in self._environment._living_creatures:
            for organ in creature._organs:
                if organ is not creature.get_body() and organ.get_shape() is not None:
                    shapes_to_render.append(copy.deepcopy(organ.get_shape()))
                    pixels.append("X")
            shapes_to_render.append(copy.deepcopy(creature.get_body().get_shape()))
            pixels.append(creature._name[len(creature._name) - 1])
        for food in self._environment._food:
            shapes_to_render.append(copy.deepcopy(food.get_shape()))
            pixels.append(".")
        additional_1 = ["width(w): " + str(self._render_width), "height(h): " + str(self._render_height),
                        "physics delay(p): " + str(self._physics_delay), "render delay(r): " + str(self._render_delay)]
        side_info = []
        for creature in sorted(self._environment._living_creatures, key=lambda x: x.get_energy()):
            side_info.append(creature.get_name() + ": " + str(creature.get_energy()))

        t = threading.Thread(target=render_to_ascii, args=(shapes_to_render, pixels, [additional_1], side_info,
                                                  self._render_width, self._render_height, self._render_left,
                                                  self._render_top, self._environment._width, self._environment._height))
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