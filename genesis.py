import math
import time, threading
import copy
import shapes
import random
import time
import numpy as np
import operator
import binary_tree
from clock import *
import rendering
import functools
import colours

# G todo: add some reusable way of mutating parameters. maybe using the Beta function?
THINKING_KEY = "thinking"
TICK_KEY = "tick"
FOOD_COLLISION_KEY = "food collision"
RENDER_KEY = "render"
RENDER_THREAD_KEY = "render thread"
FOOD_RECLASSIFICATION_KEY = "food reclassification"

MIN_FOOD_MASS_TO_CONSUME = 0.05
FOOD_GROWTH_RATE = 0.4
FOOD_GROWTH_DELAY = 10
MAX_CREATURE_AGE = 250
MAX_FOOD_MASS = 100

FONT_TYPE = "arial"
FONT_SIZE = 10
HEADLINE_SIZE = 16


def random_pos(width, height, borders=0):
    return [random_from_interval(borders, width-borders), random_from_interval(borders, height-borders)]


def random_from_interval(min_value, max_value):
    """Returns a number which lies between min and max."""
    return random.random()*(max_value - min_value) + min_value


def binary_activation(input_value):
    return 0 if input_value <= 0 else 1


def identity_activation(input_value):
    return input_value


def sigmoid_activation(input_value):
    return math.tanh(input_value)


def convert_to_delta_distance(distance, angle):
    """Takes a distance and a direction and computes the resulting 2D vector."""
    rad = math.radians(angle)
    return [math.sin(rad)*distance, math.cos(rad)*distance]


def clip(number, min_value, max_value):
    """Clips a number to be no smaller than min_value and bigger than max_value. min_value must be smaller than
    max_value, otherwise the behaviour is undefined"""
    return min(max(number, min_value), max_value)


def get_dict_attr(obj, attr):
    for obj in [obj]+obj.__class__.mro():
        if attr in obj.__dict__:
            return obj.__dict__[attr]
    raise AttributeError


class OrganHighlight(rendering.SimpleContainer):
    def __init__(self, dimensions, header_text, organ_count, padding=6,
                 camera=rendering.RelativeCamera()):
        super().__init__(shapes.rect(dimensions), camera)
        self.table = rendering.ValueDisplay(FONT_TYPE, FONT_SIZE, colours.WHITE)
        # self.table.add_row("tick cost", 0)
        self.header_text = header_text
        self.organ_graphics = []
        self.highlighted_organ = None
        self.padding = padding
        self.last_y = padding
        if organ_count > 0:
            header_text += " "+str(organ_count)
        self.header_label = rendering.TextGraphic(header_text, FONT_TYPE, HEADLINE_SIZE)
        self.header_label.position = (padding, self.last_y)
        self.last_y += self.header_label.local_bounding_rectangle.height
        self.line_graphic = None
        self.add_canvas(self.table, (padding, self.last_y+padding))
        self.last_y += self.table.local_bounding_rectangle.height + padding

    def _set_final_height(self, height):
        if self.line_graphic is not None:
            self.remove_canvas(self.line_graphic)
        line_side_distance = self.local_bounding_rectangle.width / 7
        line_height = height + self.padding * 3
        line_start = (line_side_distance, line_height)
        line_end = (self.local_bounding_rectangle.width - line_side_distance, line_height)
        self.line_graphic = rendering.SimpleOutlineGraphic(shapes.LineSegment(line_start, line_end), (100, 100, 100, 0))
        self.add_canvas(self.line_graphic)
        final_height = line_height+self.padding*0.5
        self.local_bounding_rectangle.height = final_height

    def highlight(self, organ):
        if organ is not None:
            if self.highlighted_organ is not None:
                self.highlight(None)
            self.add_canvas(self.header_label)
            self.highlighted_organ = organ
            self.visualize(organ)
            max_y = 0
            for graphic in self.organ_graphics:
                max_y = max(max_y, graphic.local_bounding_rectangle.down, graphic.local_bounding_rectangle.up)
                self.add_canvas(graphic)
            for canvas in self.children:
                max_y = max(max_y, canvas.local_bounding_rectangle.up+canvas.position[1])
            self._set_final_height(max_y)
        else:
            self.remove_canvas(self.header_label)
            self.remove_canvases(self.organ_graphics)
            self.organ_graphics = []

    def add_text(self, text, x_value=None, y_value=None):
        if x_value is None:
            x_value = self.padding
        update_last_y = y_value is None
        if update_last_y:
            y_value = self.last_y
        label = rendering.TextGraphic(text, FONT_TYPE, FONT_SIZE)
        label.position = (x_value, y_value)
        self.organ_graphics.append(label)
        if update_last_y:
            self.last_y += label.local_bounding_rectangle.height
        return label

    def print_grid_text(self, grid_text):
        grid = []
        last_column_max_x = 0
        max_x = 0
        initial_y = self.last_y
        for column in grid_text:
            column_labels = []
            grid.append(column_labels)
            last_column_max_x += self.padding
            self.last_y = initial_y
            for word in column:
                text = self.add_text(str(word), last_column_max_x)
                column_labels.append(text)
                max_x = max(max_x, text.local_bounding_rectangle.right)
            last_column_max_x = max_x
        return grid

    def visualize(self, organ):
        pass

    def refresh_values(self):
        self.table.refresh_values()


class MouthHighlight(OrganHighlight):
    def __init__(self, dimensions, organ_count, camera=rendering.RelativeCamera()):
        super().__init__(dimensions, "Mouth", organ_count, camera=camera)

    def visualize(self, mouth):
        labels = ["body distance", "rotation", "mouth_radius", "food capacity"]
        values = [mouth.body_distance, mouth.rotation, mouth.mouth_radius, mouth.food_capacity]
        self.table.add_rows(zip(labels, values))
        self.table.add_row(("total amount eaten", lambda: self.highlighted_organ.total_amount_eaten))


class LegsHighlight(OrganHighlight):
    def __init__(self, dimensions, organ_count, camera=rendering.RelativeCamera()):
        super().__init__(dimensions, "Legs", organ_count, camera=camera)

    def visualize(self, legs):
        self.table.add_row(("max travel distance", legs.max_distance))
        self.table.add_row(("max degrees turn", legs.max_degree_turn))
        self.table.add_row(("total distance traveled", lambda: legs.total_distance_moved))


class FissionHighlight(OrganHighlight):
    def __init__(self, dimensions, organ_count, camera=rendering.RelativeCamera()):
        super().__init__(dimensions, "Fission", organ_count, camera=camera)

    def visualize(self, fission):
        self.table.add_row(("offsprings produced", lambda: self.highlighted_organ.offsprings_produced))


class BodyHighlight(OrganHighlight):
    def __init__(self, dimensions, organ_count, camera=rendering.RelativeCamera()):
        super().__init__(dimensions, "Body", organ_count, camera=camera)

    def visualize(self, body):
        self.table.add_row(("mass", lambda: self.highlighted_organ.mass))
        self.table.add_row(("rotation", lambda: self.highlighted_organ.rotation))
        self.table.add_row(("position", lambda: self.highlighted_organ.center))


class EyeHighlight(OrganHighlight):
    def __init__(self, dimensions, organ_count, camera=rendering.RelativeCamera()):
        super().__init__(dimensions, "Eye", organ_count, camera=camera)

    def visualize(self, eye):
        labels = ["body distance", "rotation", "radius", "food spotted"]
        values = [eye.body_distance,  eye.rotation, eye.radius, lambda: eye.food_pellets_spotted_count]
        self.table.add_rows(zip(labels, values))

    # def refresh_values(self):
    #     self.table.set(1, 3,  str(self.highlighted_organ.food_pellets_spotted_count))


def get_graph_points(activation_function, input_range, scalar, offset, x_steps):
    points = []
    for x in np.arange(input_range[0], input_range[1], x_steps):
        points.append((x*scalar[0]+offset[0], activation_function(x)*scalar[1]+offset[1]))
    clean_points = []
    last_added_point = None
    for i in range(len(points)-1):
        point = points[i]
        if last_added_point is None:
            clean_points.append(point)
            last_added_point = point
        else:
            prev_slope = (last_added_point[1] - point[1]) / (last_added_point[0] - point[0])
            next_slope = (point[1] - points[i+1][1]) / (point[0] - points[i+1][0])
            slope_delta = prev_slope - next_slope
            if prev_slope != -1:
                if abs(slope_delta) > 0.1:
                    clean_points.append(point)
                    last_added_point = point
    clean_points.append(points[-1])
    return clean_points


class BrainHighlight(OrganHighlight):
    def __init__(self, dimensions, organ_count, camera=rendering.RelativeCamera()):
        super().__init__(dimensions, "Brain", organ_count, camera=camera)
        self.live_button = rendering.TextButton("live view", FONT_TYPE, FONT_SIZE)
        button_left = self.header_label.local_bounding_rectangle.right + self.padding
        button_top = self.header_label.local_bounding_rectangle.down
        self.add_canvas(self.live_button, (button_left, button_top))
        def live_view_pressed(event): self.toggle_live_view()
        self.live_button.action_listeners.append(live_view_pressed)
        self.neuron_row_height = 30
        self.neuron_label_width = 0
        self.neuron_radius = 10
        self.border = 10
        self.column_width = 50
        self.text_offset = 6
        self.neuron_graphics = {}
        self.synapse_graphics = {}
        self.live_view = False
        # self.grid = rendering.Table(12, colours.rgb(255), 4)
        # self.add_canvas(self.grid)

    def __draw_neuron(self, neuron, x, y):
        circle = shapes.Circle((x, y), self.neuron_radius)
        neuron_circle = rendering.SimpleMonoColouredGraphic(circle, colours.BLACK)
        self.organ_graphics.append(neuron_circle)
        self.neuron_graphics[neuron] = neuron_circle
        scaling = (self.neuron_radius / 2, -self.neuron_radius / 2)
        points = get_graph_points(neuron._activation_function, (-2, 2), scaling, (x, y), 0.2)
        synapse_lines = shapes.PointLine(points)
        self.organ_graphics.append(rendering.SimpleOutlineGraphic(synapse_lines, colours.WHITE, possibly_obstructed=True))

    def visualize(self, brain):
        max_vertical = self.last_y
        row_y_values = []
        for row_index in range(max(len(brain.input_layer),
                                   len(brain.hidden_layer),
                                   len(brain.output_layer))):
            y = self.last_y + self.neuron_radius + row_index * self.neuron_row_height
            row_y_values.append(y)
        for neuron, neuron_index in zip(brain.input_layer, range(len(brain.input_layer))):
            y = row_y_values[neuron_index]
            if neuron.label is not None:
                text = self.add_text(neuron.label, y_value=y - self.text_offset)
                self.neuron_label_width = max(self.neuron_label_width, text.local_bounding_rectangle.width)
        right_most_neuron_x = 0
        for layer, layer_index in zip(brain.layers, range(len(brain.layers))):
            x = self.padding + self.neuron_label_width + self.border + layer_index * self.column_width + self.neuron_radius
            right_most_neuron_x = max(right_most_neuron_x, x)
            neuron_index = 0
            for neuron in layer:
                y = row_y_values[neuron_index]
                max_vertical = max(max_vertical, y)
                self.__draw_neuron(neuron, x, y)

                neuron_index += 1
        right_label_x = right_most_neuron_x + self.neuron_radius + self.border
        for neuron, neuron_index in zip(brain.output_layer, range(len(brain.output_layer))):
            if neuron.label is not None:
                y = row_y_values[neuron_index]
                text = self.add_text(neuron.label, x_value=right_label_x, y_value=y-self.text_offset)
                self.neuron_label_width = max(self.neuron_label_width, text.local_bounding_rectangle.width)
                # self.organ_graphics.append(label)
        for layer, layer_index in zip(brain.layers, range(len(brain.layers))):
            if layer_index + 1 < len(brain.layers):
                next_layer = brain.layers[layer_index + 1]
                for neuron in layer:
                    neuron_shape = self.neuron_graphics[neuron].shape
                    self.synapse_graphics[neuron] = {}
                    for next_neuron in next_layer:
                        if neuron.has_weight(next_neuron):
                            next_neuron_shape = self.neuron_graphics[next_neuron].shape
                            synapse_shape = shapes.LineSegment(neuron_shape.center, next_neuron_shape.center)
                            synapse_graphic = rendering.SimpleOutlineGraphic(synapse_shape, colours.WHITE)
                            self.organ_graphics.insert(0, synapse_graphic)
                            self.synapse_graphics[neuron][next_neuron] = synapse_graphic

        # self._set_final_height(self.last_y)
        self.last_y = max_vertical
        self.table.translate((0, self.last_y))
        self.draw_static_view()

    def toggle_live_view(self):
        self.live_view = not self.live_view
        if self.live_view:
            self.draw_live_view()
        else:
            self.draw_static_view()
        self.refresh_values()

    def refresh_values(self):
        if self.live_view:
            self.draw_live_view()

    def redraw(self, neuron_colour_f, synapse_colour_f):
        brain = self.highlighted_organ
        for layer, layer_index in zip(brain.layers, range(len(brain.layers))):
            for neuron in layer:
                neuron_shape = self.neuron_graphics[neuron]
                neuron_shape.fill_colour = neuron_colour_f(neuron)
                if layer_index + 1 < len(brain.layers):
                    next_layer = brain.layers[layer_index + 1]
                    for next_neuron in next_layer:
                        if neuron.has_weight(next_neuron):
                            synapse_graphic = self.synapse_graphics[neuron][next_neuron]
                            synapse_colour = synapse_colour_f(neuron, next_neuron)
                            if sum(synapse_colour) > 20:
                                synapse_graphic.is_visible = True
                                synapse_graphic.border_colour = synapse_colour
                            else:
                                synapse_graphic.is_visible = False

    def draw_live_view(self):
        neuron_colour = lambda n: colours.visualise_magnitude(n.last_amount)
        synapse_colour = lambda n1, n2: colours.visualise_magnitude((n1.last_amount * n1.get_weight(n2)))
        self.redraw(neuron_colour, synapse_colour)

    def draw_static_view(self):
        neuron_colour = lambda n: colours.grey(50) if n.is_bias else colours.grey(150)
        synapse_colour = lambda n1, n2: colours.visualise_magnitude(n1.get_weight(n2) * 0.5)
        self.redraw(neuron_colour, synapse_colour)


class CreatureHighlight(rendering.ScrollingPane):
    def __init__(self, dimensions, camera=rendering.RelativeCamera(), padding=5):
        super().__init__(shapes.rect(dimensions), True, False, camera)
        self.border_width = 3
        self.border_colour = colours.WHITE
        self.back_ground_colour = colours.BLACK
        self.highlighted_creature = None
        self.organ_highlights = []
        self.padding = padding
        self.last_y = self.padding
        self.header_label = rendering.TextGraphic("Creature", FONT_TYPE, HEADLINE_SIZE)
        self.pane.add_canvas(self.header_label)
        self.header_label.position = (padding, self.last_y)
        self.last_y += self.header_label.local_bounding_rectangle.height
        self.initial_y = self.last_y
        self.table = None

    def highlight(self, creature):
        if self.highlighted_creature is not creature:
            if creature is not None:
                print("clicked creature!")
                self.last_y = self.initial_y

                organ_type_counter = {}
                organ_type_index = {}
                for organ in creature.organs:
                    if type(organ) not in organ_type_counter:
                        organ_type_counter[type(organ)] = 0
                        organ_type_index[type(organ)] = 1
                    organ_type_counter[type(organ)] += 1
                if self.highlighted_creature is not None:
                    self.highlight(None)
                self.table = rendering.ValueDisplay(FONT_TYPE, FONT_SIZE, colours.WHITE)
                self.organ_highlights.append(self.table)
                self.table.add_row(("generation", creature.generation))
                self.table.add_row(("is alive", lambda: "yes" if creature.alive else "no"))
                self.table.add_row(("age", lambda: creature.age))
                self.pane.add_canvas(self.table, (self.padding, self.last_y + self.padding))
                self.last_y += self.table.local_bounding_rectangle.height + self.padding
                creature.highlight()
                self.highlighted_creature = creature
                dimensions = self.local_bounding_rectangle.dimensions
                for organ in creature.organs:
                    if organ_type_counter[type(organ)] > 1:
                        index = organ_type_index[type(organ)]
                        organ_type_index[type(organ)] += 1
                    else:
                        index = 0
                    if type(organ) is Brain:
                        highlight = BrainHighlight(dimensions, index)
                    elif type(organ) is Body:
                        highlight = BodyHighlight(dimensions, index)
                    elif type(organ) is Mouth:
                        highlight = MouthHighlight(dimensions, index)
                    elif type(organ) is EuclideanEye:
                        highlight = EyeHighlight(dimensions, index)
                    elif type(organ) is Fission:
                        highlight = FissionHighlight(dimensions, index)
                    elif type(organ) is Legs:
                        highlight = LegsHighlight(dimensions, index)
                    else:
                        continue

                    self.pane.add_canvas(highlight, (3, self.last_y))
                    highlight.highlight(organ)
                    highlight.table.add_row(("tick cost", organ.tick_cost))
                    self.organ_highlights.append(highlight)
                    self.last_y += highlight.local_bounding_rectangle.height
                if creature.environment is not None:
                    creature.environment.tick_listeners.append(self.refresh_values)
                self.pane.local_bounding_rectangle.height = self.last_y
            else:
                if self.highlighted_creature is not None:
                    if self.highlighted_creature.environment is not None:
                        self.highlighted_creature.environment.tick_listeners.remove(self.refresh_values)
                    self.highlighted_creature.un_highlight()
                    self.highlighted_creature = None
                    # self.pane.remove_canvas(self.table)
                    for canvas in self.pane.children[:]:
                        self.pane.remove_canvas(canvas)
                    self.table = None

    def refresh_values(self):
        if self.table is not None:
            self.table.refresh_values()
            for highlight in self.pane.children:
                if isinstance(highlight, OrganHighlight):
                    highlight.refresh_values()


class Environment(rendering.SimpleContainer):
    """This class represents the environment in which Creatures live. Not only does it manage the creatures living in
    it but also the Food which is meant to be consumed by the Creatures. The environment has no real sense of time. It
    has to be controlled from the outside via its Environment.tick() method. Whenever this method is called, the world
    "continues". Although the environment has no real sense of time, it has a variable tick_count, which counts up one
    each call to Environment.tick() and represents its internal time keeping.

    Creatures can not simply be added to the environment at any time. They need to be queued with
    Environment.queue_creature(creature) which will then be added on the next call to Environment.tick().

    Creatures in the world can not decide for themselves how they can move around. They need to make call the method
    move_creature(creature, distance_to_travel)."""
    def __init__(self, camera, local_dimensions=(1000, 1000)):
        super().__init__(shapes.rect(local_dimensions), camera, back_ground_colour=colours.BLACK)
        self.__tick_count = 0
        self.__stage_objects = []
        self.__living_creatures = []
        self.__food_tree = binary_tree.BinaryTree(local_dimensions, 6)
        self.__queued_creatures = []
        self.tick_listeners = []
        self.__last_tick_time = -1
        self.__last_tick_delta = -1
        self.__creature_highlight = None
        self.clocks = ClockTower([Clock(FOOD_COLLISION_KEY), Clock(TICK_KEY), Clock(THINKING_KEY), Clock(RENDER_KEY),
                                 Clock(FOOD_RECLASSIFICATION_KEY), Clock(RENDER_THREAD_KEY)])

    @property
    def creature_highlight(self):
        return self.__creature_highlight

    @creature_highlight.setter
    def creature_highlight(self, creature_highlight):
        if self.__creature_highlight is not creature_highlight:
            self.__creature_highlight = creature_highlight
            def click_listener(event):
                found_creature = False
                local_point = self.transform_point_from_screen(event.screen_mouse_position)
                for creature in reversed(self.living_creatures):
                    if creature.body.shape.point_lies_within(local_point):
                        creature_highlight.highlight(creature)
                        found_creature = True
                        break
                if not found_creature:
                    creature_highlight.highlight(None)
            self.mouse_pressed_event_listeners.append(click_listener)

    @property
    def graphics(self):
        graphics = []
        for creature in self.living_creatures:
            for organ in creature.organs:
                graphics += organ.graphics
        for food in self.__food_tree.elements:
            graphics += food.graphics
        return graphics

    @property
    def last_tick_time(self):
        return self.__last_tick_time

    @property
    def last_tick_delta(self):
        return self.__last_tick_delta

    @property
    def tick_count(self):
        return self.__tick_count

    @property
    def living_creatures(self):
        return self.__living_creatures

    @property
    def stage_objects(self):
        return self.__stage_objects

    @property
    def food_tree(self):
        return self.__food_tree

    @property
    def width(self):
        return self.local_bounding_rectangle.width

    @property
    def height(self):
        return self.local_bounding_rectangle.height

    @property
    def queued_creatures(self):
        return self.__queued_creatures

    def queue_creature(self, creature):
        """Queues a creature to be placed in the Environment as soon as Environment.tick() is called"""
        self.__queued_creatures.append(creature)

    def remove_creature(self, creature):
        """Call this method to remove a creature from this Environment. The Creature will no longer receive tick()
        function calls from this Environment."""
        if creature.exists:
            self.__living_creatures.remove(creature)
        else:
            self.__queued_creatures.remove(creature)
        self.__remove_stage_object(creature)

    def __add_stage_object(self, stage_object):
        self.stage_objects.append(stage_object)
        stage_object.environment = self
        for graphic in stage_object.graphics:
            shape = graphic.shape
            pos = (shape.left, shape.down)
            self.add_canvas(graphic, pos)

    def __remove_stage_object(self, stage_object):
        if stage_object.environment is not None:
            stage_object.environment = None
        self.__stage_objects.remove(stage_object)
        self.remove_canvases(stage_object.graphics)

    def tick(self):
        """As an Environment has no real sense of time, this method must be called periodically from the outside."""
        tick_clock = self.clocks[TICK_KEY]
        tick_clock.tick()
        for food in self.food_tree.elements:
            food.tick()
            reclass_clock = self.clocks[FOOD_RECLASSIFICATION_KEY]
            reclass_clock.tick()
            if food.has_changed:
                self.food_tree.reclassify(food, food.shape)
                food.has_changed = False
            reclass_clock.tock()
        for creature in self.__queued_creatures:
            if creature.alive:
                self.__living_creatures.append(creature)
                self.__add_stage_object(creature)
        self.__queued_creatures.clear()
        for creature in self.__living_creatures[:]:
            creature.sense()
        for creature in self.__living_creatures[:]:
            creature.tick(self.__tick_count)
        for creature in self.__living_creatures[:]:
            creature.execute()
        self.__tick_count += 1
        for listener in self.tick_listeners:
            listener()
        current_time = time.time()
        self.__last_tick_delta = current_time - self.__last_tick_time
        self.__last_tick_time = current_time
        tick_clock.tock()

    def move_creature(self, creature, distance_to_travel):
        """Creatures can not move freely in the Environment as they please. To move a Creature, one must call this
        method and after the Environment has done the necessary collision detection it decides on how far the creature
        actually moved. This distance is then returned."""
        [x, y] = [creature.center_x, creature.center_y]
        rotation = creature.body.rotation
        [delta_x, delta_y] = convert_to_delta_distance(distance_to_travel, rotation)
        # G todo: calculate new position using x, y, distance_to_travel & rotation
        [new_x, new_y] = [x + delta_x, y + delta_y]
        translated_shape = copy.deepcopy(creature.body.shape)
        translated_shape.center = [new_x, new_y]
        # G todo: the following only computes if the the new position is valid.
        # G todo- it doesn't calculate how far the object should actually move instead!
        is_valid = True
        [width, height] = creature.body.shape.dimensions
        if new_x+width/2 > self.width or new_x-width/2 < 0 or new_y+height/2 > self.height or new_y-height/2 < 0:
            is_valid = False
        if is_valid:
            creature.body.center = [new_x, new_y]
        return distance_to_travel if is_valid else 0

    def turn_creature(self, creature, angle_to_turn):
        """Creatures can not turn freely, as this might result in a collision. To turn a Creature, one must call this
        method and after the Environment has done the necessary collision detection it decides on how far the creature
        actually turned. The angle is then returned."""
        if type(creature.body.shape) is shapes.Circle:
            creature.body.rotation = creature.body.rotation + angle_to_turn
            return angle_to_turn
        else:
            raise Exception("Turning creatures is not implemented for shape " + str(type(creature.body.shape)))

    def find_colliding_food(self, shape, break_if=lambda x: False):
        colliding_clock = self.clocks[FOOD_COLLISION_KEY]
        colliding_clock.tick()
        food_found = []
        for food in self.food_tree.get_collision_candidates(shape):
            if shape.collides(food.shape):
                food_found.append(food)
                if break_if(food_found):
                    break
        colliding_clock.tock()
        return food_found

    def sum_mass(self, food_pellets):
        summed = 0
        for food in food_pellets:
            summed += food.mass
        return summed

    def create_food(self, x, y, mass):
        """Creates Food of circular shape at the specified destination with the given mass."""
        self.add_food(Food(mass, x, y))

    def add_food(self, food):
        """Add the specified Food to the Environment for further consumption by Creatures populating it."""
        self.food_tree.classify(food, food.shape)
        self.__add_stage_object(food)

    def remove_food(self, food):
        """Removes the specified piece of Food from the Environment."""
        self.food_tree.remove(food)
        self.__remove_stage_object(food)


class StageObject:
    def __init__(self):
        self.__environment = None
        self._last_tick_count = -1
        self.age = 0

    @property
    def graphics(self):
        raise Exception("Not implemented!")

    @property
    def exists(self):
        return self.__environment is not None

    @property
    def environment(self):
        return self.__environment

    @environment.setter
    def environment(self, environment):
        if environment is not None and self not in environment.stage_objects:
            raise Exception("Can't set environment in StageObject as long as it doesn't exist in that environment."
                            "Please call Environment.add_stage_object(stage_object) and wait for the following"
                            "Environment.tick().")
        self.__environment = environment


class FoodGraphic(rendering.MonoColouredGraphic):
    def __init__(self, food):
        super().__init__(possibly_obstructed=True)
        self.__is_visible = True
        self.food = food

    @property
    def shape(self):
        return self.food.shape

    @property
    def fill_colour(self):
        return colours.GREEN

    @property
    def is_visible(self):
        return self.__is_visible

    @is_visible.setter
    def is_visible(self, is_visible):
        if is_visible is not self.__is_visible:
            self.__is_visible = is_visible
            self.redraw_surface = True


class Food(StageObject):
    """This class represents a piece of Food. It can be placed in the Environment and consumed by Creatures."""
    def __init__(self, mass, x, y):
        super().__init__()
        self.__shape = shapes.Circle((x, y), 0)
        self.__mass = None
        self.graphic = FoodGraphic(self)
        self.has_changed = True
        self.time_until_growth = 0
        self.mass = mass

    @property
    def mass(self):
        """Returns the Foods mass."""
        return self.__mass

    @mass.setter
    def mass(self, amount):
        """Sets the mass of this piece of Food."""
        if amount != self.__mass:
            self.has_changed = True
            self.__mass = amount
            self.__shape.radius = math.sqrt(self.__mass / 2)
            self.time_until_growth = FOOD_GROWTH_DELAY
            if self.__mass < 0.05:
                self.kill()
            self.graphic.redraw_surface = True

    @property
    def graphics(self):
        return [self.graphic]

    def kill(self):
        """Destroys this piece of Food, removing it from its Environment."""
        if self.environment is not None:
            self.environment.remove_food(self)

    @property
    def shape(self):
        """Returns the physical representation of this piece of Food."""
        return self.__shape

    def tick(self):
        """This method should be called, each time a virtual time unit (tick) has passed."""
        if self.time_until_growth == 0:
            self.mass = min(MAX_FOOD_MASS, self.__mass + FOOD_GROWTH_RATE * FOOD_GROWTH_DELAY)
        self.time_until_growth -=1


class Creature(StageObject):
    """This class represents a Creature which can populate an Environment. It consists of several Organs, of which two
    Organs with a special purpose are the Body (which must always be present) and the Brain (Which is used for
    controlling a creatures behaviour). Each Creature has an age which counts the number of calls to Creature.tick().
    If a Creatures age is greater than genesis.MAX_AGE or it's mass is lower than get_mass_threshold() it dies.

    A Creatures position and orientation are stored in and managed by the body, which is accessible via get_body().

    A Creatures brain is accessible via get_brain()."""
    def __init__(self, body, name=None, generation=0):
        super().__init__()
        self.__organs = []
        self.__body = body
        self.generation = generation
        body.mass_listeners.append(self.__register_mass)
        body.position_listeners.append(self.__notify_position_listeners)
        body.rotation_listeners.append(self.__notify_rotation_listeners)
        self.__brain = None
        self.add_organ(body)
        self.__alive = True
        self.name = name
        self.organ_tick_cost = 0

        self.position_listeners = []
        self.rotation_listeners = []

    def highlight(self):
        self.body.highlight()

    def un_highlight(self):
        if self.alive:
            self.body.un_highlight()

    def __notify_position_listeners(self, old_position, new_position):
        for position_listener in self.position_listeners:
            position_listener(old_position, new_position)

    def __notify_rotation_listeners(self, old_rotation, new_rotation):
        for rotation_listener in self.rotation_listeners:
            rotation_listener(old_rotation, new_rotation)

    @property
    def alive(self):
        return self.__alive

    def __str__(self):
        return self.name if self.name is not None else "creature"

    def __repr__(self):
        return self.__str__()

    @property
    def brain(self):
        return self.__brain

    @brain.setter
    def brain(self, brain):
        if self.__brain is not None:
            self.remove_organ(self.__brain)
        self.add_organ(brain)

    def __register_mass(self, old_mass, new_mass):
        if new_mass < self.mass_threshold:
            self.kill()

    @property
    def tick_cost(self):
        organ_tick_cost = 0
        for organ in self.__organs:
            organ_tick_cost += organ.tick_cost
        return organ_tick_cost

    @property
    def mass_threshold(self):
        return self.tick_cost * 20

    def add_organ(self, organ, mutation_model=None):
        if organ not in self.__organs:
            if self.__brain is not None:
                self.__brain.wire_organ(organ, mutation_model)
            if type(organ) is Brain:
                self.__brain = organ
                for old_organ in self.__organs:
                    if old_organ is not self.__brain:
                        self.__brain.wire_organ(old_organ)
            self.__organs.append(organ)
            if organ.creature is None:
                organ.creature = self

    def remove_organ(self, organ):
        if self.__brain is not None:
            self.__brain.unwire_organ(organ)
        if isinstance(organ, Brain):
            for wired_organ in self.__organs:
                if wired_organ is not self.__brain:
                    self.__brain.unwire_organ(wired_organ)
        self.__organs.remove(organ)
        organ.creature = None

    def sense(self):
        for organ in self.__organs:
            organ.sense()

    def execute(self):
        for organ in self.__organs:
            organ.prepare_execute()
        for organ in self.__organs:
            organ.execute()
            if not self.__alive:
                    break

    def tick(self, tick_count):
        if self.age > MAX_CREATURE_AGE:
            self.kill()
        if self.__alive:
            self._last_tick_count = tick_count
            if self.__brain is not None:
                self.__brain.think_in_thread()
            organ_tick_cost = self.tick_cost
            self.mass -= organ_tick_cost
            self.age += 1

    @property
    def mass(self):
        return self.__body.mass

    @mass.setter
    def mass(self, amount):
        self.__body.mass = amount

    def add_mass(self, amount):
        self.__body.mass += amount

    @property
    def body(self):
        return self.__body

    @body.setter
    def body(self, body):
        # G todo: implement
        raise Exception("Changing a Creatures body is not yet implemented.")

    @property
    def center_x(self):
        return self.__body.center_x

    @property
    def center_y(self):
        return self.__body.center_y

    @property
    def organs(self):
        return self.__organs

    @property
    def center(self):
        return self.__body.center

    @center.setter
    def center(self, pos):
        self.__body.shape.center = pos

    def clone(self):
        cloned_creature = Creature(self.__body.clone(), self.name)
        for organ in self.__organs:
            if organ is not self.__body:
                cloned_creature.add_organ(organ.clone())

        cloned_creature.mass = self.mass
        cloned_creature.age = self.age
        cloned_creature.__brain.rewire(self.__brain)
        # G todo: remove variable _existing
        return cloned_creature

    def mutate(self, mutation_model, new_organ_mutation):
        if random.random() < mutation_model.mutation_likelihood:
            new_organ = None
            if random.random() < 0.5:
                new_organ = Mouth(random_from_interval(0.1, 6), random_from_interval(-10, 10))
            else:
                new_organ = EuclideanEye(random_from_interval(0.1, 30),
                                         random_from_interval(-10, 10),
                                         random_from_interval(0.1, 20))
            self.add_organ(new_organ, new_organ_mutation)
            new_organ.mutate(new_organ_mutation)
        if random.random() < mutation_model.mutation_likelihood:
            random_organ = self.__organs[random.randrange(0, len(self.__organs))]
            if random_organ is not self.__brain and random_organ is not self.__body:
                self.remove_organ(random_organ)
        for organ in self.__organs:
            organ.mutate(mutation_model)

    def kill(self):
        if self.__alive:
            if self.environment is not None:
                self.environment.remove_creature(self)
            self.__alive = False
            for organ in self.__organs:
                organ.kill()

    @property
    def graphics(self):
        graphics = []
        for organ in self.__organs:
            graphics += organ.graphics
        return graphics


class MonoColouredOrganGraphic(rendering.MonoColouredGraphic):
    def __init__(self, shape_retriever, colour):
        super().__init__(possibly_obstructed=True)
        self.__is_visible = True
        self.colour = colour
        self.shape_retriever = shape_retriever

    @property
    def fill_colour(self):
        return self.colour

    @property
    def shape(self):
        return self.shape_retriever()

    @property
    def is_visible(self):
        return self.__is_visible

    @is_visible.setter
    def is_visible(self, is_visible):
        if is_visible is not self.__is_visible:
            self.__is_visible = is_visible
            self.redraw_surface = True


class OutlinedOrganGraphic(rendering.OutlineGraphic):
    def __init__(self, shape_retriever, colour, border_width=1, is_visible=True):
        super().__init__(possibly_obstructed=True)
        self.__is_visible = is_visible
        self.__border_width = border_width
        self.colour = colour
        self.shape_retriever = shape_retriever

    @property
    def border_colour(self):
        return self.colour

    @property
    def shape(self):
        return self.shape_retriever()

    @property
    def border_width(self):
        return self.__border_width

    @property
    def is_visible(self):
        return self.__is_visible

    @is_visible.setter
    def is_visible(self, is_visible):
        if is_visible is not self.__is_visible:
            self.__is_visible = is_visible
            self.redraw_surface = True


class Organ:
    """This class represents an organ which can be used by creatures. It can be added to creatures which will then
    wire the organ to their brain to be used in the future.

    Organs can acquire information about their environment
    which are then fed into the Brain or get commands from the brain in the form of an arbitrary number of floating
    values which are then to be interpreted by the Brain.

    Use the methods Organ.register_input_neuron(Neuron) to add Neurons which acquire information and
    Organ.register_output_neuron(Neuron) to add Neurons which get commands from the brain."""
    def __init__(self, label=None, creature_position_listener=None, creature_rotation_listener=None):
        self.__input_neurons = []
        self.__output_neurons = []
        self.__creature = None
        self.creature_listeners = []
        self.label = label
        self.creature_position_listener = creature_position_listener
        self.creature_rotation_listener = creature_rotation_listener
        if creature_position_listener is not None:
            self.creature_listeners.append(self.__add_position_listeners_to_creature)
        if creature_rotation_listener is not None:
            self.creature_listeners.append(self.__add_rotation_listeners_to_creature)

    def __str__(self):
        return self.label if self.label is not None else "organ"

    def __repr__(self):
        return self.__str__()

    # def notify_graphic_listeners_of_change(self, *args):
    #     for graphic in self.graphics:
    #         graphic.redraw_surface = True

    def __add_position_listeners_to_creature(self, old_creature, new_creature):
        if new_creature is not None:
            new_creature.position_listeners.append(self.creature_position_listener)
        if old_creature is not None:
            old_creature.position_listeners.remove(self.creature_position_listener)

    def __add_rotation_listeners_to_creature(self, old_creature, new_creature):
        if new_creature is not None:
            new_creature.rotation_listeners.append(self.creature_rotation_listener)
        if old_creature is not None:
            old_creature.rotation_listeners.remove(self.creature_rotation_listener)

    @property
    def input_neurons(self):
        return self.__input_neurons

    @property
    def output_neurons(self):
        return self.__output_neurons

    @property
    def creature(self):
        return self.__creature

    @creature.setter
    def creature(self, creature):
        if creature is not self.__creature:
            if creature is not None:
                if self not in creature.organs:
                    creature.add_organ(self)
                if creature.environment and creature.environment.renderer is not None:
                    creature.environment.renderer.add_canvases(self.graphics)
            else:
                if self.__creature is not None and self.__creature.environment is not None and \
                                self.__creature.environment.renderer is not None:
                    self.__creature.environment.renderer.remove_canvases(self.graphics)
            old_creature = self.__creature
            self.__creature = creature
            for creature_listener in self.creature_listeners:
                creature_listener(old_creature, creature)

    def register_input_neuron(self, neuron):
        self.__input_neurons.append(neuron)

    def register_output_neuron(self, neuron):
        self.__output_neurons.append(neuron)

    def sense(self):
        """This method will give the organ time to acquire all necessary information which is needed to fill its
        input neuron which were present in Organ.input_neurons at the time it was wired to the brain."""
        pass

    def prepare_execute(self):
        pass

    def execute(self):
        """This method should be used to consume all relevant values from the output neurons which were present in
         Organ.output_neurons at the time it was wired to the brain."""
        pass

    def mutate(self, mutation_model):
        """Mutates this organ with the specified mutation_model."""
        pass

    def kill(self):
        """Kills this organ. Releases any running threads and closes any open streams"""
        pass

    @property
    def tick_cost(self):
        """This method specifies how much mass this organ costs to maintain each tick."""
        return 0

    # @property
    # def shape(self):
    #     """Returns the shape of this organ if it has any physical representation and None otherwise."""
    #     return None
    #
    # @shape.setter
    # def shape(self, shape):
    #     raise Exception("Organ " + str(type(self)) + " can't have a shape!")

    def clone(self):
        """Clones the organ and returns it. The cloned organ is not attached to any creature."""
        raise Exception("Clone not implemented for organ of type "+str(type(self)))

    @property
    def graphics(self):
        return []


class Neuron:
    def __init__(self, activation_function, label=None, is_bias=False):
        self.is_bias = is_bias
        self._label = label
        self._connections = {}
        self._fire_listeners = []
        self._activation_function = activation_function
        self._summed_input = 0
        self._last_amount = 0

    def __str__(self):
        return self._label if self._label is not None else "neuron"

    def __repr__(self):
        return self.__str__()

    @property
    def last_amount(self):
        return self._last_amount

    @property
    def label(self):
        return self._label

    def connect_to_layer(self, neurons, weight=0, mutation_model=None):
        for neuron in neurons:
            self.connect_to_neuron(neuron, weight, mutation_model)

    def disconnect_from_neuron(self, neuron):
        self._connections.pop(neuron)

    def connect_to_neuron(self, neuron, weight=0, mutation_model=None):
        if not neuron.is_bias:
            if mutation_model is not None and random.random() < mutation_model.mutation_likelihood:
                weight += random_from_interval(-mutation_model.mutation_strength, mutation_model.mutation_strength)
            self._connections[neuron] = weight

    def receive_fire(self, amount):
        self._summed_input += amount

    def fire(self):
        amount = self.consume()
        for fire_listener in self._fire_listeners:
            fire_listener.fired(amount)
        for target, weight in self._connections.items():
            target.receive_fire(weight * amount)

    def consume(self):
        self._last_amount = self._activation_function(self._summed_input)
        self._summed_input = 0
        return self._last_amount

    def has_weight(self, target_neuron):
        return target_neuron in self._connections

    def get_weight(self, target_neuron):
        return self._connections[target_neuron]


class InputNeuron(Neuron):
    def __init__(self, label=None):
        super().__init__(identity_activation, label)


class OutputNeuron(Neuron):
    def __init__(self, label=None):
        super().__init__(identity_activation, label)


class Brain(Organ):
    def __init__(self):
        super().__init__("brain")

        self.__input_layer = []
        self.__hidden_layer = []
        self.__output_layer = []
        self.__layers = [self.__input_layer, self.__hidden_layer, self.__output_layer]
        self.__bias_input_layer = Neuron(identity_activation, "input bias", True)
        self.__bias_hidden_layer = Neuron(identity_activation, "hidden bias", True)
        self.__bias_neurons = set()

        self.add_hidden_neuron(self.__bias_hidden_layer)
        self.add_input_neuron(self.__bias_input_layer)
        self.__is_thinking = False

    @property
    def layers(self):
        return self.__layers

    @property
    def input_layer(self):
        return self.__input_layer

    @property
    def hidden_layer(self):
        return self.__hidden_layer

    @property
    def output_layer(self):
        return self.__output_layer

    @property
    def bias_input_layer(self):
        return self.__bias_input_layer

    @property
    def bias_hidden_layer(self):
        return self.__bias_hidden_layer

    def clone(self):
        # cloned_brain = Brain()
        # cloned_layers = copy.deepcopy(self._layers)
        # cloned_brain._input_layer = cloned_layers[0]
        # cloned_brain.hidden_layer = cloned_layers[1]
        # cloned_brain._output_layer = cloned_layers[2]
        # cloned_brain._layers = cloned_layers
        # cloned_brain._bias_input_layer = cloned_layers[0][0]
        # cloned_brain.bias_hidden_layer = cloned_layers[1][0]
        return Brain()

    def rewire(self, other_brain):
        for layer_index in range(len(self.__layers)-1):
            from_layer_this = self.__layers[layer_index]
            from_layer_other = other_brain.layers[layer_index]
            to_layer_this = self.__layers[layer_index + 1]
            to_layer_other = other_brain.layers[layer_index+1]
            for from_neuron_this, from_neuron_other in zip(from_layer_this, from_layer_other):
                for to_neuron_this, to_neuron_other in zip(to_layer_this, to_layer_other):
                    if from_neuron_other.has_weight(to_neuron_other):
                        weight = from_neuron_other.get_weight(to_neuron_other)
                        from_neuron_this.connect_to_neuron(to_neuron_this, weight)

    def add_input_neuron(self, neuron, mutation_model=None):
        self.__input_layer.append(neuron)
        neuron.connect_to_layer(self.__hidden_layer, mutation_model=mutation_model)
        self.fill_hidden_layer(len(self.__input_layer))

    def add_output_neuron(self, neuron, mutation_model=None):
        self.__output_layer.append(neuron)
        Brain._connect_layer_to_neuron(self.__hidden_layer, neuron, mutation_model=mutation_model)
        self.fill_hidden_layer(len(self.__output_layer))

    def remove_input_neuron(self, input_neuron):
        self.__input_layer.remove(input_neuron)

    def remove_output_neuron(self, output_neuron):
        for hidden_neuron in self.__hidden_layer:
            hidden_neuron.disconnect_from_neuron(output_neuron)
        self.__output_layer.remove(output_neuron)

    def add_hidden_neuron(self, neuron=None):
        if neuron is None:
            neuron = Neuron(sigmoid_activation, "hidden " + str(len(self.__hidden_layer)))
        self.__hidden_layer.append(neuron)
        if not neuron.is_bias:
            Brain._connect_layer_to_neuron(self.__input_layer, neuron)
        neuron.connect_to_layer(self.__output_layer)

    def fill_hidden_layer(self, count):
        for i in range(len(self.__hidden_layer), count):
            self.add_hidden_neuron()

    @staticmethod
    def _connect_layer_to_neuron(layer, neuron, weight=0, mutation_model=None):
        for neuron_from in layer:
            neuron_from.connect_to_neuron(neuron, weight, mutation_model)

    @property
    def tick_cost(self):
        return len(self.__hidden_layer) / 100  # G todo: replace with realistic tick cost

    def think_in_thread(self):
        """Computes the values for the output neurons with the given inputs in the input neurons.

        This operation is non blocking"""
        # self.__lock_1.acquire()
        # self.__lock_0.release()
        self.think()

    def think(self):
        think_clock = self.creature.environment.clocks[THINKING_KEY]
        think_clock.tick()
        t = time.time()
        self.__is_thinking = True
        self.__bias_input_layer.receive_fire(1.)
        for input_neuron in self.__input_layer:
            input_neuron.fire()
        self.__bias_hidden_layer.receive_fire(1.)
        for hidden_neuron in self.__hidden_layer:
            hidden_neuron.fire()
        self.__is_thinking = False
        think_clock.tock()


    # def __think_for_thread(self):
    #     # tick = time.time()
    #     self.__lock_0.acquire()
    #     while self.creature.alive:
    #         self.think()
    #         self.__lock_1.release()
    #         self.__lock_0.acquire()

    @property
    def is_thinking(self):
        return self.__is_thinking

    def wire_organ(self, organ, mutation_model=None):
        """Wires an organ to this brain. All the organs input and output neurons are connected to the brains neural
        network. If a mutation model is given it is used to initialize the weights of the newly formed connections."""
        for input_neuron in organ.input_neurons:
            self.add_input_neuron(input_neuron, mutation_model)
        for output_neuron in organ.output_neurons:
            self.add_output_neuron(output_neuron, mutation_model)

    def unwire_organ(self, organ):
        """Undoes what wire_organ did. Removes all the organs input and output neurons from the brain along with all
        of its connections to other neurons in the neural network."""
        for input_neuron in organ.input_neurons:
            self.remove_input_neuron(input_neuron)
        for output_neuron in organ.output_neurons:
            self.remove_output_neuron(output_neuron)

    def mutate(self, mutation_model):
        """Mutates the brain and its neural weights. The mutation is determined by the given mutation_model. If a
        mutation of a weight occurs it will simply add a random delta (size depending on mutation_model) with an 80%
        chance. In the other 20% it will flip the sign of the weight."""
        likelihood = mutation_model.mutation_likelihood
        strength = mutation_model.mutation_strength * 2
        for layer_index in range(len(self.__layers)-1):
            from_layer = self.__layers[layer_index]
            to_layer = self.__layers[layer_index + 1]
            for neuron_from in from_layer:
                for neuron_to in to_layer:
                    if not neuron_to.is_bias:
                        weight = neuron_from.get_weight(neuron_to)
                        if random.random() < likelihood:
                            if random.random() < 0.2:
                                weight *= -1
                            else:
                                weight += random_from_interval(-strength, strength)
                        neuron_from.connect_to_neuron(neuron_to, weight)

    def kill(self):
        """Unlocks all locks in this organ so that the thinking thread can terminate."""
        # if self.__lock_0.locked():
        #     self.__lock_0.release()
        # if self.__lock_1.locked():
        #     self.__lock_1.release()


class EuclideanEye(Organ):
    def __init__(self, body_distance, rotation, radius):
        self.__field_of_view_graphic = OutlinedOrganGraphic(self.get_field_of_view_shape, (150, 150, 255, 255))
        self.__eye_graphic = MonoColouredOrganGraphic(self.get_eye_shape, colours.RED)
        # self.__graphics_listener = self.notify_graphic_listeners_of_change
        def repaint(x, y): self.__eye_graphic.redraw_surface = self.__field_of_view_graphic.redraw_surface = True
        super().__init__("eye", repaint, repaint)
        self.body_distance = body_distance
        self.rotation = rotation
        self.radius = radius
        self.food_pellets_spotted_count = 0
        self.__vision_neuron = InputNeuron("eye: vision")
        self.register_input_neuron(self.__vision_neuron)

    def sense(self):
        food = self.creature.environment.find_colliding_food(self.get_field_of_view_shape(), lambda x: False)
        self.food_pellets_spotted_count = len(food)
        self.__vision_neuron.receive_fire(sigmoid_activation(self.food_pellets_spotted_count))

    @property
    def vision_neuron(self):
        return self.__vision_neuron

    @property
    def pos(self):
        [dx, dy] = convert_to_delta_distance(self.body_distance, self.rotation + self.creature.body.rotation)
        return [self.creature.center_x + dx, self.creature.center_y + dy]

    def get_eye_shape(self):
        pos = self.pos
        return shapes.Circle(pos, 1)

    def get_field_of_view_shape(self):
        pos = self.pos
        return shapes.Circle(pos, self.radius)

    def clone(self):
        m = EuclideanEye(self.body_distance, self.rotation, self.radius)
        return m

    @property
    def tick_cost(self):
        return self.body_distance / 300 + self.radius ** 2 / 500

    @property
    def graphics(self):
        return [self.__eye_graphic, self.__field_of_view_graphic]


class Mouth(Organ):
    def __init__(self, body_distance, rotation, capacity=10., mouth_radius=2):
        self.__graphic = MonoColouredOrganGraphic(self.get_mouth_shape, colours.TEAL)

        def repaint(*args): self.__graphic.redraw_surface = True
        self.__graphics_listener = repaint
        super().__init__("mouth", self.__graphics_listener, self.__graphics_listener)
        self.__body_distance = body_distance
        self.__rotation = rotation
        self.__mouth_radius = mouth_radius
        self.__food_capacity = capacity
        self.__total_amount_eaten = 0
        self.body_distance_changed_listeners = []
        self.rotation_changed_listeners = []
        self.mouth_radius_changed_listeners = []
        self.food_capacity_changed_listeners = []
        self.__amount_eaten = 0
        # self._max_consumption = max_consumption
        self.__eat_neuron = OutputNeuron("mouth: eat")
        self.__has_eaten_neuron = InputNeuron("mouth: has eaten")
        self.register_output_neuron(self.__eat_neuron)
        self.register_input_neuron(self.__has_eaten_neuron)
        # self.__colliding_food = None
        # self.__colliding_food_lock_0 = threading.Lock()
        # self.__colliding_food_lock_0.acquire()
        # self.__colliding_food_lock_1 = threading.Lock()
        # threading.Thread(target=self.find_colliding_food).start()
        self.__max_mass = 0

        self.body_distance_changed_listeners.append(self.__graphics_listener)
        self.mouth_radius_changed_listeners.append(self.__graphics_listener)
        self.rotation_changed_listeners.append(self.__graphics_listener)

    @property
    def graphics(self):
        return [self.__graphic]

    @property
    def eat_neuron(self):
        return self.__eat_neuron

    @property
    def has_eaten_neuron(self):
        return self.__has_eaten_neuron

    @property
    def body_distance(self):
        return self.__body_distance

    @body_distance.setter
    def body_distance(self, body_distance):
        old_body_distance = self.__body_distance
        self.__body_distance = body_distance
        for listener in self.body_distance_changed_listeners:
            listener(old_body_distance, body_distance)

    @property
    def rotation(self):
        return self.__rotation

    @rotation.setter
    def rotation(self, rotation):
        old_rotation = self.__rotation
        self.__rotation = rotation
        for listener in self.rotation_changed_listeners:
            listener(old_rotation, rotation)

    @property
    def mouth_radius(self):
        return self.__mouth_radius

    @mouth_radius.setter
    def mouth_radius(self, mouth_radius):
        old_radius = self.__mouth_radius
        self.__mouth_radius = mouth_radius
        for listener in self.mouth_radius_changed_listeners:
            listener(old_radius, mouth_radius)

    @property
    def food_capacity(self):
        return self.__food_capacity

    @food_capacity.setter
    def food_capacity(self, food_capacity):
        old_food_capacity = self.__food_capacity
        self.__food_capacity = food_capacity
        for listener in self.food_capacity_changed_listeners:
            listener(old_food_capacity, food_capacity)

    def mutate(self, mutation_model):
        """Mutates a creatures mouth. This will mutate the parameters rotation, food_capacity, body_distance and
        mouth_radius."""
        strength = mutation_model.mutation_strength / 2
        mut_prob = mutation_model.mutation_likelihood
        if random.random() < mut_prob:
            r = random.random()
            if r < 0.3:
                self.rotation += random_from_interval(-90, 90)
            elif r < 0.5:
                self.rotation *= -1
            else:
                self.rotation *= random_from_interval(1 - strength, 1 + strength)
        if random.random() < mut_prob:
            self.food_capacity *= random_from_interval(1 - strength, 1 + strength)
        if random.random() < mut_prob:
            r = random.random()
            if r < 0.3:
                self.body_distance = max(0, self.body_distance + random_from_interval(-3, +3))
            else:
                self.body_distance *= random_from_interval(1 - strength, 1 + strength)
        if random.random() < mut_prob:
            self.mouth_radius *= random_from_interval(1 - strength, 1 + strength)
            # G todo: change area by random factor, not radius!

    def clone(self):
        m = Mouth(self.body_distance, self.rotation, self.food_capacity, self.mouth_radius)
        m.__amount_eaten = self.__amount_eaten
        return m

    @property
    def tick_cost(self):
        return self.body_distance / 100 + self.mouth_radius / 10 + self.food_capacity / 60
        # G todo: replace with realistic tick cost -> use mouth area and not just radius!

    @property
    def pos(self):
        if self.creature is None or self.creature.body is None:
            return 0, 0
        (dx, dy) = convert_to_delta_distance(self.body_distance, self.rotation + self.creature.body.rotation)
        return self.creature.center_x + dx, self.creature.center_y + dy

    def get_mouth_shape(self):
        pos = self.pos
        return shapes.Circle(pos, self.mouth_radius)

    def prepare_execute(self):
        pass
        # t = time.time()
        # eat_factor = clip(self.__eat_neuron.consume(), 0, 1)
        # self.__max_mass = eat_factor * self.food_capacity
        # if self.__max_mass > MIN_FOOD_MASS_TO_CONSUME:
        #     self.__colliding_food_lock_1.acquire()
        #     self.__colliding_food_lock_0.release()
        # else:
        #     self._amount_eaten = 0
        # self.creature.environment.time_collision_food += (time.time() - t)

    def execute(self):
        # if self.__colliding_food is not None:
            # self.__colliding_food_lock_1.acquire()
        eat_factor = clip(self.__eat_neuron.consume(), 0, 1)
        self.__max_mass = eat_factor * self.food_capacity
        if self.__max_mass > MIN_FOOD_MASS_TO_CONSUME:
            colliding_food = self.find_colliding_food()
            self.__amount_eaten = self.consume_food(self.__max_mass, colliding_food)
        else:
            self.__amount_eaten = 0
        self.__total_amount_eaten += self.__amount_eaten
        self.creature.mass += self.__amount_eaten

    @property
    def amount_eaten(self):
        return self.__amount_eaten

    @property
    def total_amount_eaten(self):
        return self.__total_amount_eaten

    def sense(self):
        self.__has_eaten_neuron.receive_fire(sigmoid_activation(self.__amount_eaten))

    def consume_food(self, max_mass, foods):
        eaten = 0
        remaining_capacity = max_mass
        for food in foods:
            if remaining_capacity < food.mass:
                food.mass -= remaining_capacity
                eaten += remaining_capacity
                break
            else:
                eaten += food.mass
                food.mass = 0
        return eaten

    def find_colliding_food(self):
        """Tries to consume Food which intersects with shape up the given maximum mass max_mass. The consumed mass
        is returned by this function and all Food that has been consumed will be removed from the Environment."""

        environment = self.creature.environment
        if environment is not None:
            def f(foods): return environment.sum_mass(foods) > self.food_capacity
            return environment.find_colliding_food(self.get_mouth_shape(), f)
        return None


class BodyGraphic(rendering.MonoColouredGraphic):
    def __init__(self, body, is_visible=True):
        super().__init__(possibly_obstructed=True)
        self.__is_visible = is_visible
        self.body = body

    @property
    def fill_colour(self):
        if self.body.creature is not None:
            age = self.body.creature.age
        else:
            age = MAX_CREATURE_AGE
        return colours.b(255 - 200 * (age / MAX_CREATURE_AGE))

    @property
    def shape(self):
        return self.body.shape

    @property
    def is_visible(self):
        return self.__is_visible

    @is_visible.setter
    def is_visible(self, is_visible):
        if is_visible is not self.__is_visible:
            self.__is_visible = is_visible
            self.redraw_surface = True


class Body(Organ):
    def __init__(self, mass, shape, angle=0, max_mass_burn=20):
        super().__init__("body")

        self._initial_mass = mass
        self.__shape = shape
        self._max_mass_burn = max_mass_burn

        self.__rotation = angle
        # G todo: add collision neuron
        self.__mass_neuron = InputNeuron("body: mass")
        self.register_input_neuron(self.__mass_neuron)
        self.__age_neuron = InputNeuron("body: age")
        self.register_input_neuron(self.__age_neuron)

        self.mass_listeners = []
        self.position_listeners = []
        self.rotation_listeners = []
        self.__mass = mass
        self.__graphic = BodyGraphic(self)
        self.__highlight_graphic = None

        def repaint(*args):
            self.__graphic.redraw_surface = True
            if self.__highlight_graphic is not None:
                self.__highlight_graphic.redraw_surface = True
        self.mass_listeners.append(repaint)
        self.position_listeners.append(repaint)
        self.rotation_listeners.append(repaint)

    def highlight(self):
        if self.__highlight_graphic is None:
            self.__highlight_graphic = rendering.SimpleOutlineGraphic(self.shape, colours.RED)
            self.creature.environment.add_canvas(self.__highlight_graphic, self.__highlight_graphic.position)

    def un_highlight(self):
        if self.__highlight_graphic is not None:
            self.creature.environment.remove_canvas(self.__highlight_graphic)
            self.__highlight_graphic = None

    def __notify_position_listeners(self, old_position, new_position):
        for position_listener in self.position_listeners:
            position_listener(old_position, new_position)

    def __notify_rotation_listeners(self, old_rotation, new_rotation):
        for rotation_listener in self.rotation_listeners:
            rotation_listener(old_rotation, new_rotation)

    @property
    def age(self):
        return self.creature.age

    @property
    def rotation(self):
        return self.__rotation

    @rotation.setter
    def rotation(self, rotation):
        old_rotation = self.__rotation
        self.__rotation = rotation
        self.__notify_rotation_listeners(old_rotation, rotation)

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, shape):
        self.__shape = shape

    def clone(self):
        b = Body(self.__mass, copy.deepcopy(self.shape), self.__rotation, self._max_mass_burn)
        b._initial_mass = self._initial_mass
        return b

    @property
    def mass_neuron(self):
        return self.__mass_neuron

    @property
    def age_neuron(self):
        return self.__age_neuron

    @property
    def mass(self):
        return self.__mass

    @mass.setter
    def mass(self, amount):
        old_mass = self.__mass
        amount = max(0, amount)
        self.shape.radius = math.sqrt(amount / 2)
        self.__mass = amount
        for mass_listener in self.mass_listeners:
            mass_listener(old_mass, amount)

    @property
    def center(self):
        return self.shape.center

    @center.setter
    def center(self, center):
        old_center = self.shape.center
        self.shape.center = center
        self.__notify_position_listeners(old_center, center)

    def move(self, dx, dy):
        old_shape = self.shape.center
        self.shape.translate(dx, dy)
        self.__notify_position_listeners(old_shape, self.shape.center)

    @property
    def center_x(self):
        return self.shape.center_x

    @center_x.setter
    def center_x(self, x):
        old_shape = self.shape.center
        self.shape.center_x = x
        self.__notify_position_listeners(old_shape, self.shape.center)

    @property
    def center_y(self):
        return self.shape.center_y

    @center_y.setter
    def center_y(self, y):
        old_shape = self.shape.center
        self.shape.center_y = y
        self.__notify_position_listeners(old_shape, self.shape.center)

    def sense(self):
        creature_age = self.creature.age
        self.__age_neuron.receive_fire(creature_age / MAX_CREATURE_AGE)
        self.__mass_neuron.receive_fire(self.__mass / self._initial_mass)

    def execute(self):
        pass

    @property
    def tick_cost(self):
        return self.__mass / 100  # G todo: replace with realistic tick cost -> a bigger body should cost more mass

    @property
    def graphics(self):
        to_return = [self.__graphic]
        if self.__highlight_graphic is not None:
            to_return.append(self.__highlight_graphic)
        return to_return


class Legs(Organ):
    # G should we rename this to fins? the creatures are moving around in water after all. or are they? do we want to
    # G model this aspect that closely?
    # G maybe we want some parameter which guides the max speed of legs and make faster legs more expensive to maintain?
    def __init__(self, max_distance=5, max_degree_turn=10):
        super().__init__("legs")
        self.max_distance = max_distance
        self.max_degree_turn = max_degree_turn
        self.total_distance_moved = 0
        self.__distance_moved = 0
        self.__distance_moved_neuron = InputNeuron("legs: distance moved")
        self.__forward_neuron = OutputNeuron("legs: forward")
        self.__turn_clockwise_neuron = OutputNeuron("legs: turn")
        self.register_input_neuron(self.__distance_moved_neuron)
        self.register_output_neuron(self.__forward_neuron)
        self.register_output_neuron(self.__turn_clockwise_neuron)

    @property
    def distance_moved_neuron(self):
        return self.__distance_moved_neuron

    @property
    def forward_neuron(self):
        return self.__forward_neuron

    @property
    def turn_clockwise_neuron(self):
        return self.__turn_clockwise_neuron

    def clone(self):
        return Legs(self.max_distance, self.max_degree_turn)

    def execute(self):
        travel_factor = clip(self.__forward_neuron.consume(), 0, 1)
        distance_to_travel = travel_factor * self.max_distance
        angle_factor = clip(self.__turn_clockwise_neuron.consume(), -1, 1)
        angle_to_turn = self.max_degree_turn * angle_factor
        self.creature.environment.turn_creature(self.creature, angle_to_turn)
        self.__distance_moved = self.creature.environment.move_creature(self.creature, distance_to_travel)
        self.total_distance_moved += self.__distance_moved
        mass_to_burn = distance_to_travel/200 + angle_to_turn/200  # G todo: replace with realistic formula
        self.creature.mass -= mass_to_burn

    def sense(self):
        self.__distance_moved_neuron.receive_fire(self.__distance_moved / self.max_distance)

    @property
    def tick_cost(self):
        return 0  # G todo: replace with realistic tick cost.


class Fission(Organ):
    # G maybe there could be more than one offspring? this could be a parameter which results in this organ being more
    # G expensive. if more than one offspring possible, this class needs to be renamed.
    # G also another parameter could be how much mass remains with the original creature and how much is
    # G transferred to the "offsprings"..
    def __init__(self, mutation_model, new_mutation_model):
        super().__init__("fission")
        self.new_mutation_model = new_mutation_model
        self.mutation_model = mutation_model
        self.__fission_neuron = OutputNeuron("fission: fission")
        self.register_output_neuron(self.__fission_neuron)
        self.offsprings_produced = 0

    def clone(self):
        return Fission(copy.deepcopy(self.mutation_model), copy.deepcopy(self.new_mutation_model))

    def execute(self):
        fission_value = self.__fission_neuron.consume()
        if fission_value > 0:
            creature = self.creature
            environment = creature.environment
            initial_mass = creature.body.mass
            new_mass = initial_mass * 0.5
            creature.body.mass = new_mass
            if creature.alive:
                split_creature = creature.clone()
                split_creature.age = 0
                split_creature.body.rotation = random.random()*360

                split_creature.__name = creature.name + "+"
                split_creature.generation = creature.generation + 1

                split_creature.mutate(self.mutation_model, self.new_mutation_model)
                self.offsprings_produced += 1
                if split_creature.alive:
                    environment.queue_creature(split_creature)

    @property
    def fission_neuron(self):
        return self.__fission_neuron

    @property
    def tick_cost(self):
        return 0
        # todo: replace with realistic tick cost


class MutationModel:
    # G do we want to apply different mutation levels to each organ? or should the parameter mutation_level just say how
    # G mutation in general occurs? after all we might want to mutate each parameter in each organ differently as well,
    # G so just passing different parameters to the organ help that much either..
    def __init__(self, mutation_likelihood, mutation_strength):
        self.mutation_likelihood = mutation_likelihood
        self.mutation_strength = mutation_strength

