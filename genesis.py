import math
import time, threading
import copy
import shapes
import random
import time
from msvcrt import getch
import numpy as np
import operator

# G todo: add some reusable way of mutating parameters. maybe using the Beta function?


def binary_activation(input_value):
    return 0 if input_value <= 0 else 1


def identity_activation(input_value):
    return input_value


def sigmoid_activation(input_value):
    return math.tanh(input_value)


def convert_to_delta_distance(distance, angle):
    rad = math.radians(angle)
    return [math.sin(rad)*distance, math.cos(rad)*distance]


def clip(number, min_value, max_value):
    return min(max(number, min_value), max_value)


class Food:
    def __init__(self, mass, shape):
        self._mass = mass
        self._shape = shape

    def reduce_mass(self, amount):
        self._mass -= amount

    def get_shape(self):
        return self._shape

    def get_mass(self):
        return self._mass


class Environment:
    def __init__(self, width=1000, height=1000):
        self._tick_count = 0
        self._creatures = []
        self._living_creatures = []
        self._food = set()
        self._width = width
        self._height = height
        self._running = True
        self._queued_creatures = []
        self._tick_listeners = []
        self._time_collision_food = 0
        self._time_collision_creatures = 0
        self._time_thinking = 0
        self._last_tick_time = -1

    def queue_creature(self, creature):
        self._queued_creatures.append(creature)

    def kill_create(self, creature):
        if creature._existing:
            self._living_creatures.remove(creature)
        else:
            self._queued_creatures.remove(creature)
        creature._alive = False

    def tick(self):
        for creature in self._queued_creatures:
            if creature._alive:
                self._living_creatures.append(creature)
                self._creatures.append(creature)
                creature.set_environment(self)
        self._queued_creatures.clear()
        for creature in self._living_creatures[:]:
            creature.sense()
        for creature in self._living_creatures[:]:
            creature.tick(self._tick_count)
        for creature in self._living_creatures[:]:
            creature.execute()
        self._tick_count += 1
        for listener in self._tick_listeners:
            listener(self)
        self._last_tick_time = time.time()

    def move_creature(self, creature, distance_to_travel):
        tick = time.time()
        [x, y] = [creature.get_x(), creature.get_y()]
        rotation = creature.get_body().get_rotation()
        [delta_x, delta_y] = convert_to_delta_distance(distance_to_travel, rotation)  # G todo: calculate new position using x, y, distance_to_travel & rotation
        [new_x, new_y] = [x + delta_x, y + delta_y]
        translated_shape = copy.deepcopy(creature.get_body().get_shape())
        translated_shape.set_pos(new_x, new_y)
        # G todo: the following only computes if the the new position is valid.
        # G todo- it doesn't calculate how far the object should actually move instead!
        is_valid = True
        [width, height] = creature.get_body().get_shape().get_dimensions()
        if new_x+width/2 > self._width or new_x-width/2 < 0 or new_y+height/2 > self._height or new_y-height/2 < 0:
            is_valid = False
        # if is_valid:
        #     for other_creature in self._living_creatures:
        #         if other_creature is not creature and translated_shape.collides(other_creature.get_body().get_shape()):
        #             is_valid = False
        #             #print("collision detected between "+other_creature._name+ " and "+creature._name)
        #             break
        if is_valid:
            creature.get_body().set_position(new_x, new_y)
        self._time_collision_creatures += (time.time()-tick)

    def turn_creature(self, creature, angle_to_turn):
        creature.get_body().set_rotation(creature.get_body().get_rotation()+angle_to_turn)

    #def to_absolute_position(self, creature, dx, dy):
    #    return [dx, dy]  # G todo: return actual absolute position

    def consume_food(self, shape, max_mass):
        tick = time.time()
        eaten = 0
        food_eaten = set()
        capacity_available = max_mass
        for food in self._food:
            if shape.collides(food.get_shape()):
                if capacity_available < food.get_mass():
                    food.reduce_mass(capacity_available)
                    eaten += capacity_available
                    break
                else:
                    eaten += food.get_mass()
                    food_eaten.add(food)
        #self._food.difference_update(food_eaten)  # = [filter(lambda f: f not in food_eaten, self._food)]
        self._food -= food_eaten  # = [filter(lambda f: f not in food_eaten, self._food)]
        self._time_collision_food += (time.time()-tick)
        return eaten

    def create_food(self, x, y, mass, radius):
        self._food.add(Food(mass, shapes.Circle(x, y, radius)))


class Creature:
    def __init__(self, body, name=None):
        self._organs = []
        self._body = body
        self._brain = None
        self._environment = None
        self.add_organ(body)
        self._alive = body.get_energy() > 0
        self._existing = False
        self._name = name
        self._last_tick_count = -1

    def __str__(self):
        return self._name if self._name is not None else "creature"

    def __repr__(self):
        return self.__str__()

    def add_organ(self, organ):
        if self._brain is not None:
            self._brain.wire_organ(organ)
        if type(organ) is Brain:
            self._brain = organ
            for old_organ in self._organs:
                if old_organ is not self._brain:
                    self._brain.wire_organ(old_organ)
        self._organs.append(organ)

        organ._creature = self

    def sense(self):
        for organ in self._organs:
            organ.sense()

    def execute(self):
        for organ in self._organs:
            if self._body.get_energy() < 0:
                break
            organ.execute()

    def tick(self, tick_count):
        self._last_tick_count = tick_count
        #print("Creature.tick()\tCreature.name()="+self._name)
        #print("\tCreature._organs = "+str(self._organs))
        #print("\tCreature._body.get_mass() = "+str(self._body.get_mass()))
        #print("\tCreature._body.get_energy() = "+str(self._body.get_energy()))
        #print("\tCreature._body.get_pos() = ["+str(self.get_x())+", "+str(self.get_y())+"]")
        #print("\tCreature._body.get_rotation() = "+str(self._body.get_rotation()))
        if self._brain is not None:
            tick = time.time()
            self._brain.think()
            self._environment._time_thinking += (time.time() - tick)
        tick_cost_summed = 0
        for organ in self._organs:
            tick_cost_summed += organ.tick_cost()
        self.decrease_energy(tick_cost_summed)

    def get_environment(self):
        return self._environment

    def get_body(self):
        return self._body

    def get_mass(self):
        return self._body.get_mass()

    def get_energy(self):
        return self._body.get_energy()

    def get_x(self):
        return self.get_body().get_x()

    def get_y(self):
        return self.get_body().get_y()

    def set_environment(self, environment):
        self._environment = environment
        self._existing = environment is not None

    def decrease_energy(self, amount):
        self._body.set_energy(self._body.get_energy()-amount)

    def set_energy(self, amount):
        self._body.set_energy(amount)

    def get_organs(self):
        return self._organs

    def set_pos(self, x, y):
        self._body.get_shape().set_pos(x, y)

    def clone(self):
        memo_dict = {id(self._environment): None}
        to_return = copy.deepcopy(self, memo_dict)
        to_return._existing = False # G todo: remove variable _existing
        return to_return

    def mutate(self, mutation_model):
        for organ in self._organs:
            organ.mutate(mutation_model)

    def kill(self):
        if self._alive:
            self._environment.kill_create(self)
            #print(self._name + " died at time="+str(self._environment._tick_count))

    def get_name(self):
        return self._name


class Organ:
    def __init__(self, label=None):
        self._input_neurons = []
        self._output_neurons = []
        self._creature = None
        self._label = label

    def __str__(self):
        return self._label if self._label is not None else "organ"

    def __repr__(self):
        return self.__str__()

    def register_input_neuron(self, neuron):
        self._input_neurons.append(neuron)

    def register_output_neuron(self, neuron):
        self._output_neurons.append(neuron)

    def get_input_neurons(self):
        return self._input_neurons

    def get_output_neurons(self):
        return self._output_neurons

    def has_label(self):
        return self._label is not None

    def get_label(self):
        return self._label

    def sense(self):
        pass

    def execute(self):
        pass

    def mutate(self, mutation_level):
        pass

    def tick_cost(self):
        return 0

    def get_shape(self):
        pass


class Neuron:
    def __init__(self, activation_function, label=None):
        self._label = label
        self._connections = {}
        self._fire_listeners = []
        self._activation_function = activation_function
        self._summed_input = 0

    def __str__(self):
        return self._label if self._label is not None else "neuron"

    def __repr__(self):
        return self.__str__()

    def connect_to_layer(self, neurons, weight=0):
        for neuron in neurons:
            self._connections[neuron] = weight

    def connect_to_neuron(self, neuron, weight=0):
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
        amount = self._activation_function(self._summed_input)
        self._summed_input = 0
        return amount

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

        self._input_layer = []
        self._hidden_layer = []
        self._output_layer = []

        self._bias_input_layer = Neuron(identity_activation, "input bias")
        self._bias_hidden_layer = Neuron(identity_activation, "hidden bias")

        self.add_input_neuron(self._bias_input_layer)
        self.add_hidden_neuron(self._bias_hidden_layer)

    def add_input_neuron(self, neuron):
        self._input_layer.append(neuron)
        neuron.connect_to_layer(self._hidden_layer)
        self.fill_hidden_layer(len(self._input_layer))

    def add_output_neuron(self, neuron):
        self._output_layer.append(neuron)
        Brain._connect_layer_to_neuron(self._hidden_layer, neuron)
        self.fill_hidden_layer(len(self._output_layer))

    def add_hidden_neuron(self, neuron=None):
        if neuron is None:
            neuron = Neuron(sigmoid_activation, "hidden "+str(len(self._hidden_layer)))
        self._hidden_layer.append(neuron)
        Brain._connect_layer_to_neuron(self._input_layer, neuron)
        neuron.connect_to_layer(self._output_layer)

    def fill_hidden_layer(self, count):
        for i in range(len(self._hidden_layer), count):
            self.add_hidden_neuron()

    @staticmethod
    def _connect_layer_to_neuron(layer, neuron):
        for neuron_from in layer:
            neuron_from.connect_to_neuron(neuron)

    def tick_cost(self):
        return len(self._hidden_layer) / 10  # G todo: replace with realistic tick cost

    def think(self):
        #tick = time.time()
        self._bias_input_layer.receive_fire(1.)
        for input_neuron in self._input_layer:
            input_neuron.fire()
        self._bias_hidden_layer.receive_fire(1.)
        for hidden_neuron in self._hidden_layer:
            hidden_neuron.fire()
        #for output_neuron in self._output_layer:
        #    output_neuron.fire()

    def wire_organ(self, organ):
        for input_neuron in organ.get_input_neurons():
            self.add_input_neuron(input_neuron)
        for output_neuron in organ.get_output_neurons():
            self.add_output_neuron(output_neuron)

    def mutate(self, mutation_level):
        likelihood = mutation_level._mutation_likelihood
        strength = mutation_level._mutation_strength * 2
        for neuron_from in self._input_layer:
            for neuron_to in self._hidden_layer:
                weight = neuron_from.get_weight(neuron_to)
                if random.random() < likelihood:
                    weight += random.random()*strength-strength/2
                    neuron_from.connect_to_neuron(neuron_to, weight)
        for neuron_from in self._hidden_layer:
            for neuron_to in self._output_layer:
                weight = neuron_from.get_weight(neuron_to)
                if random.random() < likelihood:
                    weight += random.random()*strength-strength/2
                    neuron_from.connect_to_neuron(neuron_to, weight)


class Mouth(Organ):
    def __init__(self, body_distance, rotation, capacity=10., mouth_radius=2):
        super().__init__("mouth")
        self._body_distance = body_distance
        self._rotation = rotation
        self._amount_eaten = 0
        self._food_capacity = capacity
        self._mouth_radius = mouth_radius
        #self._max_consumption = max_consumption
        self._eat_neuron = OutputNeuron("mouth: eat")
        self._has_eaten_neuron = InputNeuron("mouth: has eaten")
        self.register_output_neuron(self._eat_neuron)
        self.register_input_neuron(self._has_eaten_neuron)

    def tick_cost(self):
        return self._body_distance  # todo: replace with realistic tick cost

    def get_pos(self):
        [dx, dy] = convert_to_delta_distance(self._body_distance, self._rotation)
        return [self._creature.get_x() + dx, self._creature.get_y() + dy]

    def get_shape(self):
        pos = self.get_pos()
        return shapes.Circle(pos[0], pos[1], self._mouth_radius)

    def execute(self):
        environment = self._creature.get_environment()
        eat_factor = clip(self._eat_neuron.consume(), 0, 1)
        max_mass = eat_factor * self._food_capacity
        self._amount_eaten = environment.consume_food(self.get_shape(), max_mass)
        self._creature.get_body().add_mass(self._amount_eaten)


class Body(Organ):
    def __init__(self, energy, mass, shape, angle=0, max_mass_burn=20):
        super().__init__("body")

        self._energy = 0
        self._initial_energy = energy
        self.set_energy(energy)

        self._mass = 0
        self._initial_mass = mass
        self.set_mass(mass)
        self._max_mass_burn = max_mass_burn

        self._total_mass_gained = 0

        self._rotation = angle
        self._shape = shape
        # G todo: add collision neuron
        self._mass_neuron = InputNeuron("body: mass")
        self._energy_neuron = InputNeuron("body: energy")
        self._burn_mass_neuron = OutputNeuron("body: burn mass")

        self.register_input_neuron(self._mass_neuron)
        self.register_input_neuron(self._energy_neuron)
        self.register_output_neuron(self._burn_mass_neuron)

    def get_mass(self):
        return self._mass

    def set_mass(self, amount):
        self._mass = amount
        self._total_mass_gained = 0
        # G todo: change shape size to reflect mass change

    def add_mass(self, amount):
        self._mass += amount
        self._total_mass_gained += amount

    def get_energy(self):
        return self._energy

    def set_energy(self, new_energy):
        self._energy = new_energy
        if self._energy < 0:
            self._creature.kill()

    def burn_mass(self, mass_to_burn):
        mass_to_burn = min(mass_to_burn, self._mass)
        energy_gained = mass_to_burn*5
        # G todo: formula for burning mass here. should be less effective if there is already a lot of energy
        self._energy += energy_gained
        self._mass -= mass_to_burn

    def set_rotation(self, angle):
        self._rotation = angle

    def get_rotation(self):
        return self._rotation

    def set_position(self, x, y):
        self._shape._x = x
        self._shape._y = y

    def move(self, dx, dy):
        self._shape._x += dx
        self._shape._y += dy

    def get_x(self):
        return self._shape.get_x()

    def get_y(self):
        return self._shape.get_y()

    def sense(self):
        self._mass_neuron.receive_fire(self._mass/self._initial_mass)
        self._energy_neuron.receive_fire(self._energy/self._initial_energy)

    def execute(self):
        mass_burn_factor = clip(self._burn_mass_neuron.consume(), 0, 1)
        amount_to_burn = mass_burn_factor * self._max_mass_burn
        self.burn_mass(amount_to_burn)

    def tick_cost(self):
        return self._mass/100  # G todo: replace with realistic tick cost -> a bigger body should cost more energy

    def get_shape(self):
        return self._shape


class Legs(Organ):
    # G should we rename this to fins? the creatures are moving around in water after all. or are they? do we want to
    # G model this aspect that closely?
    # G maybe we want some parameter which guides the max speed of legs and make faster legs more expensive to maintain?
    def __init__(self, max_speed=5, max_degree_turn=10):
        super().__init__("legs")
        self._max_speed = max_speed
        self._max_degree_turn = max_degree_turn
        self._forward_neuron = OutputNeuron("Legs: forward")
        self._turn_clockwise_neuron = OutputNeuron("Legs: turn")
        self.register_output_neuron(self._forward_neuron)
        self.register_output_neuron(self._turn_clockwise_neuron)

    def execute(self):
        travel_factor = clip(self._forward_neuron.consume(), 0, 1)
        distance_to_travel = travel_factor * self._max_speed
        angle_factor = clip(self._turn_clockwise_neuron.consume(), -1, 1)
        angle_to_turn = self._max_degree_turn * angle_factor
        self._creature.get_environment().turn_creature(self._creature, angle_to_turn)
        self._creature.get_environment().move_creature(self._creature, distance_to_travel)
        energy_to_use = distance_to_travel/50 + angle_to_turn/50  # G todo: replace with realistic formula
        self._creature.decrease_energy(energy_to_use)

    def get_forward_neuron(self):
        return self._forward_neuron

    def get_turn_clockwise_neuron(self):
        return self._turn_clockwise_neuron

    def tick_cost(self):
        return 0  # G todo: replace with realistic tick cost.


class Fission(Organ):
    # G maybe there could be more than one offspring? this could be a parameter which results in this organ being more
    # G expensive. if more than one offspring possible, this class needs to be renamed.
    # G also another parameter could be how much energy remains with the original creature and how much is
    # G transferred to the "offsprings"..
    def __init__(self, mutation_model):
        super().__init__("fission")
        self._mutation_model = mutation_model
        self._fission_neuron = OutputNeuron("Fission: fission")
        self.register_output_neuron(self._fission_neuron)

    def execute(self):
        fission_value = self._fission_neuron.consume()
        if fission_value > 0:  # G todo: reenable this
            initial_energy = self._creature.get_body().get_energy()
            new_energy = initial_energy * 0.4
            initial_mass = self._creature.get_body().get_mass()
            new_mass = initial_mass * 0.4
            self._creature.get_body().set_energy(new_energy)
            self._creature.get_body().set_mass(new_mass)

            split_creature = self._creature.clone()

            split_creature._name = self._creature._name + "+"

            split_creature.mutate(self._mutation_model)
            self._creature.get_environment().queue_creature(split_creature)
            #print("creature "+self._creature._name+" split itself")

    def tick_cost(self):
        return 0
        # todo: replace with realistic tick cost


class MutationModel:
    # G do we want to apply different mutation levels to each organ? or should the parameter mutation_level just say how
    # G mutation in general occurs? after all we might want to mutate each parameter in each organ differently as well,
    # G so just passing different parameters to the organ help that much either..
    def __init__(self, mutation_likelihood, mutation_strength):
        self._mutation_likelihood = mutation_likelihood
        self._mutation_strength = mutation_strength

