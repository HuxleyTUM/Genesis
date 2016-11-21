import math
import time, threading
import copy
import shapes
import random
import time
import numpy as np
import operator

# G todo: add some reusable way of mutating parameters. maybe using the Beta function?
max_age = 250

def random_pos(width, height, borders=0):
    return [random_from_interval(borders, width-borders), random_from_interval(borders, height-borders)]


def random_from_interval(min, max):
    return random.random()*(max-min)+min


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
        self.initial_mass = mass
        self.set_mass(mass)
        self.environment = None

    def reduce_mass(self, amount):
        self.set_mass(self._mass-amount)

    def set_mass(self, amount):
        self._mass = amount
        self._shape._radius = math.sqrt(self._mass/2)
        if self._mass < 0.05:
            self.kill()

    def kill(self):
        self.environment.remove_food(self)

    def get_shape(self):
        return self._shape

    def get_mass(self):
        return self._mass

    def tick(self):
        self.set_mass(min(self.initial_mass, self._mass+0.1))


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
        self.last_tick_time = -1
        self.last_tick_delta = -1

    def queue_creature(self, creature):
        self._queued_creatures.append(creature)

    def kill_create(self, creature):
        if creature._existing:
            self._living_creatures.remove(creature)
        else:
            self._queued_creatures.remove(creature)
        creature._alive = False

    def tick(self):
        for food in self._food:
            food.tick()
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
        current_time = time.time()
        self.last_tick_delta = current_time - self.last_tick_time
        self.last_tick_time = current_time

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
        if(max_mass > 0.05):
            tick = time.time()
            eaten = 0
            remaining_capacity = max_mass
            for food in copy.copy(self._food):
                if shape.collides(food.get_shape()):
                    if remaining_capacity < food.get_mass():
                        food.reduce_mass(remaining_capacity)
                        eaten += remaining_capacity
                        break
                    else:
                        eaten += food.get_mass()
                        food.set_mass(0)
            #self._food.difference_update(food_eaten)  # = [filter(lambda f: f not in food_eaten, self._food)]
            self._time_collision_food += (time.time()-tick)
            return eaten
        else:
            return 0

    def create_food(self, x, y, mass, radius):
        self.add_food(Food(mass, shapes.Circle(x, y, radius)))

    def add_food(self, food):
        self._food.add(food)
        food.environment = self

    def remove_food(self, food):
        self._food.remove(food)

class Creature:
    def __init__(self, body, name=None):
        self._organs = []
        self._body = body
        body.mass_listeners.append(self._register_mass)
        self._brain = None
        self._environment = None
        self.add_organ(body)
        self._alive = True
        self._existing = False
        self._name = name
        self._last_tick_count = -1
        self.age = 0
        self.organ_tick_cost = 0

    def __str__(self):
        return self._name if self._name is not None else "creature"

    def __repr__(self):
        return self.__str__()

    def _register_mass(self, amount):
        if amount < self.organ_tick_cost*20:
            self.kill()

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
            if not self._alive:
                break
            organ.execute()

    def tick(self, tick_count):
        self.age += 1
        if self.age > max_age:
            self.kill()
        self._last_tick_count = tick_count
        #print("Creature.tick()\tCreature.name()="+self._name)
        #print("\tCreature._organs = "+str(self._organs))
        #print("\tCreature._body.get_mass() = "+str(self._body.get_mass()))
        #print("\tCreature._body.get_pos() = ["+str(self.get_x())+", "+str(self.get_y())+"]")
        #print("\tCreature._body.get_rotation() = "+str(self._body.get_rotation()))
        if self._brain is not None:
            tick = time.time()
            self._brain.think()
            self._environment._time_thinking += (time.time() - tick)
        self.organ_tick_cost = 0
        for organ in self._organs:
            self.organ_tick_cost += organ.tick_cost()
        self.add_mass(-self.organ_tick_cost)

    def get_environment(self):
        return self._environment

    def get_body(self):
        return self._body

    def get_mass(self):
        return self._body.get_mass()

    def set_mass(self, amount):
        self._body.set_mass(amount)

    def add_mass(self, amount):
        self._body.add_mass(amount)

    def get_x(self):
        return self.get_body().get_x()

    def get_y(self):
        return self.get_body().get_y()

    def set_environment(self, environment):
        self._environment = environment
        self._existing = environment is not None

    def get_organs(self):
        return self._organs

    def set_pos(self, x, y):
        self._body.get_shape().set_pos(x, y)

    def clone(self):
        cloned_creature = Creature(self._body.clone(), self._name)
        for organ in self._organs:
            if organ is not self._body:
                cloned_creature.add_organ(organ.clone())
        cloned_creature.set_mass(self.get_mass())
        cloned_creature.age = self.age
        # G todo: remove variable _existing
        return cloned_creature

    def mutate(self, mutation_model):
        for organ in self._organs:
            organ.mutate(mutation_model)

    def kill(self):
        if self._alive:
            self._environment.kill_create(self)
            self._alive = False

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

    def __deepcopy__(self, memodict={}):
        return self.clone()

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

    def clone(self):
        print("invalid call in "+str(type(self)))
        a = 1/0


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
        self._layers = [self._input_layer, self._hidden_layer, self._output_layer]

        self._bias_input_layer = Neuron(identity_activation, "input bias")
        self._bias_hidden_layer = Neuron(identity_activation, "hidden bias")

        self.add_input_neuron(self._bias_input_layer)
        self.add_hidden_neuron(self._bias_hidden_layer)
        self.t = threading.Thread(target=self.think_for_thread)
        self.lock_0 = threading.Lock()
        self.lock_1 = threading.Lock()
        # print("acquire lock 0")
        self.lock_0.acquire()
        self.t.start()

    def clone(self):
        cloned_brain = Brain()
        cloned_layers = copy.deepcopy(self._layers)
        cloned_brain._input_layer = cloned_layers[0]
        cloned_brain._hidden_layer = cloned_layers[1]
        cloned_brain._output_layer = cloned_layers[2]
        cloned_brain._layers = cloned_layers
        cloned_brain._bias_input_layer = cloned_layers[0][0]
        cloned_brain._bias_hidden_layer = cloned_layers[1][0]
        return cloned_brain

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
        return len(self._hidden_layer) / 100  # G todo: replace with realistic tick cost

    def think(self):
        # print("releasing lock..")
        self.lock_1.acquire()
        self.lock_0.release()

    def think_for_thread(self):
        # tick = time.time()
        while True:
            # print("acquire lock 1")
            self.lock_0.acquire()
            self._bias_input_layer.receive_fire(1.)
            for input_neuron in self._input_layer:
                input_neuron.fire()
            self._bias_hidden_layer.receive_fire(1.)
            for hidden_neuron in self._hidden_layer:
                hidden_neuron.fire()
            self.lock_1.release()
                # for output_neuron in self._output_layer:
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

    def mutate(self, mutation_level):
        strength = mutation_level._mutation_strength/2
        mut_prob = mutation_level._mutation_likelihood
        if random.random() < mut_prob:
            self._food_capacity *= random_from_interval(1-strength, 1+strength)
        if random.random() < mut_prob:
            self._body_distance *= random_from_interval(1-strength, 1+strength)
        if random.random() < mut_prob:
            self._mouth_radius *= random_from_interval(1-strength, 1+strength)
            # G todo: change area by random factor, not radius!

    def clone(self):
        m = Mouth(self._body_distance, self._rotation, self._food_capacity, self._mouth_radius)
        m._amount_eaten = self._amount_eaten
        return m

    def tick_cost(self):
        return self._body_distance/60 + (self._mouth_radius)**2/20 + self._food_capacity/50
        # G todo: replace with realistic tick cost -> use mouth area and not just radius!

    def get_pos(self):
        [dx, dy] = convert_to_delta_distance(self._body_distance, self._rotation+self._creature.get_body()._rotation)
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
    def __init__(self, mass, shape, angle=0, max_mass_burn=20):
        super().__init__("body")

        self._mass = 0
        self._initial_mass = mass
        self._shape = shape
        self._max_mass_burn = max_mass_burn

        self._rotation = angle
        # G todo: add collision neuron
        self._mass_neuron = InputNeuron("body: mass")
        self.register_input_neuron(self._mass_neuron)
        self._age_neuron = InputNeuron("body: age")
        self.register_input_neuron(self._age_neuron)

        self.mass_listeners = []
        self.set_mass(mass)

    def clone(self):
        b = Body(self._mass, copy.deepcopy(self._shape), self._rotation, self._max_mass_burn)
        b._initial_mass = self._initial_mass
        return b

    def get_mass(self):
        return self._mass

    def set_mass(self, amount):
        self._mass = amount
        self._shape.set_radius(math.sqrt(amount/2))
        for mass_listener in self.mass_listeners:
            mass_listener(self._mass)

        # G todo: change shape size to reflect mass change

    def add_mass(self, amount):
        self.set_mass(self._mass + amount)

    def set_rotation(self, angle):
        self._rotation = angle

    def get_rotation(self):
        return self._rotation

    def set_position(self, x, y):
        self._shape._x = x
        self._shape._y = y

    def move(self, dx, dy):
        self._shape.translate(dx, dy)

    def get_x(self):
        return self._shape.get_x()

    def get_y(self):
        return self._shape.get_y()

    def sense(self):
        self._age_neuron.receive_fire(self._creature.age/max_age)
        self._mass_neuron.receive_fire(self._mass/self._initial_mass)

    def execute(self):
        pass

    def tick_cost(self):
        return self._mass/100  # G todo: replace with realistic tick cost -> a bigger body should cost more mass

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

    def clone(self):
        return Legs(self._max_speed, self._max_degree_turn)

    def execute(self):
        travel_factor = clip(self._forward_neuron.consume(), 0, 1)
        distance_to_travel = travel_factor * self._max_speed
        angle_factor = clip(self._turn_clockwise_neuron.consume(), -1, 1)
        angle_to_turn = self._max_degree_turn * angle_factor
        self._creature.get_environment().turn_creature(self._creature, angle_to_turn)
        self._creature.get_environment().move_creature(self._creature, distance_to_travel)
        mass_to_burn = distance_to_travel/200 + angle_to_turn/200  # G todo: replace with realistic formula
        self._creature.add_mass(mass_to_burn)

    def get_forward_neuron(self):
        return self._forward_neuron

    def get_turn_clockwise_neuron(self):
        return self._turn_clockwise_neuron

    def tick_cost(self):
        return 0  # G todo: replace with realistic tick cost.


class Fission(Organ):
    # G maybe there could be more than one offspring? this could be a parameter which results in this organ being more
    # G expensive. if more than one offspring possible, this class needs to be renamed.
    # G also another parameter could be how much mass remains with the original creature and how much is
    # G transferred to the "offsprings"..
    def __init__(self, mutation_model):
        super().__init__("fission")
        self._mutation_model = mutation_model
        self._fission_neuron = OutputNeuron("Fission: fission")
        self.register_output_neuron(self._fission_neuron)

    def clone(self):
        return Fission(copy.deepcopy(self._mutation_model))

    def execute(self):
        fission_value = self._fission_neuron.consume()
        if fission_value > 0:
            initial_mass = self._creature.get_body().get_mass()
            new_mass = initial_mass * 0.6
            self._creature.get_body().set_mass(new_mass)

            split_creature = self._creature.clone()
            split_creature.age = 0

            split_creature._name = self._creature._name + "+"

            split_creature.mutate(self._mutation_model)
            self._creature.get_environment().queue_creature(split_creature)
            print("creature "+self._creature._name+" split itself")

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

