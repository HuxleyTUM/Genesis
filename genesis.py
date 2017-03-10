import math
import time, threading
import copy
import shapes
import random
import time
import numpy as np
import operator
import binary_tree
import clock

# G todo: add some reusable way of mutating parameters. maybe using the Beta function?
THINKING_KEY = "thinking"
TICK_KEY = "tick"
FOOD_COLLISION_KEY = "food collision"
RENDER_KEY = "render"
RENDER_THREAD_KEY = "render thread"
FOOD_RECLASSIFICATION_KEY = "food reclassification"

MIN_FOOD_MASS_TO_CONSUME = 0.05
FOOD_GROWTH_RATE = 0.4
MAX_AGE = 250
MAX_FOOD_MASS = 100


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


class Food:
    """This class represents a piece of Food. It can be placed in the Environment and consumed by Creatures."""
    def __init__(self, mass, shape):
        self.__shape = shape
        self.__mass = mass
        self.__environment = None

    @property
    def mass(self):
        """Returns the Foods mass."""
        return self.__mass

    @mass.setter
    def mass(self, amount):
        """Sets the mass of this piece of Food."""
        self.__mass = amount
        self.__shape.radius = math.sqrt(self.__mass / 2)
        if self.__mass < 0.05:
            self.kill()

    @property
    def environment(self):
        return self.__environment

    @environment.setter
    def environment(self, environment):
        self.__environment = environment
        if environment is not None and not environment.food_tree.contains(self, self.__shape):
            environment.add_food(self)

    def kill(self):
        """Destroys this piece of Food, removing it from its Environment."""
        if self.__environment is not None:
            self.__environment.remove_food(self)

    @property
    def shape(self):
        """Returns the physical representation of this piece of Food."""
        return self.__shape

    def tick(self):
        """This method should be called, each time a virtual time unit (tick) has passed."""
        self.mass = min(MAX_FOOD_MASS, self.__mass + FOOD_GROWTH_RATE)


class Environment:
    """This class represents the environment in which Creatures live. Not only does it manage the creatures living in
    it but also the Food which is meant to be consumed by the Creatures. The environment has no real sense of time. It
    has to be controlled from the outside via its Environment.tick() method. Whenever this method is called, the world
    "continues". Although the environment has no real sense of time, it has a variable tick_count, which counts up one
    each call to Environment.tick() and represents its internal time keeping.

    Creatures can not simply be added to the environment at any time. They need to be queued with
    Environment.queue_creature(creature) which will then be added on the next call to Environment.tick().

    Creatures in the world can not decide for themselves how they can move around. They need to make call the method
    move_creature(creature, distance_to_travel)."""
    def __init__(self, width=1000, height=1000):
        self.__tick_count = 0
        self.__stage_objects = []
        self.__creatures = []
        self.__living_creatures = []
        # self.__food_pellets = set()
        self.__food_tree = binary_tree.BinaryTree(width, height, 6)
        self.__width = width
        self.__height = height
        self.__queued_creatures = []
        self.__tick_listeners = []
        self.__last_tick_time = -1
        self.__last_tick_delta = -1
        self.time = {FOOD_COLLISION_KEY: clock.Clock(), TICK_KEY: clock.Clock(), THINKING_KEY: clock.Clock(), RENDER_KEY: clock.Clock(), FOOD_RECLASSIFICATION_KEY: clock.Clock(),
                     RENDER_THREAD_KEY: clock.Clock()}

    @property
    def last_tick_time(self):
        return self.__last_tick_time

    @property
    def last_tick_delta(self):
        return self.__last_tick_delta

    @property
    def tick_count(self):
        return self.__tick_count

    def add_tick_listener(self, tick_listener):
        self.__tick_listeners.append(tick_listener)

    @property
    def creatures(self):
        return self.__creatures

    @property
    def living_creatures(self):
        return self.__living_creatures

    @property
    def food_tree(self):
        return self.__food_tree

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def queued_creatures(self):
        return self.__queued_creatures

    def queue_creature(self, creature):
        """Queues a creature to be placed in the Environment as soon as Environment.tick() is called"""
        self.__queued_creatures.append(creature)

    def remove_creature(self, creature):
        """Call this method to remove a creature from this Environment. The Creature will no longer receive tick()
        function calls."""
        if creature.exists:
            self.__living_creatures.remove(creature)
        else:
            self.__queued_creatures.remove(creature)
        if creature.environment is not None:
            creature.environment = None

    def tick(self):
        """As an environment has no real sense of time, this method must be called periodically from the outside."""
        tick_clock = self.time[TICK_KEY]
        tick_clock.tick()
        for food in self.food_tree.elements:
            food.tick()
            reclass_clock = self.time[FOOD_RECLASSIFICATION_KEY]
            reclass_clock.tick()
            self.food_tree.reclassify(food, food.shape)
            reclass_clock.tock()
        for creature in self.__queued_creatures:
            if creature.alive:
                self.__living_creatures.append(creature)
                self.__creatures.append(creature)
                creature.environment = self
        self.__queued_creatures.clear()
        for creature in self.__living_creatures[:]:
            creature.sense()
        for creature in self.__living_creatures[:]:
            creature.tick(self.__tick_count)
        for creature in self.__living_creatures[:]:
            creature.execute()
        self.__tick_count += 1
        for listener in self.__tick_listeners:
            listener(self)
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
        if new_x+width/2 > self.__width or new_x-width/2 < 0 or new_y+height/2 > self.__height or new_y-height/2 < 0:
            is_valid = False
        # if is_valid:
        #     for other_creature in self._living_creatures:
        #        if other_creature is not creature and translated_shape.collides(other_creature.body.get_shape()):
        #             is_valid = False
        #             #print("collision detected between "+other_creature._name+ " and "+creature._name)
        #             break
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
        colliding_clock = self.time[FOOD_COLLISION_KEY]
        colliding_clock.tick()
        food_found = []
        # remaining_capacity = max_mass
        for food in self.food_tree.get_collision_candidates(shape):
            if shape.collides(food.shape):
                food_found.append(food)
                if break_if(food_found):
                    break
        colliding_clock.tock()
        return food_found
        # self._food.difference_update(food_eaten)  # = [filter(lambda f: f not in food_eaten, self._food)]

    def sum_mass(self, food_pellets):
        summed = 0
        for food in food_pellets:
            summed += food.mass
        return summed

    def create_food(self, x, y, mass):
        """Creates Food of circular shape at the specified destination with the given mass."""
        self.add_food(Food(mass, shapes.Circle((x, y), 0)))

    def add_food(self, food):
        """Add the specified Food to the Environment for further consumption by Creatures populating it."""
        # self.__food_pellets.add(food)
        self.food_tree.classify(food, food.shape)
        if food.environment is None:
            food.environment = self

    def remove_food(self, food):
        """Removes the specified piece of Food from the Environment."""
        self.food_tree.remove(food)
        food.environment = None


class StageObject:
    def __init__(self):
        self.__environment = None
        self._last_tick_count = -1
        self.age = 0

    @property
    def exists(self):
        return self.__environment is not None

    @property
    def environment(self):
        return self.__environment

    @environment.setter
    def environment(self, environment):
        if environment is not None and self not in environment.living_creatures:
            raise Exception("Can't set environment in Creature as long as it hasn't doesn't exist in that environment."
                            "Please call Environment.queue_creature(creature) and wait for the following"
                            "Environment.tick().")
        self.__environment = environment


class Creature(StageObject):
    """This class represents a Creature which can populate an Environment. It consists of several Organs, of which two
    Organs with a special purpose are the Body (which must always be present) and the Brain (Which is used for
    controlling a creatures behaviour). Each Creature has an age which counts the number of calls to Creature.tick().
    If a Creatures age is greater than genesis.MAX_AGE or it's mass is lower than get_mass_threshold() it dies.

    A Creatures position and orientation are stored in and managed by the body, which is accessible via get_body().

    A Creatures brain is accessible via get_brain()."""
    def __init__(self, body, name=None):
        super().__init__()
        self.__organs = []
        self.__body = body
        body.mass_listeners.append(self.__register_mass)
        self.__brain = None
        self.add_organ(body)
        self.__alive = True
        self.name = name
        self.organ_tick_cost = 0

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

    def __register_mass(self, amount):
        if amount < self.mass_threshold:
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
        if type(organ) is Brain:
            for wired_organ in self.__organs:
                if wired_organ is not self.__brain:
                    self.__brain.unwire_organ(wired_organ)
        self.__organs.remove(organ)
        organ._creature = None

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
        if self.age > MAX_AGE:
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
        raise Exception("Changing a Creatures body is not implemented.")

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

    def mutate(self, mutation_model):
        if random.random() < mutation_model.mutation_likelihood:
            new_organ = None
            if random.random() < 0.5:
                new_organ = Mouth(random_from_interval(0.1, 6), random_from_interval(-10, 10))
            else:
                new_organ = EuclideanEye(random_from_interval(0.1, 6),
                                         random_from_interval(-10, 10),
                                         random_from_interval(0.1, 6))
            self.add_organ(new_organ, mutation_model)
            new_organ.mutate(mutation_model)
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


class Organ:
    """This class represents an organ which can be used by creatures. It can be added to creatures which will then
    wire the organ to their brain to be used in the future.

    Organs can acquire information about their environment
    which are then fed into the Brain or get commands from the brain in the form of an arbitrary number of floating
    values which are then to be interpreted by the Brain.

    Use the methods Organ.register_input_neuron(Neuron) to add Neurons which acquire information and
    Organ.register_output_neuron(Neuron) to add Neurons which get commands from the brain."""
    def __init__(self, label=None):
        self.__input_neurons = []
        self.__output_neurons = []
        self.__creature = None
        self.label = label

    def __str__(self):
        return self.label if self.label is not None else "organ"

    def __repr__(self):
        return self.__str__()

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
        self.__creature = creature
        if self not in creature.organs:
            creature.add_organ(self)

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

    @property
    def shape(self):
        """Returns the shape of this organ if it has any physical representation and None otherwise."""
        return None

    @shape.setter
    def shape(self, shape):
        raise Exception("Organ " + str(type(self)) + " can't have a shape!")

    def clone(self):
        """Clones the organ and returns it. The cloned organ is not attached to any creature."""
        raise Exception("Clone not implemented for organ of type "+str(type(self)))


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

    def connect_to_layer(self, neurons, weight=0, mutation_model=None):
        for neuron in neurons:
            self.connect_to_neuron(neuron, weight, mutation_model)

    def disconnect_from_neuron(self, neuron):
        self._connections.pop(neuron)

    def connect_to_neuron(self, neuron, weight=0, mutation_model=None):
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

        self.__input_layer = []
        self.__hidden_layer = []
        self.__output_layer = []
        self.__layers = [self.__input_layer, self.__hidden_layer, self.__output_layer]

        self.__bias_input_layer = Neuron(identity_activation, "input bias")
        self.__bias_hidden_layer = Neuron(identity_activation, "hidden bias")

        self.add_hidden_neuron(self.__bias_hidden_layer)
        self.add_input_neuron(self.__bias_input_layer)
        # self.__thinking_thread = threading.Thread(target=self.__think_for_thread)
        # self.__lock_0 = threading.Lock()
        # self.__lock_1 = threading.Lock()
        # print("acquire lock 0")
        # self.__lock_0.acquire()
        # self.__thinking_thread.start()
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
        think_clock = self.creature.environment.time[THINKING_KEY]
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
        super().__init__("eye")
        self.body_distance = body_distance
        self.rotation = rotation
        self.radius = radius
        self.__vision_neuron = InputNeuron("eye: vision")
        self.register_input_neuron(self.__vision_neuron)

    def sense(self):
        food = self.creature.environment.find_colliding_food(self.shape, lambda x:False)
        self.__vision_neuron.receive_fire(len(food))

    @property
    def vision_neuron(self):
        return self.__vision_neuron

    @property
    def pos(self):
        [dx, dy] = convert_to_delta_distance(self.body_distance, self.rotation + self.creature.body.rotation)
        return [self.creature.center_x + dx, self.creature.center_y + dy]

    @property
    def shape(self):
        pos = self.pos
        return shapes.Circle(pos, self.radius)

    def clone(self):
        m = EuclideanEye(self.body_distance, self.rotation, self.radius)
        return m

    @property
    def tick_cost(self):
        return self.body_distance / 60 + self.radius ** 2 / 20


class Mouth(Organ):
    def __init__(self, body_distance, rotation, capacity=10., mouth_radius=2):
        super().__init__("mouth")
        self.body_distance = body_distance
        self.rotation = rotation
        self.mouth_radius = mouth_radius
        self.food_capacity = capacity
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

    @property
    def eat_neuron(self):
        return self.__eat_neuron

    @property
    def has_eaten_neuron(self):
        return self.__has_eaten_neuron

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
        return self.body_distance / 60 + self.mouth_radius ** 2 / 20 + self.food_capacity / 50
        # G todo: replace with realistic tick cost -> use mouth area and not just radius!

    @property
    def pos(self):
        [dx, dy] = convert_to_delta_distance(self.body_distance, self.rotation + self.creature.body.rotation)
        return [self.creature.center_x + dx, self.creature.center_y + dy]

    @property
    def shape(self):
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
        self.creature.mass += self.__amount_eaten

    def sense(self):
        self.__has_eaten_neuron.receive_fire(self.__amount_eaten)

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
        # self._food.difference_update(food_eaten)  # = [filter(lambda f: f not in food_eaten, self._food)]
        return eaten

    # def find_colliding_food_for_thread(self):
    #     self.__colliding_food_lock_0.acquire()
    #     while True:
    #         self.__colliding_food = self.find_colliding_food()
    #         self.__colliding_food_lock_1.release()
    #         self.__colliding_food_lock_0.acquire()

    def find_colliding_food(self):
        """Tries to consume Food which intersects with shape up the given maximum mass max_mass. The consumed mass
        is returned by this function and all Food that has been consumed will be removed from the Environment."""

        # r = random.randint(0, 1000)
        # print("start finding food ("+str(r)+")")
        environment = self.creature.environment
        if environment is not None:
            def f(foods): return environment.sum_mass(foods) > self.food_capacity
            return environment.find_colliding_food(self.shape, f)
        return None
        # print("finished finding food (" + str(r) + ")")


class Body(Organ):
    def __init__(self, mass, shape, angle=0, max_mass_burn=20):
        super().__init__("body")

        self._initial_mass = mass
        self.__shape = shape
        self._max_mass_burn = max_mass_burn

        self.rotation = angle
        # G todo: add collision neuron
        self.__mass_neuron = InputNeuron("body: mass")
        self.register_input_neuron(self.__mass_neuron)
        self.__age_neuron = InputNeuron("body: age")
        self.register_input_neuron(self.__age_neuron)

        self.mass_listeners = []
        self.__mass = mass

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, shape):
        self.__shape = shape

    def clone(self):
        b = Body(self.__mass, copy.deepcopy(self.shape), self.rotation, self._max_mass_burn)
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
        amount = max(0, amount)
        self.__mass = amount
        self.shape.radius = math.sqrt(amount / 2)
        for mass_listener in self.mass_listeners:
            mass_listener(self.__mass)

            # G todo: change shape size to reflect mass change

    @property
    def center(self):
        return self.shape.center

    @center.setter
    def center(self, center):
        self.shape.center = center

    def move(self, dx, dy):
        self.shape.translate(dx, dy)

    @property
    def center_x(self):
        return self.shape.center_x

    @center_x.setter
    def center_x(self, x):
        self.shape.center_x = x

    @property
    def center_y(self):
        return self.shape.center_y

    @center_y.setter
    def center_y(self, y):
        self.shape.center_y = y

    def sense(self):
        creature_age = self.creature.age
        self.__age_neuron.receive_fire(creature_age / MAX_AGE)
        self.__mass_neuron.receive_fire(self.__mass / self._initial_mass)

    def execute(self):
        pass

    @property
    def tick_cost(self):
        return self.__mass / 100  # G todo: replace with realistic tick cost -> a bigger body should cost more mass


class Legs(Organ):
    # G should we rename this to fins? the creatures are moving around in water after all. or are they? do we want to
    # G model this aspect that closely?
    # G maybe we want some parameter which guides the max speed of legs and make faster legs more expensive to maintain?
    def __init__(self, max_distance=5, max_degree_turn=10):
        super().__init__("legs")
        self.max_distance = max_distance
        self.max_degree_turn = max_degree_turn
        self.__distance_moved = 0
        self.__distance_moved_neuron = InputNeuron("Legs: distance moved")
        self.__forward_neuron = OutputNeuron("Legs: forward")
        self.__turn_clockwise_neuron = OutputNeuron("Legs: turn")
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
    def __init__(self, mutation_model):
        super().__init__("fission")
        self.mutation_model = mutation_model
        self.__fission_neuron = OutputNeuron("Fission: fission")
        self.register_output_neuron(self.__fission_neuron)

    def clone(self):
        return Fission(copy.deepcopy(self.mutation_model))

    def execute(self):
        fission_value = self.__fission_neuron.consume()
        if fission_value > 0:
            creature = self.creature
            environment = creature.environment
            initial_mass = creature.body.mass
            new_mass = initial_mass * 0.6
            creature.body.mass = new_mass
            if creature.alive:

                split_creature = creature.clone()
                split_creature.age = 0
                split_creature.body.rotation = random.random()*360

                split_creature.__name = creature.name + "+"

                split_creature.mutate(self.mutation_model)
                if split_creature.alive:
                    environment.queue_creature(split_creature)
                # print("creature "+self._creature._name+" split itself")

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

