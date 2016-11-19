import math
import time, threading
import copy
import shapes


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


class Food:
    def __init__(self, mass, shape):
        self._mass = mass
        self._shape = shape

    def get_shape(self):
        return self._shape

    def get_mass(self):
        return self._mass


class Environment:
    def __init__(self, width=1000, height=1000):
        self._creatures = []
        self._food = []
        self._width = width
        self._height = height
        self._running = True
        self._queued_creatures = []

    def start(self):
        self.tick()

    def queue_creature(self, creature):
        self._queued_creatures.append(creature)

    def kill_create(self, creature):
        if creature._existing:
            self._creatures.remove(creature)
        else:
            self._queued_creatures.remove(creature)
        creature._alive = False

    def tick(self):
        for creature in self._queued_creatures:
            if creature._alive:
                self._creatures.append(creature)
                creature.set_environment(self)
        self._queued_creatures.clear()
        for creature in self._creatures[:]:
            creature.sense()
        for creature in self._creatures[:]:
            creature.tick()
        for creature in self._creatures[:]:
            creature.execute()
        threading.Timer(0.1, self.tick).start()

    def move_creature(self, creature, distance_to_travel):
        [x, y] = [creature.get_x(), creature.get_y()]
        rotation = creature.get_body().get_rotation()
        [delta_x, delta_y] = convert_to_delta_distance(distance_to_travel, rotation)  # todo: calculate new position using x, y, distance_to_travel & rotation
        [new_x, new_y] = [x + delta_x, y + delta_y]
        translated_shape = copy.deepcopy(creature.get_body().get_shape())
        translated_shape.set_pos(new_x, new_y)
        # todo: the following only computes if the the new position is valid.
        # todo- it doesn't calculate how far the object should actually move instead!
        is_valid = True
        [width, height] = creature.get_body().get_shape().get_dimensions()
        if new_x+width/2 > self._width or new_x-width/2 < 0 or new_y+height/2 > self._height or new_y-height/2 < 0:
            is_valid = False
        if is_valid:
            for other_creature in self._creatures:
                if other_creature is not creature and translated_shape.collides(other_creature.get_body().get_shape()):
                    is_valid = False
                    break
            if is_valid:
                creature.get_body().set_position(new_x, new_y)

    def turn_creature(self, creature, angle_to_turn):
        creature.get_body().set_rotation(creature.get_body().get_rotation()+angle_to_turn)

    #def to_absolute_position(self, creature, dx, dy):
    #    return [dx, dy]  # G todo: return actual absolute position

    def consume_food(self, x, y):
        eaten = 0
        point = shapes.Circle(x, y, 0)
        food_eaten = set()
        for food in self._food:
            if point.collides(food.get_shape()):
                eaten += food.get_mass()
                food_eaten.add(food)
        self._food = filter(lambda f: f not in food_eaten, self._food)
        return eaten

    def create_food(self, x, y, mass):
        self._food.append(Food(mass, shapes.Circle(x, y, 15)))


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

    def tick(self):
        print("Creature.tick()\tCreature.name()="+self._name)
        #print("\tCreature._organs = "+str(self._organs))
        print("\tCreature._body.get_mass() = "+str(self._body.get_mass()))
        print("\tCreature._body.get_energy() = "+str(self._body.get_energy()))
        print("\tCreature._body.get_pos() = ["+str(self.get_x())+", "+str(self.get_y())+"]")
        print("\tCreature._body.get_rotation() = "+str(self._body.get_rotation()))
        if self._brain is not None:
            self._brain.think()
        tick_cost_summed = 0
        for organ in self._organs:
            tick_cost_summed += organ.tick_cost()
        self.decrease_energy(tick_cost_summed)

    def get_environment(self):
        return self._environment

    def get_body(self):
        return self._body

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

    def clone(self):
        memo_dict = {id(self._environment): None}
        to_return = copy.deepcopy(self, memo_dict)
        to_return._existing = False
        return to_return

    def kill(self):
        print("Creature.kill()")
        self._environment.kill_create(self)


class Organ:
    def __init__(self, label=None):
        self._input_neurons = []
        self._output_neurons = []
        self._creature = None
        self._label = label

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


class Neuron:
    def __init__(self, activation_function, label=None):
        self._label = label
        self._connections = {}
        self._fire_listeners = []
        self._activation_function = activation_function
        self._summed_input = 0

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
        self._brain_hidden_layer = []
        self._brain_input_neurons = []
        self._brain_output_neurons = []
        self._input_layer_bias = Neuron(identity_activation, "input bias")
        self._hidden_layer_bias = Neuron(identity_activation, "hidden bias")
        self.add_input_neuron(self._input_layer_bias)
        self.add_hidden_neuron(self._hidden_layer_bias)

    def add_input_neuron(self, neuron):
        self._input_neurons.append(neuron)
        neuron.connect_to_layer(self._brain_hidden_layer)
        self.fill_hidden_layer(len(self._input_neurons))

    def add_output_neuron(self, neuron):
        self._output_neurons.append(neuron)
        Brain._connect_layer_to_neuron(self._brain_hidden_layer, neuron)
        self.fill_hidden_layer(len(self._output_neurons))

    def fill_hidden_layer(self, count):
        for i in range(len(self._brain_hidden_layer), count):
            self.add_hidden_neuron()

    def add_hidden_neuron(self, neuron=Neuron(sigmoid_activation)):
        self._brain_hidden_layer.append(neuron)
        Brain._connect_layer_to_neuron(self._brain_input_neurons, neuron)
        neuron.connect_to_layer(self._brain_output_neurons)

    @staticmethod
    def _connect_layer_to_neuron(layer, neuron):
        for neuron_from in layer:
            neuron_from.connect_to_neuron(neuron)

    def tick_cost(self):
        return len(self._brain_hidden_layer)/10  # todo: replace with realistic tick cost

    def think(self):
        self._input_layer_bias.receive_fire(1)
        for input_neuron in self._brain_input_neurons:
            input_neuron.fire()
        self._hidden_layer_bias.receive_fire(1)
        for hidden_neuron in self._brain_hidden_layer:
            hidden_neuron.fire()
        for output_neuron in self._brain_output_neurons:
            output_neuron.fire()

    def wire_organ(self, organ):
        for input_neuron in organ.get_input_neurons():
            self.add_input_neuron(input_neuron)
        for output_neuron in organ.get_output_neurons():
            self.add_output_neuron(output_neuron)


class Mouth(Organ):
    def __init__(self, body_distance, rotation):
        super().__init__("mouth")
        self._body_distance = body_distance
        self._rotation = rotation
        self._amount_eaten = 0
        self._eat_neuron = OutputNeuron("mouth: eat")
        self._has_eaten_neuron = InputNeuron("mouth: has eaten")
        self.register_output_neuron(self._eat_neuron)
        self.register_input_neuron(self._has_eaten_neuron)

    def tick_cost(self):
        return self._body_distance  # todo: replace with realistic tick cost

    def execute(self):
        environment = self._creature.get_environment()
        [dx, dy] = convert_to_delta_distance(self._body_distance, self._rotation)
        [x, y] = [self._creature.get_x() + dx, self._creature.get_y() + dy]
        self._amount_eaten = environment.consume_food(x, y)
        self._creature.get_body().set_mass(self._creature.get_body().get_mass() + self._amount_eaten)


class Body(Organ):
    def __init__(self, energy, mass, shape, angle=0):
        super().__init__("body")
        self._energy = 0
        self.set_energy(energy)
        self._mass = 0
        self.set_mass(mass)
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
        # G todo: change shape size to reflect mass change

    def get_energy(self):
        return self._energy

    def set_energy(self, new_energy):
        self._energy = new_energy
        if self._energy < 0:
            self._creature.kill()

    def burn_mass(self, mass_to_burn):
        mass_to_burn = min(mass_to_burn, self._mass)
        energy_gained = mass_to_burn
        # G todo: formula for burning mass here. should be less effective if there is already a lot of energy
        self._energy += energy_gained
        self._mass -= mass_to_burn

    def set_rotation(self, angle):
        self._rotation = angle

    def get_rotation(self):
        return self._rotation

    def set_position(self, x, y):
        self._shape._y = y
        self._shape._x = x

    def get_x(self):
        return self._shape.get_x()

    def get_y(self):
        return self._shape.get_y()

    def sense(self):
        self._mass_neuron.receive_fire(self._mass)
        self._energy_neuron.receive_fire(self._energy)

    def execute(self):
        amount_to_burn = self._burn_mass_neuron.consume()
        self.burn_mass(amount_to_burn)

    def tick_cost(self):
        return self._mass/100  # G todo: replace with realistic tick cost -> a bigger body should cost more energy

    def get_shape(self):
        return self._shape


class Legs(Organ):
    # G should we rename this to fins? the creatures are moving around in water after all. or are they? do we want to
    # G model this aspect that closely?
    # G maybe we want some parameter which guides the max speed of legs and make faster legs more expensive to maintain?
    def __init__(self):
        super().__init__("legs")
        self._forward_neuron = OutputNeuron("Legs: forward")
        self._turn_clockwise_neuron = OutputNeuron("Legs: turn")
        self.register_output_neuron(self._forward_neuron)
        self.register_output_neuron(self._turn_clockwise_neuron)

    def execute(self):
        distance_to_travel = self._forward_neuron.consume()
        angle_to_turn = self._turn_clockwise_neuron.consume()
        self._creature.get_environment().turn_creature(self._creature, angle_to_turn)
        self._creature.get_environment().move_creature(self._creature, distance_to_travel)
        energy_to_use = distance_to_travel + angle_to_turn/10  # G todo: replace with realistic formula
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
        if fission_value > 0:
            initial_energy = self._creature.get_body().get_energy()
            new_energy = initial_energy * 0.4
            new_mass = self._creature.get_body().get_mass() * 0.4
            self._creature.get_body().set_energy(new_energy)
            self._creature.get_body().set_mass(new_mass)

            split_creature = self._creature.clone()

            split_creature.get_body().set_energy(new_energy)
            split_creature.get_body().set_mass(new_mass)
            split_creature._name = self._creature._name + "."

            self._mutation_model.mutate(split_creature)
            self._creature.get_environment().queue_creature(split_creature)

    def tick_cost(self):
        return 0
        # todo: replace with realistic tick cost


class MutationModel:
    # G do we want to apply different mutation levels to each organ? or should the parameter mutation_level just say how
    # G mutation in general occurs? after all we might want to mutate each parameter in each organ differently as well,
    # G so just passing different parameters to the organ help that much either..
    def __init__(self, mutation_level):
        self._mutation_level = mutation_level

    def mutate(self, creature):
        for organ in creature.get_organs():
            organ.mutate(self._mutation_level)

