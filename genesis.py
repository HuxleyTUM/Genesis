import math

# G todo: add some reusable way of mutating parameters. maybe using the Beta function?

def binary_activate(input_value):
    return 0 if input_value < 0 else 1


class Environment:
    def __init__(self, width, height):
        self._creatures = []
        self._width = width
        self._height = height

    def place_creature(self, creature):
        self._creatures.append(creature)
        creature.set_environment(self)

    def tick(self):
        for creature in self._creatures:
            creature.sense()
        for creature in self._creatures:
            creature.tick()
        for creature in self._creatures:
            creature.execute()

    def move_creature(self, creature, distance_to_travel):
        x = creature.get_body().get_x()
        y = creature.get_body().get_y()
        rotation = creature.get_body.get_rotation()
        new_x = x  # todo: calculate new position using x, y, distance_to_travel & rotation
        new_y = y
        # todo: the following only computes if the the new position is valid.
        # todo- it doesn't calculate how far the object can actually move instead!
        is_valid = True
        for other_creature in self._creatures:
            other_x = other_creature.get_x()
            other_y = other_creature.get_y()
            distance = math.sqrt((new_x-other_x)**2 + (new_y-other_y)**2)
            if distance < creature.get_radius() + other_creature.get_radius():
                is_valid = False
                break
        if is_valid:
            creature.get_body().set_position(new_x, new_y)

    def turn_creature(self, creature, angle_to_turn):
        creature.set_rotation(creature.get_rotation()+angle_to_turn)


class Creature:
    def __init__(self, body):
        self._organs = []
        self._body = body
        self._brain = None
        self._environment = None

    def add_organ(self, organ):
        if organ is Brain:
            self._brain = organ
            for organ in self._organs:
                self._brain.wire_organ(organ)
        self._organs.append(organ)
        if self._brain is not None:
            self._brain.wire_organ(organ)
        # todo: set creature in organ

    def sense(self):
        for organ in self._organs:
            organ.sense()

    def execute(self):
        for organ in self._organs:
            organ.execute()

    def tick(self):
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

    def set_environment(self, environment):
        self._environment = environment

    def decrease_energy(self, amount):
        self._body.set_energy(self._body.set_energy-amount)

    def set_energy(self, amount):
        self._body.set_energy(amount)

    def get_organs(self):
        return self._organs

    def clone(self):
        pass  # todo: returned cloned creature


class Organ:
    def __init__(self):
        self._input_neurons = []
        self._output_neurons = []
        self._creature = None

    def register_input_neuron(self, neuron):
        self._input_neurons.append(neuron)

    def register_output_neuron(self, neuron):
        self._output_neurons.append(neuron)

    def get_input_neurons(self):
        return self._input_neurons

    def get_output_neurons(self):
        return self._output_neurons

    def sense(self):
        pass

    def execute(self):
        pass

    def mutate(self, mutation_level):
        pass

    def tick_cost(self):
        pass


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
        for connection in self._connections:
            target = connection[0]
            weight = connection[1]
            target.receive_fire(weight * amount)

    def consume(self):
        amount = self._activation_function(self._summed_input)
        self._summed_input = 0
        return amount

    def get_weight(self, target_neuron):
        return self._connections[target_neuron]


class Brain(Organ):
    def __init__(self):
        super().__init__()
        self._brain_hidden_layer = []
        self._brain_input_neurons = []
        self._brain_output_neurons = []

    def add_input_neuron(self, neuron):
        self._input_neurons.append(neuron)
        neuron.connect_to_layer(self._brain_hidden_layer)

    def add_output_neuron(self, neuron):
        self._output_neurons.append(neuron)
        Brain._connect_layer_to_neuron(self._brain_hidden_layer, neuron)

    def add_hidden_neuron(self, neuron):
        self._brain_hidden_layer.append(neuron)
        Brain._connect_layer_to_neuron(self._brain_input_neurons, neuron)
        neuron.connect_to_layer(self._brain_output_neurons)

    @staticmethod
    def _connect_layer_to_neuron(layer, neuron):
        for neuron_from in layer:
            neuron_from.connect_to_neuron(neuron)

    def tick_cost(self):
        return len(self._brain_hidden_layer)  # todo: replace with realistic tick cost

    def think(self):
        for input_neuron in self._brain_input_neurons:
            input_neuron.fire()
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
    def __init__(self, dx, dy):
        super().__init__()
        self._dx = dx
        self._dy = dy
        self._amount_eaten = 0
        self._eat_neuron = Neuron(binary_activate, "mouth: eat")
        self._has_eaten_neuron = Neuron(binary_activate, "mouth: has eaten")
        self.register_output_neuron(self._eat_neuron)
        self.register_input_neuron(self._has_eaten_neuron)

    def tick_cost(self):
        return math.sqrt(self._dx ** 2 + self._dy ** 2)  # todo: replace with realistic tick cost

    def execute(self):
        environment = self._creature.get_environment()
        [x, y] = environment.to_absolute_position(self._creature, self._dx, self._dy)
        self._amount_eaten = environment.consume_food(x, y)


class Body(Organ):
    # todo: add collision neuron
    def __init__(self, energy, mass, x=0, y=0, angle=0):
        super().__init__()
        self._energy = energy
        self._mass = mass
        self._rotation = angle
        self._y = y
        self._x = x
        self._mass_neuron = Neuron(binary_activate, "body: mass")
        self._energy_neuron = Neuron(binary_activate, "body: energy")
        self._burn_mass_neuron = Neuron(binary_activate, "body: burn mass")
        self.register_input_neuron(self._mass_neuron)
        self.register_input_neuron(self._energy_neuron)
        self.register_output_neuron(self._burn_mass_neuron)

    def get_mass(self):
        return self._mass

    def get_energy(self):
        return self._energy

    def set_energy(self, new_energy):
        self._energy = new_energy

    def burn_mass(self, mass_to_burn):
        energy_gained = mass_to_burn  # todo: formula for burning mass here. should be less effective if there is
        #                               todo- already a lot of energy
        self._energy += energy_gained
        self._mass -= mass_to_burn

    def set_rotation(self, angle):
        self._rotation = angle

    def get_rotation(self):
        return self._rotation

    def set_position(self, x, y):
        self._y = y
        self._x = x

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def tick_cost(self):
        pass  # todo: replace with realistic tick cost


class CircularBody(Body):
    def __init__(self, energy, mass, x=0, y=0, angle=0):
        super().__init__(energy, mass, x, y, angle)

    def sense(self):
        self._mass_neuron.receive_fire(self._mass)
        self._energy_neuron.receive_fire(self._energy)

    def tick_cost(self):
        return self._mass  # todo: replace with realistic upkeep cost

    def get_radius(self):
        return self.get_mass()  # todo: implement radius from spherical mass + density function

    def tick_cost(self):
        pass  # todo: replace with realistic tick cost -> a bigger body should cost more energy


class Legs(Organ):
    # G should we rename this to fins? the creatures are moving around in water after all. or are they? do we want to
    # G model this aspect that closely?
    # G maybe we want some parameter which guides the max speed of legs and make faster legs more expensive to maintain?
    def __init__(self):
        super().__init__()
        self._forward_neuron = Neuron(binary_activate, "Legs: forward")
        self._turn_clockwise_neuron = Neuron(binary_activate, "Legs: turn")
        self.register_output_neuron(self._forward_neuron)
        self.register_output_neuron(self._turn_clockwise_neuron)

    def execute(self):
        distance_to_travel = self._forward_neuron.consume()
        angle_to_turn = self._turn_clockwise_neuron.consume()
        self._creature.get_environment().move_creature(self._creature, distance_to_travel)
        self._creature.get_environment().turn_creature(self._creature, angle_to_turn)
        energy_to_use = distance_to_travel + angle_to_turn  # todo: replace with realistic formula
        self._creature.decrease_energy(energy_to_use)

    def get_forward_neuron(self):
        return self._forward_neuron

    def get_turn_clockwise_neuron(self):
        return self._turn_clockwise_neuron

    def tick_cost(self):
        pass  # G todo: replace with realistic tick cost.


class Fission(Organ):
    # G maybe there could be more than one offspring? this could be a parameter which results in this organ being more
    # G expensive. if more than one offspring possible, this class needs to be renamed.
    # G also another parameter could be how much energy remains with the original creature and how much is
    # G transferred to the "offsprings"..
    def __init__(self, mutation_model):
        super().__init__()
        self._mutation_model = mutation_model
        self._fission_neuron = Neuron(binary_activate, "Binary Fission: fission")
        self.register_output_neuron(self._fission_neuron)

    def execute(self):
        fission_value = self._fission_neuron.consume()
        if fission_value > 0:
            initial_energy = self._creature.get_body().get_energy()
            new_energy = initial_energy * 0.4
            new_mass = self._creature.get_mass() * 0.4
            self._creature.set_energy(new_energy)
            self._creature.set_mass(new_mass)

            split_creature = self._creature.clone()
            split_creature.set_energy(new_energy)
            split_creature.set_mass(new_mass)

            self._mutation_model.mutate(split_creature)
            self._creature.get_environment().place_creature(split_creature)

    def tick_cost(self):
        pass
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

