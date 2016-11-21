import genesis as gen
import random as r
import shapes
import copy
import render_management
import rendering
import operator
import pygame

start_mass = 200
mutation_model = gen.MutationModel(0.2, 0.3)
init_mutation_model = gen.MutationModel(1, 1)
creature_radius = 5

factor = 1.5
width = 180 * factor
height = 120 * factor

init_food_count = int(50 * factor ** 2)
max_food_count = int(100 * factor ** 2)
food_mass_min = 80
food_mass_max = 130

init_creature_count = 5
min_creature_count = int(5 * factor ** 2)


def random_pos(width, height, borders=0):
    return [random_from_interval(borders, width-borders), random_from_interval(borders, height-borders)]


def random_from_interval(min, max):
    return r.random()*(max-min)+min


def create_number_listener(environment):
    if len(environment._living_creatures) < min_creature_count:
        place_random_creature(environment)


def food_listener(environment):
    if environment._tick_count % 10 /(factor**2) == 0 and len(environment._food) < max_food_count:
        place_random_food(environment)


def place_random_food(environment):
    amount = random_from_interval(food_mass_min, food_mass_max)
    food_radius = amount / 20
    r_pos = random_pos(width, height, food_radius)
    environment.create_food(r_pos[0], r_pos[1], amount, food_radius)


def place_random_creature(environment):
    if False:
        environment.queue_creature(create_master_creature())
    else:
        r_pos = random_pos(width, height, creature_radius)
        name = str(r.randint(0, 1000000))
        name = "0" * (4 - len(name)) + name
        print("Creating " + name + " at " + str(r_pos))
        body = gen.Body(start_mass, shapes.Circle(r_pos[0], r_pos[1], creature_radius))
        brain = gen.Brain()
        legs = gen.Legs()
        mouth = gen.Mouth(3, 0)
        fission = gen.Fission(mutation_model)

        creature = gen.Creature(body, name)
        creature.add_organ(brain)
        creature.add_organ(legs)
        creature.add_organ(mouth)
        creature.add_organ(fission)

        creature.mutate(init_mutation_model)

        r_pos = random_pos(width, height, creature_radius)
        creature.set_pos(r_pos[0], r_pos[1])
        environment.queue_creature(creature)


def create_master_creature():
    name = "Genesis"
    body = gen.Body(start_mass, shapes.Circle(0, 0, creature_radius))
    brain = gen.Brain()
    legs = gen.Legs()
    mouth = gen.Mouth(0, 0)
    fission = gen.Fission(mutation_model)

    creature = gen.Creature(body, name)
    creature.add_organ(brain)
    creature.add_organ(legs)
    creature.add_organ(mouth)
    creature.add_organ(fission)

    age_hn = brain.hidden_layer[0]
    body.age_neuron.connect_to_neuron(age_hn, 1)
    age_hn.connect_to_neuron(fission.fission_neuron, 1)
    brain.bias_hidden_layer.connect_to_neuron(fission.fission_neuron, -0.7)
    brain.bias_hidden_layer.connect_to_neuron(legs.forward_neuron, 0.5)
    brain.bias_hidden_layer.connect_to_neuron(legs.turn_clockwise_neuron, 0.5)
    brain.bias_hidden_layer.connect_to_neuron(mouth.eat_neuron, 1)

    r_pos = random_pos(width, height, creature_radius)
    creature.set_pos(r_pos[0], r_pos[1])
    return creature


def start(width, height):
    #print("running simulation with " + str(len(creatures)) + " creatures")
    environment = gen.Environment(width, height)
    # environment.queue_creature(create_master_creature())
    for i in range(init_food_count):
        place_random_food(environment)
    for i in range(min_creature_count):
        place_random_creature(environment)

    environment._tick_listeners.append(food_listener)
    environment._tick_listeners.append(create_number_listener)
    renderer = rendering.PyGame(environment)
    manager = render_management.Manager(environment.tick, renderer.render)
    manager.start()

start(width, height)
