import genesis as gen
import random as r
import shapes
import copy

start_mass = 200
start_energy = 300
mutation_model = gen.MutationModel(0.1, 1)
init_mutation_model = gen.MutationModel(1, 1)
creature_radius = 5
width = 300
height = 50
food_count = 100
food_mass_min = 20
food_mass_max = 50
mutation_count = 10


def random_pos(width, height, borders=0):
    return [random_from_interval(borders, width-borders), random_from_interval(borders, height-borders)]


def random_from_interval(min, max):
    return r.random()*(max-min)+min


def revive(creature):
    creature._alive = True
    creature.get_body().set_energy(start_energy)
    creature.get_body().set_mass(start_mass)


def listener(environment):
    if len(environment._living_creatures) == 0:
        print("finished! "+str(len(environment._food))+" pieces of food left")
        print(str(environment._time_collision_food)+", "+
              str(environment._time_collision_creatures)+", "+
              str(environment._time_thinking))
        environment._running = False
        new_creatures = []
        best_creature = None
        best_fitness = -1
        for old_creature in environment._creatures:
            if old_creature._last_tick_count > best_fitness:
                best_fitness = old_creature._last_tick_count
                best_creature = old_creature
        print("best creature from last run survived was "+best_creature._name+" with " +
              str(best_fitness)+" ticks!")
        print("best creature energy="+str(best_creature.get_energy())+", mass="+str(best_creature.get_mass())+
              ", _total_mass_gained="+str(best_creature.get_body()._total_mass_gained))
        fitness.append(best_fitness)
        print("hidden layer biases: "+str(best_creature._brain._bias_hidden_layer._connections))
        print("new fitness vector: "+str(fitness))

        for i in range(mutation_count):
            new_creature = copy.deepcopy(best_creature)
            revive(new_creature)
            new_creature.mutate(mutation_model)
            new_creature._existing = False
            new_creature._name = "Offspring "+str(i)
            new_creatures.append(new_creature)
        mutation_model._mutation_likelihood = max(0.05, mutation_model._mutation_likelihood * 0.9)
        mutation_model._mutation_strength = max(0.1, mutation_model._mutation_strength * 0.9)
        run_simulation(new_creatures, environment._width, environment._height)


def run_simulation(creatures, width, height):
    print("running simulation with "+str(len(creatures))+" creatures")
    environment = gen.Environment(width, height)
    for i in range(food_count):
        amount = random_from_interval(food_mass_min, food_mass_max)
        food_radius = amount/20
        r_pos = random_pos(width, height, food_radius)
        environment.create_food(r_pos[0], r_pos[1], amount, food_radius)
    for creature in creatures:
        r_pos = random_pos(width, height, creature_radius)
        creature.set_pos(r_pos[0], r_pos[1])
        environment.queue_creature(creature)

    environment._tick_listeners.append(listener)
    environment.start()


def start(width, height):
    creatures = []
    for name in ["Genesis", "Evo", "Ada", "Miri", "Benedicte"]:
        r_pos = random_pos(width, height, creature_radius)
        print("Creating "+name+" at "+str(r_pos))
        body = gen.Body(start_energy, start_mass, shapes.Circle(r_pos[0], r_pos[1], creature_radius))
        brain = gen.Brain()
        legs = gen.Legs()
        mouth = gen.Mouth(1, 0)
        fission = gen.Fission(mutation_model)

        creature = gen.Creature(body, name)
        creature.add_organ(brain)
        creature.add_organ(legs)
        creature.add_organ(mouth)
        creature.add_organ(fission)

        creature.mutate(init_mutation_model)
        creatures.append(creature)

        #brain._bias_hidden_layer.connect_to_neuron(legs.get_forward_neuron(), 0)
        #brain._bias_hidden_layer.connect_to_neuron(body._burn_mass_neuron, 1)
        #brain._bias_hidden_layer.connect_to_neuron(legs.get_turn_clockwise_neuron(), 5)
        #brain._bias_hidden_layer.connect_to_neuron(mouth._eat_neuron, 1)

    run_simulation(creatures, width, height)

fitness = []
start(width, height)
