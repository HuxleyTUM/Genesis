import genesis as gen
import random as r
import shapes
import copy
import render_management
import rendering
import event_management

start_mass = 200
mutation_model = gen.MutationModel(0.2, 0.3)
init_mutation_model = gen.MutationModel(1, 1)
creature_radius = 5

factor = 1.5
width = 180 * factor
height = 120 * factor

init_food_count = int(50 * factor ** 2)
max_food_count = int(100 * factor ** 2)
init_food_mass = 5

init_creature_count = 5
min_creature_count = int(5 * factor ** 2)


def create_number_listener(environment):
    if len(environment.living_creatures) < min_creature_count:
        place_random_creature(environment)


def food_listener(environment):
    if environment.tick_count % 50 == 0:
        print("\n--- times ---")
        for clock in environment.clocks.clocks:
            print(str(clock))
            clock.reset()
        print("\n")
    #     print("total time ticking: "+str(environment.clocks_ticking))
    #     print("time thinking: "+str(environment.clocks_thinking))
    #     print("time food consumption: "+str(environment.clocks_consumption_food))
    #     print("time food collision: "+str(environment.clocks_collision_food))
    #     print("time creature sensing: "+str(environment.clocks_creature_sensing))
    #     print("time creature ticking: "+str(environment.clocks_creature_ticking))
    #     print("time creature executing: "+str(environment.clocks_creature_executing))
    #     print("time creature collision: "+str(environment.clocks_collision_creatures))
    #     print("time fission: "+str(environment.clocks_fission_executing))
    #     print("times: "+str(environment.clockss))
    #     print("organ times: "+str(environment.organ_times))
    #     print("organ clone times: "+str(environment.organ_clone_time))
    for i in range(environment.food_tree.size, max_food_count):
        place_random_food(environment)


def place_random_food(environment):
    r_pos = gen.random_pos(width, height)
    environment.create_food(r_pos[0], r_pos[1], init_food_mass)


def place_random_creature(environment):
    if False:
        environment.queue_creature(create_master_creature())
    else:
        r_pos = gen.random_pos(width, height, creature_radius)
        name = str(r.randint(0, 1000000))
        name = "0" * (4 - len(name)) + name
        # print("Creating " + name + " at " + str(r_pos))
        body = gen.Body(start_mass, shapes.Circle(r_pos, creature_radius))
        brain = gen.Brain()
        legs = gen.Legs()
        mouth = gen.Mouth(3, 0)
        fission = gen.Fission(mutation_model)

        creature = gen.Creature(body, name)
        creature.add_organ(brain)
        creature.add_organ(legs)
        creature.add_organ(mouth)
        creature.add_organ(fission)

        for organ in creature.organs:
            organ.mutate(init_mutation_model)

        r_pos = gen.random_pos(width, height, creature_radius)
        creature.pos = [r_pos[0], r_pos[1]]
        environment.queue_creature(creature)


def create_master_creature():
    name = "Genesis"
    body = gen.Body(start_mass, shapes.Circle((0, 0), creature_radius))
    brain = gen.Brain()
    legs = gen.Legs()
    mouth = gen.Mouth(1, 0, 10, 1)
    fission = gen.Fission(mutation_model)
    eye_left = gen.EuclideanEye(10, 40, 6)
    eye_right = gen.EuclideanEye(10, -40, 6)

    creature = gen.Creature(body, name)
    creature.add_organ(brain)
    creature.add_organ(legs)
    creature.add_organ(mouth)
    creature.add_organ(fission)
    creature.add_organ(eye_left)
    creature.add_organ(eye_right)

    fission_hn = brain.hidden_layer[1]
    body.age_neuron.connect_to_neuron(fission_hn, 1)
    body.mass_neuron.connect_to_neuron(fission_hn, 1)
    fission_hn.connect_to_neuron(fission.fission_neuron, 1)

    eye_left_hn = brain.hidden_layer[2]
    eye_left.vision_neuron.connect_to_neuron(eye_left_hn, 1)
    eye_left_hn.connect_to_neuron(legs.turn_clockwise_neuron, 1)

    eye_right_hn = brain.hidden_layer[3]
    eye_right.vision_neuron.connect_to_neuron(eye_right_hn, 1)
    eye_right_hn.connect_to_neuron(legs.turn_clockwise_neuron, -1)

    has_eaten_hn = brain.hidden_layer[4]
    mouth.has_eaten_neuron.connect_to_neuron(has_eaten_hn, 1)
    has_eaten_hn.connect_to_neuron(legs.forward_neuron, -0.5)

    distance_moved_hn = brain.hidden_layer[5]
    legs.distance_moved_neuron.connect_to_neuron(distance_moved_hn, 1)
    brain.bias_hidden_layer.connect_to_neuron(distance_moved_hn, -1)
    distance_moved_hn.connect_to_neuron(legs.turn_clockwise_neuron, 1)

    brain.bias_hidden_layer.connect_to_neuron(fission.fission_neuron, -0.7)
    brain.bias_hidden_layer.connect_to_neuron(legs.forward_neuron, 1)
    brain.bias_hidden_layer.connect_to_neuron(mouth.eat_neuron, 1)

    r_pos = gen.random_pos(width, height, creature_radius)
    creature.pos = [r_pos[0], r_pos[1]]
    return creature


def start(dimensions):
    screen = rendering.Screen((1280, 700))
    side_bar_width = 400
    environment_camera = rendering.Camera((0, 0), dimensions, (0, 0), (screen.dimensions[0]-side_bar_width, screen.dimensions[1]))
    highlight_dimension = (side_bar_width, screen.dimensions[1])
    highlight_camera = rendering.Camera((0, 0), highlight_dimension, (environment_camera.screen_dimensions[0], 0), highlight_dimension)
    environment = gen.Environment(environment_camera, dimensions)
    creature_highlight = gen.CreatureHighlight(highlight_camera)

    def process_click(canvas_pos):
        print("clicked!")
        found_creature = False
        for creature in environment.living_creatures:
            if creature.body.shape.point_lies_within(canvas_pos):
                creature_highlight.highlight(creature)
                found_creature = True
                print("clicked creature!")
                break
        if not found_creature:
            creature_highlight.highlight(None)

    environment.queue_creature(create_master_creature())
    for i in range(init_food_count):
        place_random_food(environment)
    # for i in range(min_creature_count):
    #     place_random_creature(environment)

    environment.add_tick_listener(food_listener)
    environment.add_tick_listener(create_number_listener)

    # environment_canvas = rendering.Canvas(environment_camera)
    screen.add_canvas(environment)
    screen.add_canvas(creature_highlight)
    event_manager = event_management.EventManager(environment_camera)
    event_manager.canvas_clicked_listeners.append(process_click)
    renderer = rendering.PyGameRenderer(screen, render_clock=environment.clocks[gen.RENDER_KEY],
                                        thread_render_clock=environment.clocks[gen.RENDER_THREAD_KEY])
    manager = render_management.Manager(environment.tick, renderer.render, event_manager)
    manager.start()

start((width, height))
