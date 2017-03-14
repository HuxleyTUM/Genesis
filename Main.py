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


class TaskBar(rendering.SimpleCanvas):
    def __init__(self, dimensions, render_manager, camera=rendering.RelativeCamera()):
        super().__init__(dimensions, camera, 1, (255, 255, 255, 0), (0, 0, 0, 0))
        self.render_manager = render_manager
        padding = dimensions[1] * 0.1
        size = dimensions[1] - padding*2
        play_rect_background = shapes.Rectangle(padding, padding, size, size)
        play_rect_icon = shapes.Rectangle(padding + size/3, padding + size/3, size/3, size/3)
        self.play_button_graphic_background = rendering.SimpleMonoColouredGraphic(play_rect_background, (200, 200, 200, 0))
        self.play_button_graphic_icon = rendering.SimpleMonoColouredGraphic(play_rect_icon, (0, 255, 0, 0))
        play_button = rendering.Button([self.play_button_graphic_background, self.play_button_graphic_icon],
                                       play_rect_background)
        self.register_button(play_button)
        play_button.listeners.append(render_manager.resume)

        pause_rect_background = shapes.Rectangle(play_rect_background.right + padding, padding, size, size)
        pause_rect_icon = shapes.Rectangle(pause_rect_background.left + size/3, pause_rect_background.down + size/3, size/3, size/3)
        self.pause_button_graphic_background = rendering.SimpleMonoColouredGraphic(pause_rect_background, (200, 200, 200, 0))
        self.pause_button_graphic_icon = rendering.SimpleMonoColouredGraphic(pause_rect_icon, (100, 100, 100, 0))
        pause_button = rendering.Button([self.pause_button_graphic_background, self.pause_button_graphic_icon],
                                        pause_rect_background)
        self.register_button(pause_button)
        pause_button.listeners.append(render_manager.pause)


def start(environment_dimensions):
    screen = rendering.Screen((1280, 700))
    side_bar_width = 400
    task_bar_height = 50
    environment_canvas_dimensions = (screen.dimensions[0] - side_bar_width, screen.dimensions[1] - task_bar_height)
    # environment_camera = rendering.AbsoluteCamera((0, 0), dimensions, (0, 0), (screen.dimensions[0]-side_bar_width, screen.dimensions[1]))
    environment_camera = rendering.RelativeCamera((0, 0), (4, 4))
    task_bar_camera = rendering.RelativeCamera()
    highlight_dimension = (side_bar_width, screen.dimensions[1])
    highlight_camera = rendering.RelativeCamera()
    environment = gen.Environment(environment_camera, environment_canvas_dimensions, environment_dimensions)
    creature_highlight = gen.CreatureHighlight(highlight_dimension, highlight_camera)



    environment.queue_creature(create_master_creature())
    for i in range(init_food_count):
        place_random_food(environment)
    # for i in range(min_creature_count):
    #     place_random_creature(environment)

    environment.add_tick_listener(food_listener)
    environment.add_tick_listener(create_number_listener)

    # environment_canvas = rendering.Canvas(environment_camera)
    screen.add_canvas(environment, (0, task_bar_height))
    screen.add_canvas(creature_highlight, (environment_canvas_dimensions[0], 0))
    event_manager = event_management.EventManager(screen)

    renderer = rendering.PyGameRenderer(screen, render_clock=environment.clocks[gen.RENDER_KEY],
                                        thread_render_clock=environment.clocks[gen.RENDER_THREAD_KEY])
    manager = render_management.Manager(environment.tick, renderer.render, event_manager)
    task_bar = TaskBar((environment_canvas_dimensions[0], task_bar_height), manager)
    screen.add_canvas(task_bar)

    def process_click(canvas_pos):
        found_creature = False
        for creature in environment.living_creatures:
            if creature.body.shape.point_lies_within(canvas_pos):
                creature_highlight.highlight(creature)
                found_creature = True
                break
        if not found_creature:
            creature_highlight.highlight(None)
    environment.clicked_listeners.append(process_click)
    manager.start()

start((width, height))
