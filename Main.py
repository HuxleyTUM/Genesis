import genesis as gen
import random as r
import shapes
import copy
import render_management
import rendering
import events
import functools
import math

start_mass = 200
mutation_model = gen.MutationModel(0.2, 0.3)
init_mutation_model = gen.MutationModel(1, 1)
creature_radius = 5

factor = 1.5
width = 180 * factor
height = 120 * factor

init_food_count = int(30 * factor ** 2)
max_food_count = int(80 * factor ** 2)
init_food_mass = 5

init_creature_count = 5
min_creature_count = int(5 * factor ** 2)

active_environment = None


def create_number_listener(environment):
    if len(environment.living_creatures) < min_creature_count:
        place_random_creature(environment)


def food_listener(environment):
    # if environment.tick_count % 50 == 0:
        # print("\n--- times ---")
        # for clock in environment.clocks.clocks:
        #     print(str(clock))
        #     clock.reset()
        # print("\n")
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
        body = gen.Body(start_mass, shapes.Circle(r_pos, creature_radius))
        brain = gen.Brain()
        legs = gen.Legs()
        mouth = gen.Mouth(3, 0)
        fission = gen.Fission(mutation_model, init_mutation_model)

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
    body = gen.Body(start_mass, shapes.Circle((100, 100), creature_radius))
    body.rotation = 90
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
    has_eaten_hn.connect_to_neuron(legs.forward_neuron, -0.9)
    has_eaten_hn.connect_to_neuron(mouth.eat_neuron, 2)

    distance_moved_hn = brain.hidden_layer[5]
    legs.distance_moved_neuron.connect_to_neuron(distance_moved_hn, 1)
    brain.bias_input_layer.connect_to_neuron(distance_moved_hn, -1)
    distance_moved_hn.connect_to_neuron(legs.turn_clockwise_neuron, 1)

    brain.bias_hidden_layer.connect_to_neuron(fission.fission_neuron, -0.7)
    brain.bias_hidden_layer.connect_to_neuron(legs.forward_neuron, 1)
    brain.bias_hidden_layer.connect_to_neuron(mouth.eat_neuron, 0.1)

    r_pos = gen.random_pos(width, height, creature_radius)
    creature.pos = [r_pos[0], r_pos[1]]
    return creature


def create_play_button():
    button_area = shapes.Rectangle(0, 0, 30, 30)
    button_icon = shapes.Polygon([(6, 0), (-6, -10), (-6, 10)])
    # self.button_graphic_background = rendering.SimpleMonoColouredGraphic(button_rect_background, )
    button_graphic_icon = rendering.SimpleMonoColouredGraphic(button_icon, (0, 255, 0, 0))
    button = rendering.Button(button_area)
    button.register_and_center_graphic(button_graphic_icon)
    return button


def create_time_warp_button(increase):
    button = rendering.RectangularButton((30, 30))
    c = 1 if increase else -1
    for dx in [-5, 5]:
        button_icon = shapes.Polygon([(3*c, 0), (-3*c, -10), (-3*c, 10)])
        # self.button_graphic_background = rendering.SimpleMonoColouredGraphic(button_rect_background, )
        button_graphic = rendering.SimpleMonoColouredGraphic(button_icon, (0, 255, 0, 0))
        button.register_and_center_graphic(button_graphic)
        button_graphic.translate((dx, 0))
    return button


def create_pause_button():
    button_area = shapes.Rectangle(0, 0, 30, 30)
    button = rendering.Button(button_area)
    for x in (-4, 4):
        button_icon_left = shapes.Rectangle(0, 0, 6, 20)
        button_graphic_icon = rendering.SimpleMonoColouredGraphic(button_icon_left, (100, 100, 100, 0))
        button.register_and_center_graphic(button_graphic_icon)
        button_graphic_icon.translate((x, 0))
    return button


def create_visualise_bounding_button():
    button_area = shapes.Rectangle(0, 0, 30, 30)
    button = rendering.Button(button_area)
    icon_shapes = [shapes.Circle((8, 10), 7), shapes.LineSegment((5, 28), (28, 18))]
    for shape in icon_shapes:
        if shape.has_area():
            shape_graphic = rendering.SimpleMonoColouredGraphic(shape, (0, 0, 255))
        else:
            shape_graphic = rendering.SimpleOutlineGraphic(shape, (0, 0, 255))
        button.register_graphic(shape_graphic)
        button.register_graphic(rendering.SimpleOutlineGraphic(shape.to_bounding_rectangle(), (255, 0, 0)))

    return button


def create_refresh_button():
    button_area = shapes.Rectangle(0, 0, 30, 30)
    button = rendering.Button(button_area)
    points = []
    dy = None
    y = 0
    for angle in range(0, 270, 15):
        point_outer = (-math.sin(math.radians(angle)) * 9, -math.cos(math.radians(angle)) * 9)
        point_inner = (-math.sin(math.radians(angle)) * 7, -math.cos(math.radians(angle)) * 7)
        points.append(point_outer)
        points.insert(0, point_inner)
        if dy is None:
            dy = (point_outer[1] - point_inner[1])
            y = (point_outer[1] + point_inner[1])/2
    polygon_points = [(5, y), (0, y+5), (0, y-5)]
    arrow = shapes.Polygon(polygon_points)
    open_circle = shapes.Polygon(points)
    shape_graphic = rendering.SimpleMonoColouredGraphic(open_circle, (0, 255, 0, 0))
    arrow_graphic = rendering.SimpleMonoColouredGraphic(arrow, (0, 255, 0, 0))
    old_pos = (open_circle.left, open_circle.down)
    button.register_and_center_graphic(shape_graphic)
    new_pos = (open_circle.left, open_circle.down)
    arrow.translate([x-y for x, y in zip(new_pos, old_pos)])
    button.register_graphic(arrow_graphic)

    return button


def alter_speed_function(render_manager, factor):
    def f(event): render_manager.pps *= factor
    return f


def create_task_bar(dimensions, render_manager, renderer, create_environment):
    task_bar = rendering.ButtonBar(dimensions)

    refresh_button = create_refresh_button()
    task_bar.add_button(refresh_button)
    refresh_button.action_listeners.append(lambda event: create_environment())

    play_button = create_play_button()
    task_bar.add_button(play_button)
    play_button.action_listeners.append(lambda event: render_manager.resume())

    pause_button = create_pause_button()    
    task_bar.add_button(pause_button)
    pause_button.action_listeners.append(lambda event: render_manager.pause())

    visualise_bounding_button = create_visualise_bounding_button()
    task_bar.add_button(visualise_bounding_button)
    visualise_bounding_button.action_listeners.append(renderer.visualise_boundings)

    for forward in [True, False]:
        forward_button = create_time_warp_button(forward)
        task_bar.add_button(forward_button)
        time_factor = 2 if forward else 0.5
        forward_button.action_listeners.append(alter_speed_function(render_manager, time_factor))
    return task_bar


# class TaskBar(rendering.SimpleCanvas):
#     def __init__(self, dimensions, render_manager, camera=rendering.RelativeCamera()):
#         super().__init__(canvas_area=shapes.Rectangle(0, 0, dimensions[0], dimensions[1]), camera=camera,
#                          border_thickness=1, border_colour=(255, 255, 255, 0), back_ground_colour=(0, 0, 0, 0))
#         self.render_manager = render_manager
#         padding = dimensions[1] * 0.1
#         size = dimensions[1] - padding*2
#
#
        #play_rect_background.right + padding

def create_environment(screen, environment_camera, environment_dimensions, creature_highlight,
                       task_bar_height, renderer, manager):
    global active_environment
    if active_environment is not None:
        screen.queue_canvas_for_removal(active_environment)
    environment = gen.Environment(environment_camera, environment_dimensions)
    environment.creature_highlight = creature_highlight

    # environment.queue_creature(create_master_creature())
    for i in range(init_food_count):
        place_random_food(environment)

    environment.tick_listeners.append(functools.partial(food_listener, environment))
    environment.tick_listeners.append(functools.partial(create_number_listener, environment))
    screen.add_canvas(environment, (0, task_bar_height), 0)
    active_environment = environment
    renderer.render_clock = environment.clocks[gen.RENDER_KEY]
    renderer.thread_render_clock = environment.clocks[gen.RENDER_THREAD_KEY]
    manager.physics = environment.tick


def start(environment_dimensions):
    screen = rendering.Screen((1280, 700))
    side_bar_width = 400
    task_bar_height = 40
    environment_canvas_dimensions = (screen.dimensions[0] - side_bar_width, screen.dimensions[1] - task_bar_height)
    width_ratio = environment_canvas_dimensions[0] / environment_dimensions[0]
    environment_camera = rendering.RelativeCamera((0, 0), (width_ratio, width_ratio))
    highlight_dimension = (side_bar_width, screen.dimensions[1])
    creature_highlight = gen.CreatureHighlight(highlight_dimension)
    screen.add_canvas(creature_highlight, (environment_canvas_dimensions[0], 0))
    event_manager = events.EventManager(screen)

    renderer = rendering.PyGameRenderer(screen)
    manager = render_management.Manager(event_manager, render=renderer.render)

    create_environment_f = functools.partial(create_environment, screen, environment_camera, environment_dimensions,
                                             creature_highlight, task_bar_height, renderer, manager)
    create_environment_f()
    task_bar = create_task_bar((environment_canvas_dimensions[0], task_bar_height), manager, renderer, create_environment_f)
    screen.add_canvas(task_bar)

    manager.start()

start((width, height))
