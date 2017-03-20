import pygame
import objgraph
import gc


def fire_listeners(listeners, event):
    for listener in listeners:
        listener(event)


def event_condition(action, condition, event):
    if condition:
        event.consume()
        action()


class Event:
    def __init__(self):
        self.consumed = False
        self.screen_mouse_position = None

    def consume(self):
        self.consumed = True


class EventManager:
    def __init__(self, screen):
        self.screen = screen
        self.quit_listeners = []
        self.screen_clicked_listeners = []
        self.mouse_released_canvas = None
        # pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP, pygame.KEYRIGHT, pygame.KEYUP])

    def process_event(self, pyg_event):
        if pyg_event.type == pygame.QUIT or (pyg_event.type == pygame.KEYDOWN and pyg_event.key == pygame.K_ESCAPE):  # If user clicked close
            for listener in self.quit_listeners:
                listener()
        else:
            screen_pos = pygame.mouse.get_pos()
            event = Event()
            event.screen_mouse_position = screen_pos
            canvas_point_list = self.find_canvases(screen_pos, self.screen)
            for active_canvas in canvas_point_list:
                if pyg_event.type == pygame.MOUSEBUTTONDOWN:
                    if pyg_event.button == 1:
                        active_canvas.mouse_pressed(event)
                        if event.consumed:
                            self.mouse_released_canvas = active_canvas
                    elif pyg_event.button == 4:
                        active_canvas.mouse_wheel_scrolled_up(event)
                    elif pyg_event.button == 5:
                        active_canvas.mouse_wheel_scrolled_down(event)
                elif pyg_event.type == pygame.MOUSEBUTTONUP:
                    if pyg_event.button == 1:
                        active_canvas.mouse_released(event)
                        if event.consumed and active_canvas is self.mouse_released_canvas:
                            self.mouse_released_canvas = None
                elif pyg_event.type == pygame.KEYDOWN:
                    if pyg_event.key == pygame.K_LEFT:
                        active_canvas.left_key_pressed(event)
                    elif pyg_event.key == pygame.K_RIGHT:
                        active_canvas.right_key_pressed(event)
                    elif pyg_event.key == pygame.K_UP:
                        active_canvas.up_key_pressed(event)
                    elif pyg_event.key == pygame.K_DOWN:
                        active_canvas.down_key_pressed(event)
                    elif pyg_event.key == pygame.K_0:
                        gc.collect()
                        print("\nprinting most common types..")
                        objgraph.show_most_common_types(limit=10)
                    #     x = []
                    #     y = [x, [x], dict(x=x)]
                    #     gc.collect()
                    #     objgraph.show_refs([y])
                if event.consumed:
                    break
            if pyg_event.type == pygame.MOUSEBUTTONUP and \
                    self.mouse_released_canvas is not None and \
                    pyg_event.button == 1:
                self.mouse_released_canvas.mouse_canceled(event)

    def find_canvases(self, screen_point, parent_canvas, canvases=None):
        if canvases is None:
            canvases = []
        for canvas in reversed(parent_canvas.children):
            global_bounding_rectangle = canvas.global_bounding_rectangle
            if global_bounding_rectangle.point_lies_within(screen_point):
                self.find_canvases(screen_point, canvas, canvases)
        canvases.append(parent_canvas)
        return canvases
