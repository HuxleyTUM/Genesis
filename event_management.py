import pygame


class EventManager:
    def __init__(self, screen):
        self.screen = screen
        self.quit_listeners = []
        self.screen_clicked_listeners = []
        self.mouse_released_canvas = None

    def process_event(self, event):
        if event.type == pygame.QUIT:  # If user clicked close
            for listener in self.quit_listeners:
                listener()
        else:
            screen_pos = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONDOWN:
                (active_canvas, local_point) = self.find_canvas(screen_pos, screen_pos, self.screen)
                print("clicked in "+str(type(active_canvas)))
                for listener in self.screen_clicked_listeners:
                    listener(screen_pos)
                active_canvas.mouse_pressed(local_point)
                self.mouse_released_canvas = active_canvas
            elif event.type == pygame.MOUSEBUTTONUP:
                canvas_local_point = self.mouse_released_canvas.transform_point_from_screen(screen_pos)
                self.mouse_released_canvas.mouse_released(canvas_local_point)

    def find_canvas(self, screen_point, local_point, parent_canvas):
        for canvas in reversed(parent_canvas.canvases):
            canvas_local_point = canvas.transform_point_from_screen(screen_point)
            if canvas.canvas_area.point_lies_within(canvas_local_point):
                return self.find_canvas(screen_point, canvas_local_point, canvas)
        return parent_canvas, local_point
