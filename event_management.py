import pygame


class EventManager:
    def __init__(self, screen):
        self.screen = screen
        self.quit_listeners = []
        self.screen_clicked_listeners = []

    def process_event(self, event):
        if event.type == pygame.QUIT:  # If user clicked close
            for listener in self.quit_listeners:
                listener()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            screen_pos = pygame.mouse.get_pos()
            (active_canvas, local_point) = self.find_canvas(screen_pos, screen_pos, self.screen)
            print("clicked in "+str(type(active_canvas)))
            for listener in self.screen_clicked_listeners:
                listener(screen_pos)
            active_canvas.clicked(local_point)
            # if len(self.canvas_clicked_listeners) > 0:
            #     canvas_pos = self.camera.transform_point_from_parent(screen_pos)
            #     for listener in self.canvas_clicked_listeners:
            #         listener(canvas_pos)
    
    def find_canvas(self, screen_point, local_point, parent_canvas):
        for canvas in reversed(parent_canvas.canvases):
            canvas_local_point = canvas.transform_point_from_screen(screen_point)
            if canvas.local_bounding.point_lies_within(canvas_local_point):
                return self.find_canvas(screen_point, canvas_local_point, canvas)
        return parent_canvas, local_point
