import pygame

class EventManager:
    def __init__(self, camera):
        self.camera = camera
        self.quit_listeners = []
        self.screen_clicked_listeners = []
        self.canvas_clicked_listeners = []

    def process_event(self, event):
        if event.type == pygame.QUIT:  # If user clicked close
            for listener in self.quit_listeners:
                listener()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            screen_pos = pygame.mouse.get_pos()
            for listener in self.screen_clicked_listeners:
                listener(screen_pos)
            if len(self.canvas_clicked_listeners) > 0:
                canvas_pos = self.camera.transform_point_to_canvas(screen_pos)
                for listener in self.canvas_clicked_listeners:
                    listener(canvas_pos)
