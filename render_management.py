import threading
import time
import pygame


class Manager:
    def __init__(self, physics, render, event_manager):
        self.event_manager = event_manager
        self._running = False
        self._physics = physics
        self._render = render
        self._render_delta = 1/30
        self._physics_delta = 1/60
        self._last_physics_call = -1
        self._last_render_call = -1
        event_manager.quit_listeners.append(self.quit)

    def start(self):
        self._running = True
        t = threading.Thread(target=self.run)
        t.start()
        while True:
            for event in pygame.event.get():  # User did something
                self.event_manager.process_event(event)

    def quit(self):
        self._running = False
        pygame.quit()

    # def stop(self):
    #     self._running = False

    def run(self):
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    pygame.quit()
                    # sys.exit()
            current_time = time.time()
            time_to_next_physics_call = (self._last_physics_call + self._physics_delta) - current_time
            time_to_next_render_call = (self._last_render_call + self._render_delta) - current_time
            if time_to_next_physics_call < time_to_next_render_call:
                if time_to_next_physics_call > 0:
                    time.sleep(time_to_next_physics_call)
                self._last_physics_call = time.time()
                self._physics()
            else:
                if time_to_next_render_call > 0:
                    time.sleep(time_to_next_render_call)
                self._last_render_call = time.time()
                self._render()
