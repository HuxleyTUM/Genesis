import threading
import time
import pygame


class Manager:
    def __init__(self, event_manager, physics=None, render=None):
        self.event_manager = event_manager
        self._running = False
        self._physics = physics
        self._render = render
        self._render_delta = 1/30
        self._physics_delta = 1/30
        self._last_physics_call = -1
        self._last_render_call = -1
        event_manager.quit_listeners.append(self.quit)
        # self.pause_lock = threading.Lock()
        self._paused = False
        self.pyg_events = []

    def start(self):
        self._running = True
        t = threading.Thread(target=self.run)
        t.start()
        while self._running:
            for event in pygame.event.get():
                self.pyg_events.append(event)

    def quit(self):
        self._running = False
        pygame.quit()

    @property
    def pps(self):
        return 1/self._physics_delta

    @pps.setter
    def pps(self, value):
        self._physics_delta = 1/value

    @property
    def paused(self):
        return self._paused

    def pause(self, *args):
        if not self._paused:
            self._paused = True
            # self.pause_lock.acquire()

    def resume(self):
        if self._paused:
            self._paused = False
            # self.pause_lock.release()

    def run(self):
        while self._running:
            # self.pause_lock.acquire()
            # self.pause_lock.release()
            current_time = time.time()
            time_to_next_physics_call = (self._last_physics_call + self._physics_delta) - current_time
            time_to_next_render_call = (self._last_render_call + self._render_delta) - current_time
            time_to_next_call = min(time_to_next_physics_call, time_to_next_render_call)
            if time_to_next_call > 0:
                time.sleep(time_to_next_call)
            if time_to_next_physics_call < time_to_next_render_call:
                self._last_physics_call = time.time()
                if not self.paused and self._physics is not None:
                    self._physics()
            else:
                events = self.pyg_events
                self.pyg_events = []
                for pyg_event in events:  # User did something
                    self.event_manager.process_event(pyg_event)
                self._last_render_call = time.time()
                if self._render is not None:
                    self._render()
