import threading
import time


class Manager:
    def __init__(self, physics, render):
        self._running = False
        self._physics = physics
        self._render = render
        self._render_delta = 1/1
        self._physics_delta = 1/10
        self._last_physics_call = -1
        self._last_render_call = -1

    def start(self):
        self._running = True
        t = threading.Thread(target=self.run)
        t.start()

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            current_time = time.time()
            time_to_next_physics_call = (self._last_physics_call + self._physics_delta) - current_time
            time_to_next_render_call = (self._last_render_call + self._render_delta) - current_time
            if time_to_next_physics_call < time_to_next_render_call:
                if time_to_next_physics_call > 0:
                    time.sleep(time_to_next_physics_call)
                self._physics()
                self._last_physics_call = time.time()
            else:
                if time_to_next_render_call > 0:
                    time.sleep(time_to_next_render_call)
                self._render()
                self._last_render_call = time.time()



