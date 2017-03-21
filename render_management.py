import threading
import time
import pygame
import sys
import cProfile
import pstats

PHYSICS_PROFILING_FILE_NAME = 'profiling_data_physics'
RENDER_PROFILING_FILE_NAME = 'profiling_data_render'

g_physics = None
g_render = None


class Manager:
    def __init__(self, event_manager, physics=None, render=None):
        self.event_manager = event_manager
        self._running = False
        self._physics = None
        self.physics = physics
        self._render = None
        self.render = render
        self._render_delta = 1/30
        self._physics_delta = 1/30
        self._last_physics_call = -1
        self._last_render_call = -1
        self._physics_counter = 0
        self._render_counter = 0
        event_manager.quit_listeners.append(self.quit)
        # self.pause_lock = threading.Lock()
        self._paused = False
        self.pyg_events = []
        self.profile_physics_calls = False
        self.profile_render_calls = False
        self.physics_stats = None
        self.render_stats = None

    def start(self):
        self._running = True
        t = threading.Thread(target=self.run)
        t.start()
        while self._running:
            for event in pygame.event.get():
                self.pyg_events.append(event)
        pygame.quit()
        sys.exit()

    @property
    def physics(self):
        return self._physics

    @physics.setter
    def physics(self, value):
        self._physics = value
        global g_physics
        g_physics = value
        self._physics_counter = 0
    
    @property
    def render(self):
        return self._render 
    
    @render.setter
    def render(self, value):
        self._render = value
        global g_render
        g_render = value
        self._render_counter = 0

    def quit(self):
        self._running = False

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
                    if self.profile_physics_calls:
                        self.physics_stats = profile('render_management.g_physics()', PHYSICS_PROFILING_FILE_NAME,
                                                     self.physics_stats, self._physics_counter)
                    else:
                        self._physics()
                    self._physics_counter += 1
            else:
                events = self.pyg_events
                self.pyg_events = []
                for pyg_event in events:  # User did something
                    self.event_manager.process_event(pyg_event)
                self._last_render_call = time.time()
                if self._render is not None:
                    if self.profile_render_calls:
                        self.render_stats = profile('render_management.g_render()', RENDER_PROFILING_FILE_NAME,
                                                    self.render_stats, self._render_counter)
                    else:
                        self._render()
                    self._render_counter += 1
                    

def profile(func_name, file_name, stats, print_counter):
    cProfile.run(func_name, file_name)
    if stats is not None:
        stats.add(file_name)
    else:
        stats = pstats.Stats(file_name)
    if print_counter % 50 == 49:
        print("printing results from "+file_name)
        stats.sort_stats('cumulative').print_stats(30)
    return stats
