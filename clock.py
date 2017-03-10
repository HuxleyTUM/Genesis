import time

class Clock:
    def __init__(self):
        self.__tick_time = 0
        self.time = 0

    def tick(self):
        self.__tick_time = time.time()

    def tock(self):
        self.time = time.time() - self.__tick_time

    def reset(self):
        self.time = 0
