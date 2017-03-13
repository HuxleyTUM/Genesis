import time


class ClockTower:
    def __init__(self, clocks=[]):
        self.__clocks = {}
        self.add_clocks(clocks)

    def __getitem__(self, key):
        return self.__clocks[key]

    @property
    def clocks(self):
        return self.__clocks.values()

    def add_clock(self, clock):
        self.__clocks[clock.name] = clock

    def add_clocks(self, clocks):
        for clock in clocks:
            self.add_clock(clock)


class Clock:
    def __init__(self, name):
        self.name = name
        self.__tick_time = 0
        self.time = 0

    def __str__(self):
        return self.name+": "+str(self.time)

    def tick(self):
        self.__tick_time = time.time()

    def tock(self):
        t = time.time()
        self.time += t - self.__tick_time

    def reset(self):
        self.time = 0
