import math

import numpy


class Shape:
    @property
    def x(self):
        raise Exception("Not implemented!")

    @x.setter
    def x(self, x):
        raise Exception("Not implemented!")

    @property
    def y(self):
        raise Exception("Not implemented!")

    @y.setter
    def y(self, y):
        raise Exception("Not implemented!")

    def translate(self, dx, dy):
        self.pos = [self.x() + dx, self.y() + dy]

    @property
    def pos(self):
        raise Exception("Not implemented!")

    @pos.setter
    def pos(self, pos):
        raise Exception("Not implemented!")
    
    def collides(self, other):
        raise Exception("Not implemented!")
    
    def collides_with_circle(self, circle):
        raise Exception("Not implemented!")

    @property
    def dimensions(self):
        raise Exception("Not implemented!")


class Circle(Shape):
    def __init__(self, x, y, radius):
        self.__x = x
        self.__y = y
        self.__radius = radius
        
    @property
    def pos(self):
        return [self.__x, self.__y]

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @pos.setter
    def pos(self, pos):
        self.__x = pos[0]
        self.__y = pos[1]
        
    def collides(self, other):
        return other.collides_with_circle(self)
    
    def collides_with_circle(self, circle):
        other_pos = circle.pos
        distance = numpy.linalg.norm([self.__x - other_pos[0], self.__y - other_pos[1]])
        # distance = math.sqrt((self.__x - other_pos[0]) ** 2 + (self.__y - other_pos[1]) ** 2)
        return distance < self.__radius + circle.radius
            
    @property
    def dimensions(self):
        return [self.__radius, self.__radius]
        
    @property
    def radius(self):
        return self.__radius
    
    @radius.setter
    def radius(self, radius):
        self.__radius = radius

