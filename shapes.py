import math


class Shape:
    def get_pos(self):
        pass

    def set_pos(self, x, y):
        pass
    
    def collides(self, other):
        pass
    
    def collides_with_circle(self, circle):
        pass
    
    def get_dimensions(self):
        pass


class Circle(Shape):
    def __init__(self, x, y, radius):
        self._x = x
        self._y = y
        self._radius = radius
        
    def get_pos(self):
        return [self._x, self._y]

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def set_pos(self, x, y):
        self._x = x
        self._y = y
        
    def collides(self, other):
        return other.collides_with_circle(self)
    
    def collides_with_circle(self, circle):
        other_pos = circle.get_pos()
        distance = math.sqrt((self._x - other_pos[0]) ** 2 + (self._y - other_pos[1]) ** 2)
        return distance < self.get_radius() + circle.get_radius()
            
    def get_dimensions(self):
        return [self._radius, self._radius]
        
    def get_radius(self):
        return self._radius
    
    def set_radius(self, radius):
        self._radius = radius

        
