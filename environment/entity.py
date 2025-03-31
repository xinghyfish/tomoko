import threading

class Entity:
    def __init__(self, radius, polar_angle):
        self.radius = radius
        self.polar_angle = polar_angle


class TeaCup(Entity):
    def __init__(self):
        radius, polar_angle = 27.5, 90
        super().__init__(radius, polar_angle)
        self.height = 43.3


class TeaCan(Entity):
    def __init__(self):
        radius, polar_angle = 22, 90
        super().__init__(radius, polar_angle)
        self.height = 53.6
        self.cover = 10


class Teapot(Entity):
    def __init__(self):
        radius, polar_angle = 85, 110
        super().__init__(radius, polar_angle)
        self.inner_radius = 45
        self.height = 64.1
        self.cover = 15.5


class Faucet(Entity):
    def __init__(self):
        radius, polar_angle = 0, 110
        super().__init__(radius, polar_angle)
