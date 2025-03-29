import threading

class Entity:
    def __init__(self, radius, polar_angle):
        self.radius = radius
        self.polar_angle = polar_angle


class TeaCup(Entity):
    def __init__(self):
        radius, polar_angle = 32, 90
        super().__init__(radius, polar_angle)
        self.height = 48


class TeaCan(Entity):
    def __init__(self):
        radius, polar_angle = 35, 90
        super().__init__(radius, polar_angle)
        self.height = 24


class Teapot(Entity):
    def __init__(self):
        radius, polar_angle = 85, 110
        super().__init__(radius, polar_angle)
        self.inner_radius = 45


class Faucet(Entity):
    def __init__(self):
        radius, polar_angle = 0, 110
        super().__init__(radius, polar_angle)
