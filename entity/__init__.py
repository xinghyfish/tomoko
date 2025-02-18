class GraspInfo:
    def __init__(self, target_index=None, end_pose=None):
        self.position = target_index
        self.end_pose = end_pose

class Entity:
    def __init__(self, radius, polar_angle):
        self.radius = radius
        self.polar_angle = polar_angle

class Cup(Entity):
    def __init__(self):
        radius = 32
        polar_angle = 90
        super().__init__(radius, polar_angle)
        self.height = 48


class Can(Entity):
    def __init__(self):
        radius = 28
        polar_angle = 90
        super().__init__(radius, polar_angle)
        self.height = 54

class Teapot(Entity):
    def __init__(self):
        radius = 800
        polar_angle = 110
        super().__init__(radius, polar_angle)
        self.inner_radius = 44
