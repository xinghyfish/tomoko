import threading


class GraspInfo:
    def __init__(self, target_index=None, end_pose=None):
        self.position = target_index
        self.end_pose = end_pose


class Entity:
    def __init__(self, radius, polar_angle):
        self.radius = radius
        self.polar_angle = polar_angle


class TeaCup(Entity):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(TeaCup, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        radius, polar_angle = 32, 90
        super().__init__(radius, polar_angle)
        self.height = 48
        _instance = TeaCup()

class TeaCan(Entity):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(TeaCan, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        radius, polar_angle = 35, 90
        super().__init__(radius, polar_angle)
        self.height = 54


class Teapot(Entity):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Teapot, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        radius, polar_angle = 80, 110
        super().__init__(radius, polar_angle)
        self.inner_radius = 44
