from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sympy import andre


class Command(ABC):
    def __init__(self, tomoko):
        if TYPE_CHECKING:
            from tomoko import Tomoko
            assert isinstance(tomoko, Tomoko)
        self.arm_controller = tomoko.jarvis.arm_controller

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass


class BottomTurnCommand(Command):
    def __init__(self, tomoko, *args):
        super().__init__(tomoko)
        self.angle = args[0]

    def execute(self):
        return self.arm_controller.bottom_turn(self.angle)

    def undo(self):
        return self.arm_controller.bottom_turn(-self.angle)

    def __str__(self):
        return f"Command(Bottom Turn) - {self.angle} degree"


class WristRollCommand(Command):
    def __init__(self, tomoko, *args):
        super().__init__(tomoko)
        self.angle = args[0]

    def execute(self):
        return self.arm_controller.wrist_roll(self.angle)

    def undo(self):
        return self.arm_controller.wrist_roll(-self.angle)

    def __str__(self):
        return f"Command(Wrist Roll) - {self.angle} degree"


class LiftMoveCommand(Command):
    def __init__(self, tomoko, *args):
        super().__init__(tomoko)
        self.height = args[0]

    def execute(self):
        return self.arm_controller.lift(self.height)

    def undo(self):
        return self.arm_controller.lift(-self.height)

    def __str__(self):
        return f"Command(Lift) - {self.height} mm"


class GripperCloseCommand(Command):
    def __init__(self, tomoko):
        super().__init__(tomoko)
        self.close_flag = True

    def execute(self):
        self.close_flag = True
        return self.arm_controller.set_grip_degree(0)

    def undo(self):
        self.close_flag = False
        return self.arm_controller.set_grip_degree(70)

    def __str__(self):
        return f"Command(GripperClose)"


class GripperOpenCommand(Command):
    def __init__(self, tomoko):
        super().__init__(tomoko)
        self.open_flag = True

    def execute(self):
        self.open_flag = True
        return self.arm_controller.set_grip_degree(70)

    def undo(self):
        self.open_flag = False
        return self.arm_controller.set_grip_degree(0)

    def __str__(self):
        return f"Command(GripperOpen)"


class GraspCommand(Command):
    def __init__(self, tomoko, target, end_pose):
        super().__init__(tomoko)
        self.target = target
        self.end_pose = end_pose

    def execute(self):
        return self.arm_controller.set_grip_degree(70) and \
                self.arm_controller.end_pose_control(self.end_pose) and \
                self.arm_controller.set_grip_degree(0)

    def undo(self):
        return self.arm_controller.set_grip_degree(0) and \
            self.arm_controller.end_pose_control(self.end_pose) and \
            self.arm_controller.set_grip_degree(70)

    def __str__(self):
        return f"Command(Grasp) - {self.target}"



class EndPoseMoveCommand(Command):
    def __init__(self, tomoko, *args):
        super().__init__(tomoko)
        assert len(args) == 3
        self.end_pose = args[0]
        self.move_mode = 0x2 if args[1] == "linear" else 0x0
        self.move_speed_rate = args[2]

    def execute(self):
        return self.arm_controller.end_pose_control(self.end_pose, self.move_mode, self.move_speed_rate)

    def undo(self):
        return self.execute()

    def __str__(self):
        return f"Command(End Pose) - [{self.end_pose}, {self.move_mode}, {self.move_speed_rate}]"


class SetZeroCommand(Command):
    def __init__(self, tomoko):
        super().__init__(tomoko)

    def execute(self):
        return self.arm_controller.lift(50) and \
            self.arm_controller.end_pose_control(self.arm_controller.init_end_pose)

    def undo(self):
        return self.execute()

    def __str__(self):
        return f"Command(Set Zero State)"
