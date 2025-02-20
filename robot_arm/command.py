from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


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
        self.arm_controller.bottom_turn(self.angle)

    def undo(self):
        self.arm_controller.bottom_turn(-self.angle)


class WristRollCommand(Command):
    def __init__(self, tomoko, *args):
        super().__init__(tomoko)
        self.angle = args[0]

    def execute(self):
        self.arm_controller.wrist_roll(self.angle)

    def undo(self):
        self.arm_controller.wrist_roll(-self.angle)


class LiftMoveCommand(Command):
    def __init__(self, tomoko, *args):
        super().__init__(tomoko)
        self.height = args[0]

    def execute(self):
        print("lift: ", self.arm_controller.lift(self.height))

    def undo(self):
        self.arm_controller.lift(-self.height)


class GripperCloseCommand(Command):
    def __init__(self, tomoko):
        super().__init__(tomoko)

    def execute(self):
        self.arm_controller.set_grip_degree(0)

    def undo(self):
        self.arm_controller.set_grip_degree(70)


class EndPoseMoveCommand(Command):
    def __init__(self, tomoko, *args):
        super().__init__(tomoko)
        assert len(args) == 3
        self.end_pose = args[0]
        self.move_mode = 0x2 if args[1] == "linear" else 0x0
        self.move_speed_rate = args[2]

    def execute(self):
        print("\n\nmove success?", self.arm_controller.end_pose_control(self.end_pose, self.move_mode, self.move_speed_rate))

    def undo(self):
        self.execute()


class SetZeroCommand(Command):
    def __init__(self, tomoko):
        super().__init__(tomoko)

    def execute(self):
        height = 50
        self.arm_controller.lift(height)
        self.arm_controller.set_zero_state()

    def undo(self):
        pass
