import time
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
        flag = self.arm_controller.bottom_turn(self.angle)
        time.sleep(2)
        return flag

    def undo(self):
        flag = self.arm_controller.bottom_turn(-self.angle)
        time.sleep(2.5)
        return flag

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
        flag = self.arm_controller.lift(self.height)
        time.sleep(0.5)
        return flag

    def undo(self):
        flag = self.arm_controller.lift(-self.height)
        time.sleep(0.5)
        return flag

    def __str__(self):
        return f"Command(Lift) - {self.height} mm"


class GraspCommand(Command):
    def __init__(self, tomoko, target, end_pose):
        super().__init__(tomoko)
        self.target = target
        self.end_pose = end_pose
        self.last_joints = None

    def execute(self):
        self.last_joints = self.arm_controller.joint_state()
        gripper_open = self.arm_controller.set_grip_degree(70)
        time.sleep(0.2)
        end_pose_control = self.arm_controller.end_pose_control(self.end_pose)
        time.sleep(0.2)
        gripper_close = self.arm_controller.set_grip_degree(0)
        time.sleep(0.2)
        return all([gripper_open, end_pose_control, gripper_close])

    def undo(self):
        gripper_open = self.arm_controller.set_grip_degree(70)
        time.sleep(0.5)
        height = 20
        lift = self.arm_controller.lift(height)
        x, y, z, Rx, Ry, Rz = self.arm_controller.end_pose_state()
        x -= 40 * abs(x) / x
        y -= 40 * abs(y) / y
        z += 40
        time.sleep(0.2)
        via_flag1 = self.arm_controller.end_pose_control([x, y, z, Rx, Ry, Rz], 0x02, 30)
        time.sleep(0.2)
        joint_state = self.arm_controller.joint_state()
        joint_state[1] -= 15
        via_flag2 = self.arm_controller.joint_control(joint_state)
        time.sleep(0.2)
        init_joints_control = self.arm_controller.joint_control(self.last_joints)
        time.sleep(0.2)
        return all([gripper_open, lift, via_flag1, init_joints_control])

    def __str__(self):
        return f"Command(Grasp) - {self.target}"


class EndPoseMoveCommand(Command):
    def __init__(self, tomoko, *args):
        super().__init__(tomoko)
        assert len(args) == 3
        self.end_pose = args[0]
        self.move_mode = 0x2 if args[1] == "linear" else 0x0
        self.move_speed_rate = args[2]
        self.last_end_pose = None

    def execute(self):
        self.last_end_pose = self.arm_controller.end_pose_state()
        time.sleep(0.1)
        flag = self.arm_controller.end_pose_control(self.end_pose, self.move_mode, self.move_speed_rate)
        if self.move_mode == 0x2:
            time.sleep(3)
        time.sleep(0.1)
        return flag

    def undo(self):
        flag = self.arm_controller.end_pose_control(self.last_end_pose, self.move_mode, self.move_speed_rate)
        if self.move_mode == 0x2:
            time.sleep(3)
        return flag

    def __str__(self):
        return f"Command(End Pose) - {[round(_, 3) for _ in self.end_pose]}"


class JointControlCommand(Command):
    def __init__(self, tomoko, *args):
        super().__init__(tomoko)
        self.last_state = None
        assert len(args) == 2
        self.joints = args[0]
        self.move_speed_rate = args[1]

    def execute(self):
        self.last_state = self.arm_controller.joint_state()
        return self.arm_controller.joint_control(self.joints, move_speed_rate=self.move_speed_rate)

    def undo(self):
        return self.arm_controller.joint_control(self.last_state, move_speed_rate=self.move_speed_rate)

    def __str__(self):
        return f"Command(Joints) - {[round(_, 3) for _ in self.joints]}"


class TimerCommand(Command):
    def __init__(self, tomoko, *args):
        """time elapse in second"""
        super().__init__(tomoko)
        assert len(args) == 1
        self.seconds = args[0]

    def execute(self):
        time.sleep(self.seconds)
        return True

    def undo(self):
        return True

    def __repr__(self):
        return f"Command(Timer) - elapse {self.seconds} seconds"


class GripperCommand(Command):
    def __init__(self, tomoko, *args):
        super().__init__(tomoko)
        self.degree = args[0]

    def execute(self):
        flag = self.arm_controller.set_grip_degree(self.degree)
        time.sleep(0.2)
        return flag

    def undo(self):
        flag = self.arm_controller.set_grip_degree(0 if self.degree else 70)
        time.sleep(0.2)
        return flag


class HeadUpCommand(Command):
    def __init__(self, tomoko, degree=10):
        super().__init__(tomoko)
        self.degree = degree

    def execute(self):
        joints = self.arm_controller.joint_state()
        joints[1] -= self.degree
        flag = self.arm_controller.joint_control(joints)
        time.sleep(1)
        return flag

    def undo(self):
        joints = self.arm_controller.joint_state()
        joints[1] += self.degree
        flag = self.arm_controller.joint_control(joints)
        return flag