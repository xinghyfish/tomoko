import math
import time
from typing import List

from piper_sdk import *


class ArmController:
    def __init__(self):
        self.hand_length = 170.0
        self.gripper_degree = 0
        self.piper = C_PiperInterface("can0")
        self.factor = 1000 # 0.001 degree --> 1 degree
        self.init_end_pose = [55, 0, 203, 0, 85, 0]
        self.init_joint = [0, 0, 0, 0, 0, 0]
        # dir_path = os.path.dirname(__file__)
        # subprocess.run(["bash", dir_path + '/can_activate.sh', 'can0', '1000000'])
        self.init_status()

    def init_status(self):
        self.piper.ConnectPort()
        self.piper.EnableArm(7)
        self.enable_fun()
        self.piper.GripperCtrl(0, 1000, 0x01, 0)

    def enable_fun(self):
        """
        Enable the robotic arm and check the enabling status.
        Try for 5 seconds, and if the enabling times out, exit the program.
        """
        enable_flag = False
        # timeout limit in seconds
        timeout = 5
        start_time = time.time()
        elapsed_time_flag = False
        while not enable_flag:
            elapsed_time = time.time() - start_time
            print("--------------------")
            assert type(self.piper is not None and self.piper == C_PiperInterface)
            enable_flag = self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                          self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                          self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                          self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                          self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                          self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            print("Enable state:", enable_flag)
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            print("--------------------")
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print("Time out")
                elapsed_time_flag = True
                enable_flag = True
                break
            time.sleep(1)
            pass
        if elapsed_time_flag:
            print("Program auto enabling times out, EXIT the program.")
            exit(0)

    def joint_control(self, position: List):
        """Control Joint by 6 joints."""
        joints = list(map(lambda x: round(x * self.factor), position))
        self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        self.piper.JointCtrl(*joints)
        self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        # 轮循，直到机械臂各电机运转到位，停止休眠
        while not self.is_joint_in_position(joints):
            time.sleep(0.01)

    def is_joint_in_position(self, joints: List):
        current_joint_state = self.joint_state()
        epsilon = 1000
        return all([abs(joints[i] - current_joint_state[i] * self.factor) < epsilon for i in range(len(current_joint_state))])

    def joint_state(self):
        """Get current state of joint."""
        s = self.piper.GetArmJointMsgs().joint_state
        joint = [s.joint_1, s.joint_2, s.joint_3, s.joint_4, s.joint_5, s.joint_6]
        return [x / self.factor for x in joint]

    def end_pose_control(self, position: List):
        """Control end pose by 6 arguments."""
        end_pos = list(map(lambda x: round(x * self.factor), position))
        self.piper.MotionCtrl_2(0x01, 0x00, 80, 0x00)
        self.piper.EndPoseCtrl(*end_pos)
        self.piper.MotionCtrl_2(0x01, 0x00, 80, 0x00)
        while not self.is_end_pose_in_position(end_pos):
            time.sleep(0.01)

    def is_end_pose_in_position(self, end_pose: List):
        current_end_pose = self.end_pose_state()
        epsilon = 1000
        return all([abs(end_pose[i] - current_end_pose[i] * self.factor) < epsilon for i in range(3)])

    @staticmethod
    def is_angel_in_position(current_angle: List, desired_angle: List, epsilon: float=1000):
        """Determine whether the rotation angle is in place"""
        # for i in range(len(desired_angle)):
        #     if 180000 - epsilon > desired_angle[i] > epsilon:
        #         if abs(current_angle[i] - desired_angle[i]) > epsilon:
        #             return False
        #     # desired angle in range of [180 - epsilon, 180] U [0, epsilon]
        #     elif desired_angle[i] + epsilon >= 180000:
        #         if epsilon + desired_angle[i] - 180000 < current_angle[i] < desired_angle[i] - epsilon:
        #             return False
        #     else:
        #         if desired_angle[i] + epsilon < current_angle[i] < 180000 + desired_angle[i] - epsilon:
        #             return False
        # return True
        current_angle = list(map(lambda x: x if x >= 0 else x + 180000, current_angle))
        return all(abs(current_angle[i] - desired_angle[i]) <= epsilon for i in range(3))

    def end_pose_state(self):
        """Get current state of end pose."""
        s = self.piper.GetArmEndPoseMsgs().end_pose
        end_pose = [s.X_axis, s.Y_axis, s.Z_axis, s.RX_axis, s.RY_axis, s.RZ_axis]
        return [x / self.factor for x in end_pose]

    def set_grip_degree(self, degree):
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.GripperCtrl(abs(round(degree * self.factor)), 1000, 0x01, 0)
        self.gripper_degree = degree

    @staticmethod
    def end_to_hand(end_pose: List, hand_length: float) -> List:
        """
        由于手眼标定得到的是机械臂末端位姿，而这个位置实际上是我们希望夹取点的位置。
        因此，我们需要根据夹取点和末端的几何关系，对手眼标定后的末端位姿进行调整，使得标定的坐标和夹取点一致，
        调整后的末端位姿作为机械臂末端输入从而进行控制。
        :param end_pose: [x, y, z, Rx, Ry, Rz] 末端位姿六元组
        :param hand_length: 末端到夹取点的距离
        :return: 将当前末端位姿设置为夹取点的实际末端位姿六元组
        """
        x, y, z, Rx, Ry, Rz = end_pose
        nx = x - hand_length * math.cos(Rx)
        ny = y - hand_length * math.cos(Ry)
        nz = z - hand_length * math.cos(Rz)
        return [nx, ny, nz, Rx, Ry, Rz]

    def set_zero_state(self):
        if self.gripper_degree:
            self.set_grip_degree(70)
        self.joint_control(self.init_joint)
        self.end_pose_control(self.init_end_pose)
        self.set_grip_degree(0)
        time.sleep(0.5)


if __name__ == '__main__':
    arm_controller = ArmController()
    # position = [30, 40, -10, 0, -20, 0, 0]
    # arm_controller.joint_control(position)
    # position = [30, 40, -10, 0, -20, 45, 0]
    # arm_controller.joint_control(position)
    # time.sleep(3)
    # position = [30, 40, -10, 0, -20, 0, 0]
    # arm_controller.joint_control(position)
    arm_controller.set_grip_degree(80)
    pos = [0, 0, 0, 0, 0, 0]
    arm_controller.joint_control(pos)
    # pos = [65, 0, 220, 0, 90, 0, 80]
    # arm_controller.end_pose_control(pos)
    # time.sleep(2)
    # pos = [0] * 7
    # arm_controller.joint_control(pos)
    # arm_controller.set_grip_degree(65)
