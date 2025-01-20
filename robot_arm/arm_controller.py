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
        self.current_joint = [None] * 6
        self.init_end_pose = [55, 0, 203, 0, 85, 0]
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
        Enable arm and detect status of
        使能机械臂并检测使能状态,尝试 5s,如果使能超时则退出程序
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
            print("使能状态:", enable_flag)
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
            print("程序自动使能超时,退出程序")
            exit(0)

    def joint_control(self, position: List):
        joints = [round(position[i] * self.factor) for i in range(6)]
        joint_6 = round(position[6] * 1000)
        self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        self.piper.JointCtrl(*joints)
        self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        # 轮循，直到机械臂各电机运转到位，停止休眠
        while not self.is_joint_in_position(joints):
            time.sleep(0.01)

    def is_joint_in_position(self, joints: List):
        current_joint_state = self.joint_state()
        epsilon = 100
        return all([abs(joints[i] - current_joint_state[i]) < epsilon for i in range(len(current_joint_state))])

    def joint_state(self):
        """Get current state of joint."""
        s = self.piper.GetArmJointMsgs().joint_state
        return [s.joint_1, s.joint_2, s.joint_3, s.joint_4, s.joint_5, s.joint_6]

    def end_pose_control(self, position: List):
        """Control end pose by 6 arguments."""
        end_pos = [round(position[i] * self.factor) for i in range(6)]
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.EndPoseCtrl(*end_pos)
        while not self.is_end_pose_in_position(end_pos):
            time.sleep(0.01)

    def is_end_pose_in_position(self, end_pose: List):
        current_end_pose = self.end_pose_state()
        epsilon = 1000
        print(current_end_pose)
        return all([abs(end_pose[i] - current_end_pose[i]) < epsilon for i in range(len(current_end_pose))])

    def end_pose_state(self):
        """Get current state of end pose."""
        s = self.piper.GetArmEndPoseMsgs().end_pose
        return [s.X_axis, s.Y_axis, s.Z_axis, s.RX_axis, s.RY_axis, s.RZ_axis]

    def set_grip_degree(self, degree):
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.GripperCtrl(abs(round(degree * self.factor)), 1000, 0x01, 0)

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


if __name__ == '__main__':
    arm_controller = ArmController()
    # position = [30, 40, -10, 0, -20, 0, 0]
    # arm_controller.joint_control(position)
    # position = [30, 40, -10, 0, -20, 45, 0]
    # arm_controller.joint_control(position)
    # time.sleep(3)
    # position = [30, 40, -10, 0, -20, 0, 0]
    # arm_controller.joint_control(position)
    pos = [0, 0, 0, 0, 0, 0, 0]
    arm_controller.joint_control(pos)
    pos = [65, 0, 220, 0, 90, 0, 80]
    arm_controller.end_pose_control(pos)
    time.sleep(2)
    pos = [0] * 7
    arm_controller.joint_control(pos)
    arm_controller.set_grip_degree(65)
