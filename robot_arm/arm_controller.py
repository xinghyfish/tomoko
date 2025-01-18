import math
import os
import subprocess
from typing import List

from piper_sdk import *
import time


class ArmController:
    def __init__(self):
        self.hand_length = 170.0
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
        使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        """
        enable_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
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
                print("超时....")
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
        joint_6 = round(position[6] * 1000 * 1000)

        self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        self.piper.JointCtrl(*joints)
        self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)

        epsilon = 100
        # 轮循，直到机械臂各电机运转到位，停止休眠
        while not self.in_position(joints):
            time.sleep(0.01)

    def end_pose_control(self, position: List):
        end_pos = [round(position[i] * self.factor) for i in range(6)]
        joint_6 = round(position[6] * self.factor)
        # piper.MotionCtrl_1()
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.EndPoseCtrl(*end_pos)
        self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

    def in_position(self, joints: List):
        current_joint_state = self.piper.GetArmJointMsgs().joint_state
        epsilon = 100
        return abs(current_joint_state.joint_1 - joints[0]) < epsilon and \
            abs(current_joint_state.joint_2 - joints[1]) < epsilon and \
            abs(current_joint_state.joint_3 - joints[2]) < epsilon and \
            abs(current_joint_state.joint_4 - joints[3]) < epsilon and \
            abs(current_joint_state.joint_5 - joints[4]) < epsilon and \
            abs(current_joint_state.joint_6 - joints[5]) < epsilon

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
        # TODO: 将末端坐标转化为夹爪坐标后，末端实际位姿
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
    position = [0, 0, 0, 0, 0, 0, 0]
    arm_controller.joint_control(position)
    position = [165, 0, 270, 0, 90, 0, 80]
    # position = arm_controller.init_end_pose + [10]
    arm_controller.end_pose_control(position)
    time.sleep(3)
    print(arm_controller.piper.GetArmEndPoseMsgs().end_pose)
    position = [0] * 7
    arm_controller.joint_control(position)
