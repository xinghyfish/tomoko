import time
from typing import List

from piper_sdk import *

hand_length = 140.0
arm_radius = 25.0

class ArmController:
    def __init__(self):
        self.gripper_degree = 0
        self.piper = C_PiperInterface("can0")
        self.factor = 1000 # 0.001 degree --> 1 degree
        self.init_end_pose = [55, 0, 203, 0, 90, 0]
        self.init_joint = [0, 0, 0, 0, 0, 0]
        self.init_status()
        self.motion_flag = False

    def init_status(self):
        self.piper.ConnectPort()
        self.piper.EnableArm(7)
        self.enable_fun()
        # self.piper.GripperCtrl(0, 1000, 0x01, 0)

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
            self.piper.GripperCtrl(80, 1000, 0x01, 0)
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

    def joint_control(self, position: List, move_mode=0x1, move_speed_rate=30):
        """Control Joint by 6 joints."""
        joints = list(map(lambda x: round(x * self.factor), position))
        start_joints = self.joint_state()
        self.piper.MotionCtrl_2(0x01, move_mode, move_speed_rate, 0x00)
        self.piper.JointCtrl(*joints)
        # record current joint status
        time_elapsed = 0
        while not self.is_joint_in_position(position):
            time_elapsed += 1
            if time_elapsed >= 10:
                # after 10 iterations but still not position
                if self.is_joint_in_position(start_joints):
                    print("Joint parameters are incorrect.")
                    return False
            time.sleep(0.1)
        return True

    def is_joint_in_position(self, joints: List, error=1.0):
        current_joint_state = self.joint_state()
        return all([abs(joints[i] - current_joint_state[i]) < error for i in range(len(current_joint_state))])

    def joint_state(self):
        """Get current state of joint."""
        s = self.piper.GetArmJointMsgs().joint_state
        joint = [s.joint_1, s.joint_2, s.joint_3, s.joint_4, s.joint_5, s.joint_6]
        return [x / self.factor for x in joint]

    def end_pose_control(self, position: List, move_mode=0x00, move_speed_rate=80) -> bool:
        """
        Control end pose by 6 arguments.
        In this case, [Rx, Ry, Rz] describes the rotation angle of the coordinate system of the arm.
        The rotation order is (by pixel): X -> Y-> Z. NEVER CHANGE THE ORDER.
        :param position: position including [x, y, z, Rx, Ry, Rz]
        :param move_mode:
            - 0x00 position move
            - 0x02 linear move
        :param move_speed_rate: rate of move [0, 100]
        :return: True if end pose control is successful else False
        """
        assert move_mode in [0x00, 0x02]
        end_pos = list(map(lambda x: round(x * self.factor), position))
        start_end_pos = self.end_pose_state()
        self.piper.MotionCtrl_2(0x01, move_mode, move_speed_rate, 0x00)
        self.piper.EndPoseCtrl(*end_pos)
        time_elapsed = 0
        while not self.is_end_pose_in_position(position):
            time.sleep(0.01)
            time_elapsed += 1
            if time_elapsed >= 10:
                if self.is_end_pose_in_position(start_end_pos):
                    print("Unreachable")
                    return True
        return True

    def is_end_pose_in_position(self, end_pose: List, error: float=1.0):
        current_end_pose = self.end_pose_state()
        return all([abs(end_pose[i] - current_end_pose[i]) < error for i in range(3)])

    def is_angel_in_position(self, joints: List, error: float=1.0):
        """Determine whether the rotation angle is in place"""
        current_joints = self.joint_state()
        current_angle = list(map(lambda x: x if x >= 0 else x + 180, current_joints))
        return all(abs(current_angle[i] - joints[i]) <= error for i in range(3))

    def end_pose_state(self):
        """Get current state of end pose."""
        s = self.piper.GetArmEndPoseMsgs().end_pose
        end_pose = [s.X_axis, s.Y_axis, s.Z_axis, s.RX_axis, s.RY_axis, s.RZ_axis]
        return [x / self.factor for x in end_pose]

    def set_grip_degree(self, degree):
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.GripperCtrl(abs(round(degree * self.factor)), 1000, 0x01, 0)
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.gripper_degree = degree
        return True

    def set_zero_state(self):
        if self.gripper_degree:
            self.set_grip_degree(70)
        self.joint_control(self.init_joint, move_mode=0x0, move_speed_rate=20)

    def lift(self, height) -> bool:
        """
        Lift the grasped object. Positive to up and negative to down.
        :param height: distance to lift.
        :return: True if lift is successful else False
        """
        x, y, z, Rx, Ry, Rz = self.end_pose_state()
        z += height
        return self.end_pose_control([x, y, z, Rx, Ry, Rz], move_mode=0x2, move_speed_rate=30)

    def snake_observe(self):
        """
        Make the pose of the arm like snake so that camera can observe the whole object.
        :return: True if reachable else False
        """
        snake_joints = [0, 40, -10, 0, -20, 45, 0]
        return self.joint_control(snake_joints)

    def bottom_turn(self, degree):
        """
        Turn the bottom joint of the arm.
        :param degree: angle in degree
        :return: If the turing is in valid range.
        """
        current_joints = self.joint_state()
        current_joints[0] += degree
        return self.joint_control(current_joints, move_speed_rate=15)

    def wrist_roll(self, angle):
        """
        Turn the last joint (like wrist of arm).
        :param angle: in degree
        :return: if the wrist is in valid range.
        """
        current_joints = self.joint_state()
        current_joints[-1] += angle
        return self.joint_control(current_joints)


if __name__ == '__main__':
    arm_controller = ArmController()
    arm_controller.set_grip_degree(80)
    while True:
        print(arm_controller.end_pose_state())
        # time.sleep(1)
        input()
    # time.sleep(2)
    # arm_controller.lift(100)
    # time.sleep(1)
    # observer_pos = [-45, 10, -10, 0, 10, 0]
    # pos1 = [300.5489204539047, -233.33717115071423, 218.28423826192972, 0, 110, -37.82469282746757]
    # flag = arm_controller.end_pose_control(pos)
    # flag = arm_controller.joint_control(observer_pos)
    # arm_controller.end_pose_control(pos1)
    # pos = [0, 10, -10, 0, 30, -5]
    # arm_controller.joint_control(pos)
    # time.sleep(2)
    #
    # pos = [-35, 10, -10, 0, 10, -5]
    # arm_controller.joint_control(pos)
    # time.sleep(2)
    #
    # pos = [0, 10, -10, 0, 30, -5]
    # arm_controller.joint_control(pos)