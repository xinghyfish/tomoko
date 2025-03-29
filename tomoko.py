import logging
import math
import time

import numpy as np
from numpy.ma.core import arctan

from entity import GraspInfo
from entity.entity import *
from jarvis import Jarvis
from robot_arm.command import EndPoseMoveCommand, BottomTurnCommand, WristRollCommand, GraspCommand, \
    JointControlCommand, LiftMoveCommand, TimerCommand
from robot_arm.pipeline import Pipeline
from vlm.image_utils import center_of_mask
from vlm.lang_sam_demo import show_masks_on_image

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class Tomoko:
    def __init__(self):
        self.jarvis = Jarvis()
        # position in 3D space [x, y, z] and corresponding grasp end pose
        self.grasp_info = dict()
        self.label_to_entity = {
            "teapot": Teapot(),
            "can": TeaCan(),
        }
        self.current_grasp_object = None

    def detect_teapot(self):
        """
        Detect the position of the center of teapot handle.
        :return:
            - (x, y): coordinates of the center of teapot handle
            - distance: distance of the center of teapot handle
            - depth_intrin: depth intrin of current frame
        """
        # detect the whole teapot
        prompt = "yellow circle in teapot"
        obj_lst, color_image, depth_image, depth_intrin, depth_frame = self.jarvis.detect(prompt)
        assert obj_lst, len(obj_lst) == 1
        xe, ye, dis, mask1, _ = obj_lst[0]

        # segment the part of handle from the whole
        teapot_depth = depth_image * mask1
        replaced_teapot_depth = np.where(teapot_depth == 0, 10000, teapot_depth)
        # find the index of the nearest part
        min_index = np.unravel_index(np.argmin(replaced_teapot_depth), replaced_teapot_depth.shape)
        min_depth = teapot_depth[min_index]
        thickness = 20
        # so this is the mask of handle
        wooden_mask = np.where(abs(teapot_depth - min_depth) <= thickness, 1, 0)[np.newaxis, :]
        show_masks_on_image(color_image, wooden_mask)
        # time to check it out
        time.sleep(3)

        y, x = center_of_mask(wooden_mask)[0]
        distance = self.jarvis.realsense_camera.try_get_object_distance(x, y, depth_frame)
        entity = Teapot()
        end_pose, target_index = self.jarvis.pixel_to_3d(x, y, distance, depth_intrin, entity)
        prompt = "teapot"
        self.grasp_info[prompt] = GraspInfo(target_index, end_pose)
        logging.info(f"{prompt} position: {target_index}")
        return target_index

    def detect_can(self, color):
        prompt = f"{color} can"
        objects_position, _, _, depth_intrin, _ = self.jarvis.detect(prompt)
        assert objects_position, len(objects_position) == 1
        x, y, distance, mask, _ = objects_position[0]
        entity = TeaCan()
        end_pose, target_index = self.jarvis.pixel_to_3d(x, y, distance, depth_intrin, entity)
        self.grasp_info[prompt] = GraspInfo(target_index, end_pose)
        logging.info(f"{prompt} position: {target_index}")
        return target_index

    def detect_water_dispenser(self):
        prompt = "small black rectangle on the bottom"

        objects_position, _, _, depth_intrin, _ = self.jarvis.detect(prompt)
        boxes = [_[-1] for _ in objects_position]
        print(boxes)
        left_idx = 0 if boxes[0][0] < boxes[1][0] else 1
        right_idx = 1 - left_idx
        faucets = {'left': left_idx, 'right': right_idx}
        entity = Faucet()

        for key in faucets.keys():
            x, y, distance, mask, _ = objects_position[faucets[key]]
            end_pose, target_index = self.jarvis.pixel_to_3d(x, y, distance, depth_intrin, entity)
            self.grasp_info[key] = GraspInfo(target_index, end_pose)

    def add_water(self, side):
        """
        Grasp teapot, push faucet and get water. Specify direction to get water
        with different temperature.
        :param side: left = # Hot, right = # Cold
        """
        # 0. satisfy the prerequisites
        assert self.grasp_info.get('teapot') is not None    # detect_teapot is done
        assert self.grasp_info.get('left') is not None      # detect_faucet is done
        assert self.grasp_info.get('right') is not None

        # 1. grasp teapot
        add_water_pipeline = Pipeline('add water')
        joints = [0, 10, -20, 0, 30, -5]
        add_water_pipeline.add_command(JointControlCommand(self, joints, 20))
        teapot_grasp_end_pose = self.grasp_info['teapot'].end_pose
        add_water_pipeline.add_command(GraspCommand(self, 'teapot', teapot_grasp_end_pose))

        # 2. via moves states
        end_pose_faucet = self.grasp_info[side].end_pose
        X_faucet, Y_faucet, Z_faucet, RX_faucet, RY_faucet, RZ_faucet = end_pose_faucet
        X_teapot, Y_teapot, Z_teapot, RX_teapot, RY_teapot, RZ_teapot = teapot_grasp_end_pose
        # via_end_pose = [X_teapot, Y_teapot, Z_teapot , RX_teapot, RY_teapot, RZ_teapot]

        # add_water_pipeline.add_command(EndPoseMoveCommand(self, via_end_pose, "linear", 20))
        add_water_pipeline.add_command(LiftMoveCommand(self, 10))

        # X_back, Y_back, Z_back = X_teapot * 0.5, Y_teapot * 0.5, Z_teapot
        # back_end_pose = [X_back, Y_back, Z_back, RX_teapot, RY_teapot, RZ_teapot]
        theta_back = math.atan(Y_teapot / X_teapot)
        theta_faucet = math.atan(Y_faucet / X_faucet)


        delta_theta = math.degrees(theta_faucet - theta_back)
        add_water_pipeline.add_command(BottomTurnCommand(self, delta_theta))

        # get water
        delta_distance = 118   # push forward distance to get water
        delta_teapot = -(Teapot().radius * 2) - 10
        distance_push_faucet = delta_teapot + delta_distance
        X_push_faucet, Y_push_faucet, Z_push_faucet = (
            X_faucet + distance_push_faucet * math.cos(theta_faucet),
            Y_faucet + distance_push_faucet * math.sin(theta_faucet),
            Z_faucet - 10
        )
        end_pose_push_faucet = [X_push_faucet, Y_push_faucet, Z_push_faucet, RX_faucet, RY_faucet, RZ_faucet]

        add_water_pipeline.add_command(EndPoseMoveCommand(self, end_pose_push_faucet, "linear", 20))
        wait_add_water_elapse_second = 2   # waiting...
        add_water_pipeline.add_command(TimerCommand(self, wait_add_water_elapse_second))

        if add_water_pipeline.run() and add_water_pipeline.undo():
            return True


    def add_tea_to_teapot(self, color):
        """
        Sub-task 1: Add tea from tea can to teapot.
        :param color: color of the tea can
        :return:
        """
        init_joints = self.jarvis.arm_controller.joint_state()
        via_joints = [45, 10, -20, 0, 30, 0]
        self.jarvis.arm_controller.joint_control(via_joints)
        grasp_target = f"{color} can"
        self.current_grasp_object = grasp_target
        end_pose_can = self.grasp_info[grasp_target].end_pose

        label = "teapot"
        wrist_turn_angle = 160
        drop_tea_pipeline = Pipeline("drop tea")
        teapot_grasp_info = self.grasp_info[label]
        can_grasp_info = self.grasp_info[self.current_grasp_object]
        x_teapot_handle, y_teapot_handle, z_teapot_handle = teapot_grasp_info.position
        teapot = self.label_to_entity[label]
        can = self.label_to_entity["can"]
        assert self.grasp_info.get(label)
        assert isinstance(teapot_grasp_info, GraspInfo)
        assert isinstance(teapot, Teapot)
        assert isinstance(can, TeaCan)
        assert isinstance(can_grasp_info, GraspInfo)

        # calculate center of teapot
        theta = math.radians(teapot.polar_angle)
        vec = np.array([
            x_teapot_handle * math.sin(theta) / math.sqrt(x_teapot_handle ** 2 + y_teapot_handle ** 2),
            y_teapot_handle * math.sin(theta) / math.sqrt(x_teapot_handle ** 2 + y_teapot_handle ** 2),
            math.cos(theta)
        ])
        center_of_inner_space = np.array([x_teapot_handle, y_teapot_handle, z_teapot_handle]) + (
                    teapot.inner_radius + teapot.radius) * vec / math.sin(theta)
        x_teapot, y_teapot, z_teapot = center_of_inner_space.tolist()

        # calculate the coordinate above the teapot
        x_can, y_can, z_can = can_grasp_info.position
        angle_can = math.degrees(arctan(y_can / x_can))
        angle_teapot = math.degrees(arctan((y_teapot + can.height / 2) / x_teapot))
        xOy_distance_teapot = math.sqrt(x_teapot ** 2 + y_teapot ** 2)
        x_can, y_can, z_can = can_grasp_info.position
        xOy_distance_can = math.sqrt(x_can ** 2 + y_can ** 2)
        factor = xOy_distance_teapot / xOy_distance_can
        x_target, y_target = factor * x_can, factor * y_can
        drop_height = z_teapot_handle + 150
        end_pose = self.jarvis.end_pose_transform(x_target, y_target, drop_height, can.radius, 90)

        drop_tea_pipeline.add_command(GraspCommand(self, grasp_target, end_pose_can))
        drop_tea_pipeline.add_command(EndPoseMoveCommand(self, end_pose, "p", 20))
        drop_tea_pipeline.add_command(BottomTurnCommand(self, angle_teapot - angle_can))
        drop_tea_pipeline.add_command(WristRollCommand(self, wrist_turn_angle))

        # perform the whole pipeline
        drop_tea_pipeline.run()
        # this is a time-consuming operation
        time.sleep(2)
        # go back
        drop_tea_pipeline.undo()

        self.jarvis.arm_controller.lift(10)
        self.jarvis.arm_controller.joint_control(via_joints)
        self.jarvis.arm_controller.joint_control(init_joints)

    def scan_desk(self):
        teapot_detectable_joints = [0, 10, -20, 0, 30, 0]
        self.jarvis.arm_controller.joint_control(teapot_detectable_joints)
        self.detect_teapot()
        tea_can_detectable_joints = [40, 10, -20, 0, 30, 0]
        time.sleep(3)
        # suppose color
        colors = ["silver", "red", "yellow"]
        for color in colors:
            self.jarvis.arm_controller.joint_control(tea_can_detectable_joints)
            time.sleep(1)
            self.detect_can(color)
            print(color)
            print(self.jarvis.arm_controller.joint_state())
        self.jarvis.arm_controller.joint_control(teapot_detectable_joints)


def pipeline():
    tomoko = Tomoko()
    pos = [0, 10, -10, 0, 30, -5]
    tomoko.jarvis.arm_controller.joint_control(pos)
    time.sleep(2)
    tomoko.detect_teapot()

    pos = [-40, 10, -10, 0, 10, -5]
    tomoko.jarvis.arm_controller.joint_control(pos)
    time.sleep(2)
    tomoko.detect_water_dispenser()
    time.sleep(1)

    pos = [0, 10, -10, 0, 30, -5]
    tomoko.jarvis.arm_controller.joint_control(pos)

    for info in tomoko.grasp_info.items():
        print(info)

    # tomoko.jarvis.arm_controller.joint_control([0] * 6)
    tomoko.add_water('left')
    # tomoko.scan_desk()
    # tomoko.add_tea_to_teapot("red")


if __name__ == '__main__':
    # time.sleep(5)
    pipeline()