import logging
import math
import time

import numpy as np
from numpy.ma.core import arctan

from entity import GraspInfo
from entity.entity import *
from jarvis import Jarvis
from robot_arm.command import EndPoseMoveCommand, BottomTurnCommand, WristRollCommand, GraspCommand, \
    JointControlCommand, LiftMoveCommand
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
        thickness = 10
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
        pos = [-35, 10, -10, 0, 10, 0]
        prompt = "small black rectangle on the bottom"

        self.jarvis.arm_controller.joint_control(pos)
        objects_position, _, _, depth_intrin, _ = self.jarvis.detect(prompt)
        boxes = [_[-1] for _ in objects_position]
        left_idx = 0 if boxes[0][0] < boxes[1][0] else 1
        right_idx = 1 - left_idx
        faucets = {'left': left_idx, 'right': right_idx}

        entity = Faucet()

        for key in faucets.keys():
            x, y, distance, mask, _ = objects_position[faucets[key]]
            end_pose, target_index = self.jarvis.pixel_to_3d(x, y, distance, depth_intrin, entity)
            X, Y, Z, RX, RY, RZ = end_pose
            back_factor = Teapot().inner_radius * 2 / math.sqrt(X ** 2 + Y ** 2)
            X *= (1 - back_factor)
            Y *= (1 - back_factor)
            end_pose = [X, Y, Z,RX, RY, RZ]
            self.grasp_info[faucets[key]] = GraspInfo(target_index, end_pose)




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
        teapot_detectable_joints = [0, 0, 0, 0, 10, 0]
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
    tomoko.jarvis.arm_controller.end_pose_control(tomoko.jarvis.arm_controller.init_end_pose)
    # tomoko.scan_desk()
    # tomoko.add_tea_to_teapot("red")


if __name__ == '__main__':
    # time.sleep(5)
    pipeline()