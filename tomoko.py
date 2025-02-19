import math
import time

import numpy as np
from numpy.ma.core import arctan

from entity import GraspInfo, Teapot, TeaCup, TeaCan
from jarvis import Jarvis
from vlm.image_utils import center_of_mask
from vlm.lang_sam_demo import show_masks_on_image


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
        xe, ye, dis, mask1 = obj_lst[0]

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
        return target_index

    def drop_tea(self):
        label = "teapot"
        assert self.grasp_info.get(label)
        teapot_grasp_info = self.grasp_info[label]
        assert isinstance(teapot_grasp_info, GraspInfo)
        x_teapot_handle, y_teapot_handle, z_teapot_handle = teapot_grasp_info.position
        print("teapot position: ", x_teapot_handle, y_teapot_handle, z_teapot_handle)

        teapot = self.label_to_entity[label]
        assert isinstance(teapot, Teapot)
        theta = math.radians(teapot.polar_angle)
        vec = np.array([
            x_teapot_handle * math.sin(theta) / math.sqrt(x_teapot_handle ** 2 + y_teapot_handle ** 2),
            y_teapot_handle * math.sin(theta) / math.sqrt(x_teapot_handle ** 2 + y_teapot_handle ** 2),
            math.cos(theta)
        ])
        center_of_inner_space = np.array([x_teapot_handle, y_teapot_handle, z_teapot_handle]) + (teapot.inner_radius + teapot.radius) * vec / math.sin(theta)
        x_teapot, y_teapot, z_teapot = center_of_inner_space.tolist()
        label = "can"
        can = self.label_to_entity[label]
        assert isinstance(can, TeaCan)
        angle = math.degrees(arctan((y_teapot + can.height / 2) / x_teapot))
        xOy_distance_teapot = math.sqrt(x_teapot ** 2 + y_teapot ** 2)

        can_grasp_info = self.grasp_info[self.current_grasp_object]
        assert isinstance(can_grasp_info, GraspInfo)
        x_can, y_can, z_can = can_grasp_info.position
        xOy_distance_can = math.sqrt(x_can ** 2 + y_can ** 2)
        factor = xOy_distance_teapot / xOy_distance_can
        x_target, y_target = factor * x_can, factor * y_can
        drop_height = z_teapot_handle + 150
        end_pose = self.jarvis.end_pose_transform(x_target, y_target, drop_height, can.radius, 90)
        print("ready to drop tea end pose: ", end_pose)
        print("corresponding coordinate: ", x_target, y_target, drop_height)
        self.jarvis.arm_controller.end_pose_control(end_pose, move_mode=0x02, move_speed_rate=30)
        time.sleep(3)
        self.jarvis.arm_controller.bottom_turn(angle)
        time.sleep(5)
        self.jarvis.arm_controller.wrist_roll(160)
        time.sleep(1)
        self.jarvis.arm_controller.wrist_roll(-160)

    def detect_can(self, color):
        prompt = f"{color} can"
        objects_position, _, _, depth_intrin, _ = self.jarvis.detect(prompt)
        assert objects_position, len(objects_position) == 1
        x, y, distance, mask = objects_position[0]
        entity = TeaCan()
        end_pose, target_index = self.jarvis.pixel_to_3d(x, y, distance, depth_intrin, entity)
        self.grasp_info[prompt] = GraspInfo(target_index, end_pose)
        return target_index

    def grasp(self, prompt):
        end_pose = self.grasp_info[prompt].end_pose
        time.sleep(2)
        self.jarvis.arm_controller.set_grip_degree(70)
        time.sleep(2)
        self.jarvis.arm_controller.end_pose_control(end_pose, move_speed_rate=50)
        time.sleep(2)
        self.current_grasp_object = prompt
        self.jarvis.arm_controller.set_grip_degree(0)

def pipeline():
    tomoko = Tomoko()
    teapot_detectable_joints = [0, 0, 0, 0, 10, 0]
    tomoko.jarvis.arm_controller.joint_control(teapot_detectable_joints)
    time.sleep(2)
    print(tomoko.detect_teapot())
    tea_can_detectable_joints = [35, 10, -20, 0, 30, 0]
    tomoko.jarvis.arm_controller.joint_control(tea_can_detectable_joints)
    time.sleep(3)
    print(tomoko.detect_can("red"))
    tomoko.jarvis.arm_controller.set_grip_degree(70)
    time.sleep(2)
    tomoko.grasp("red can")
    time.sleep(5)
    tomoko.drop_tea()


if __name__ == '__main__':
    pipeline()