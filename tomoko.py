import math
import time

import numpy as np
from numpy.ma.core import arctan

from entity import GraspInfo, Teapot, Cup
from jarvis import Jarvis
from vlm.image_utils import center_of_mask
from vlm.lang_sam_demo import show_masks_on_image


class Tomoko:
    def __init__(self):
        self.jarvis = Jarvis()
        # position in 3D space [x, y, z] and corresponding grasp end pose
        self.grasp_info = dict()
        self.label_to_entity = dict()
        self.current_grasp_object = {
            "teapot": Teapot(),
        }

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
        obj_lst = obj_lst[0]
        xe, ye, dis, mask1 = obj_lst[0]

        # segment the part of handle from the whole
        teapot_depth = depth_image * mask1
        replaced_teapot_depth = np.where(teapot_depth == 0, np.max(teapot_depth), teapot_depth)
        # find the index of the nearest part
        min_index = np.unravel_index(np.argmin(replaced_teapot_depth), replaced_teapot_depth.shape)
        min_depth = teapot_depth[min_index]
        thickness = 20
        # so this is the mask of handle
        wooden_mask = np.where(teapot_depth - min_depth <= thickness, 1, 0)[np.newaxis, :]
        show_masks_on_image(color_image, wooden_mask)
        # time to check it out
        time.sleep(3)

        y, x = center_of_mask(wooden_mask)[0]
        distance = self.jarvis.realsense_camera.try_get_object_distance(x, y, depth_frame)
        prompt = "teapot"
        end_pose, target_index = self.jarvis.pixel_to_3d(x, y, distance, depth_intrin, prompt)
        self.grasp_info[prompt] = GraspInfo(target_index, end_pose)

    def drop_tea(self):
        label = "teapot"
        assert self.grasp_info.get(label)
        teapot_grasp_info = self.grasp_info[label]
        assert isinstance(teapot_grasp_info, GraspInfo)
        x, y, z = teapot_grasp_info.position
        teapot = self.label_to_entity[label]
        assert isinstance(teapot, Teapot)
        theta = teapot.polar_angle
        vec = np.array([
            x * math.sin(theta) / math.sqrt(x ** 2 + y ** 2),
            y * math.sin(theta) / math.sqrt(x ** 2 + y ** 2),
            math.cos(theta)
        ])
        height, lift_height = z, 100
        _, _, z, Rx, Ry, Rz = self.jarvis.arm_controller.end_pose_state()
        label = "cup"
        cup = self.label_to_entity[label]
        assert isinstance(cup, Cup)
        drop_height = z - height + lift_height + cup.height / 2
        self.jarvis.arm_controller.lift(drop_height)
        r = teapot.inner_radius
        center_of_inner_space = np.array([x, y, z]) + r * vec / math.sin(theta)
        angle = arctan(y / x)
        self.jarvis.arm_controller.bottom_turn(angle)
        x, y, z = center_of_inner_space.tolist()
        end_pose = self.jarvis.end_pose_transform(x, y, drop_height, cup.radius, 90)
        self.jarvis.arm_controller.end_pose_control(end_pose)
        self.jarvis.arm_controller.wrist_roll(180)
