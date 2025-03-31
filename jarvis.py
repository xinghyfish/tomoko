import math
import time

import PIL
from PIL import Image
from numpy.ma.core import arctan

from realsense_camera.realsense_camera import RealsenseCamera
from algorithm import eye2arm_transform, end_pose_transform
from robot_arm.arm_controller import ArmController
from perception.vision import image_utils
from perception.vision.image_utils import show_masks_on_image, show_box_on_image
from perception.vision.vision_model import VisionModel


class Jarvis:
    def __init__(self):
        self.arm_controller = ArmController()
        self.arm_state = "horizontal"
        print("Loading vision model...")
        self.vision_model = VisionModel()
        print("Loading realsense camera...")
        self.realsense_camera = RealsenseCamera()
        self.grasp_pose = {
            "black cup": [28, 90],
            "red cup": [28, 90],
            "yellow cup": [22.5, 90],
            "silver cup": [28, 90],
            "teapot": [80.0, 110],
        }

    def segment(self, text_prompt: str):
        # count the times of invalid detect
        # RealSense camera will return 0 when the depth is invalid (undetectable/too far/too near)
        invalid_count, invalid_threshold, distance, x, y = 0, 4, 0.0, -1, -1

        color_image, depth_intrin, depth_image, depth_frame = [None] * 4
        objects_position = []
        while invalid_count < invalid_threshold and distance == 0.0:
            objects_position.clear()
            color_image, depth_image, depth_intrin, depth_frame = self.realsense_camera.get_aligned_images()
            results = self.vision_model.segment(PIL.Image.fromarray(color_image), text_prompt)
            masks, boxes = results['masks'], results['boxes']

            show_masks_on_image(PIL.Image.fromarray(color_image), masks)
            for i, (y, x) in enumerate(image_utils.center_of_mask(masks)):
                distance = self.realsense_camera.get_pixel_distance(x, y, depth_frame)
                if distance:
                    objects_position.append((x, y, distance, masks[i], boxes[i]))
            if objects_position:
                break
            else:
                print("Invalid frame. Try again.")
                invalid_count += 1

        if invalid_threshold == invalid_count:
            distance = self.realsense_camera.try_get_object_distance(x, y, depth_frame)
            if not distance:
                print("Unable to grasp target object.")
                return None

        return objects_position, color_image, depth_image, depth_intrin, depth_frame

    def detect(self, class_name: str):
        # count the times of invalid detect
        # RealSense camera will return 0 when the depth is invalid (undetectable/too far/too near)
        invalid_count, invalid_threshold, distance, x, y = 0, 4, 0.0, -1, -1

        color_image, depth_intrin, depth_image, depth_frame = [None] * 4
        objects_position = []
        while invalid_count < invalid_threshold and distance == 0.0:
            objects_position.clear()
            color_image, depth_image, depth_intrin, depth_frame = self.realsense_camera.get_aligned_images()
            results = self.vision_model.detect(PIL.Image.fromarray(color_image), class_name)

            show_box_on_image(color_image, class_name, results)
            centers = []
            for info in results:
                x1, y1, x2, y2 = info['box']
                centers.append(((y1 + y2) >> 1, (x1 + x2) >> 1))
            for i, (y, x) in enumerate(centers):
                distance = self.realsense_camera.get_pixel_distance(x, y, depth_frame)
                if distance:
                    objects_position.append((x, y, distance, results[i]['box']))
            if objects_position:
                break
            else:
                print("Invalid frame. Try again.")
                invalid_count += 1

        if invalid_threshold == invalid_count:
            distance = self.realsense_camera.try_get_object_distance(x, y, depth_frame)
            if not distance:
                print("Unable to grasp target object.")
                return None

        return objects_position, color_image, depth_image, depth_intrin, depth_frame

    def pixel_to_3d(self, x, y, distance, depth_intrin):
        """
        Grasp target object by text prompt.
        :param x: x coordinate of center of object surface
        :param y: y coordinate of center of object surface
        :param distance: distance of (x, y) and deep camera
        :param depth_intrin: depth camera intrinsics
        :return:
            - end_pose: end pose to grasp target object
            - target_index: 3D coordinate of target object
        """
        px, py, pz, Rx, Ry, Rz = self.arm_controller.end_pose_state()
        T_c = RealsenseCamera.get_coordinate_3d(x, y, distance, depth_intrin) + [1]
        R_g = eye2arm_transform.euler2quaternion([Rx, Ry, Rz])
        T_g = eye2arm_transform.mm2m([px, py, pz])
        target_index = eye2arm_transform.get_target_index(R_g, T_g, T_c).tolist()
        target_index = eye2arm_transform.m2mm(target_index)
        # should be generated by some of Grasp Network
        # but here we temporarily use a dict
        return target_index

    def set_zero(self):
        self.arm_controller.set_zero_state()
        time.sleep(1)


if __name__ == '__main__':
    jarvis = Jarvis()
    jarvis.arm_controller.lift(100)
    time.sleep(2)
    # jarvis.arm_controller.lift(-100)
    # time.sleep(2)
    # jarvis.arm_controller.set_grip_degree(80)
    # time.sleep(1)
    # jarvis.arm_controller.lift(100)
    # time.sleep(1)
    jarvis.set_zero()