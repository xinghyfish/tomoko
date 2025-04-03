import math
import time
from typing import List

import PIL
from PIL import Image
from numpy.ma.core import arctan, arcsin, arccos

from algorithm import eye2arm_transform
from environment.camera import CameraPosition
from perception.vision import image_utils
from perception.vision.image_utils import show_masks_on_image, show_box_on_image
from perception.vision.vision_model import VisionModel
from realsense_camera.realsense_camera import RealsenseCamera
from robot_arm.arm_controller import ArmController


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
        """
        目标分割算法，采用容错机制，得到分割结果和对应图像的深度信息等。
        :param text_prompt: 文本提示词
        :return:
            - objects_position: 分割结果，包含掩码几何中心、深度相机镜头和中心的距离、掩码和分割框
            - color_image: RGB 模态图像
            - depth_image: 深度图像
            - depth_intrin: 深度相机内参
            - depth_frame: 深度帧
        """
        # 由于深度相机可能有噪点，中心坐标可能检测不到深度信息
        # RealSense camera 在目标不可检测/过近/过远时返回深度信息为0
        invalid_count, invalid_threshold, distance, x, y = 0, 4, 0.0, -1, -1

        color_image, depth_intrin, depth_image, depth_frame = [None] * 4
        objects_position = []
        # 多次检测排除噪点干扰
        while invalid_count < invalid_threshold and distance == 0.0:
            objects_position.clear()
            color_image, depth_image, depth_intrin, depth_frame = self.realsense_camera.get_aligned_images()
            results = self.vision_model.segment(PIL.Image.fromarray(color_image), text_prompt)
            masks, boxes = results['masks'], results['boxes']

            show_masks_on_image(PIL.Image.fromarray(color_image), masks)
            for i, (y, x) in enumerate(image_utils.center_of_mask(masks)):
                # 计算二维图像中目标中心和深度相机的距离，用于后续计算三维坐标
                distance = self.realsense_camera.get_pixel_distance(x, y, depth_frame)
                if distance:
                    objects_position.append((x, y, distance, masks[i], boxes[i]))
            if objects_position:
                break
            else:
                print("Invalid frame. Try again.")
                invalid_count += 1

        # 可以认为该位置深度信息不可知
        if invalid_threshold == invalid_count:
            # 如果中心坐标深度信息不可知，则对像素邻域进行搜索
            distance = self.realsense_camera.try_get_object_distance(x, y, depth_frame)
            if not distance:
                print("Unable to grasp target object.")
                return None

        return objects_position, color_image, depth_image, depth_intrin, depth_frame

    def detect(self, class_name: str):
        """
        调用 YOLOv11 进行目标检测，加入容错机制，得到目标检测结果和对应图像深度信息等。
        :param class_name: 类别标签名称
        :return:
            - objects_position: 检测结果，包含目标几何中心、深度相机镜头和中心的距离和检测框
            - 其他和 segment 函数相同
        """
        # 这里和分割采取的容错机制一致
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
        将二维平面的像素坐标转化为
        :param x: 物体表面中心的 x 坐标
        :param y: 物体表面中心的 y 坐标
        :param distance: (x, y) 和深度相机之间的距离
        :param depth_intrin: 深度相机内参
        :return: 几何中心在机械臂基坐标系下的坐标
        """
        px, py, pz, Rx, Ry, Rz = self.arm_controller.end_pose_state()
        T_c = RealsenseCamera.get_coordinate_3d(x, y, distance, depth_intrin) + [1]
        R_g = eye2arm_transform.euler2quaternion([Rx, Ry, Rz])
        T_g = eye2arm_transform.mm2m([px, py, pz])
        target_index = eye2arm_transform.get_target_index(R_g, T_g, T_c).tolist()
        target_index = eye2arm_transform.m2mm(target_index)
        return target_index

    def set_camera_view(self, target_index: List[float]) -> bool:
        """
        迭代法对齐相机 RGB 镜头目标中轴线。
        :param target_index: 目标位置
        :return: 如果在重复定位精度以内则返回 True 并进行进一步调整，否则为 False
        """
        joints = self.arm_controller.joint_state()
        D = CameraPosition.w

        xd, yd, _ = target_index
        x, y, _, _, _, _ = self.arm_controller.end_pose_state()

        A = math.sqrt(xd ** 2 + yd ** 2)        # 目标在xOy平面上和原点的距离
        theta = arctan(yd / xd) - arcsin(D / A)  # 对齐深度相机中轴和目标中心的机械臂Joint1角度

        joints[0] = math.degrees(theta)
        self.arm_controller.joint_control(joints)


if __name__ == '__main__':
    jarvis = Jarvis()
    time.sleep(2)
