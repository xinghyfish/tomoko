import math
import time

from realsense_camera.realsense_camera import RealsenseCamera
from robot_arm.arm_controller import ArmController
from vlm import image_utils
from vlm.lang_sam_demo import show_masks_on_image
from vlm.vision_model import VisionModel
from robot_arm import eye2arm_transform


if __name__ == '__main__':
    time.sleep(10)
    realsense_camera = RealsenseCamera()
    vision_model = VisionModel()
    robot_arm = ArmController()
    robot_arm.end_pose_control([55, 0, 203, 0, 90, 0, 2])
    time.sleep(3)

    color_image, depth_image, depth_intrin, depth_frames = realsense_camera.get_aligned_images()
    text_prompt = "black cup"
    masks = vision_model.segment(color_image, text_prompt)
    show_masks_on_image(color_image, masks)
    y, x = image_utils.center_of_mask(masks)[0]
    Rx, Ry, Rz = 0, 90, 0
    R_g = eye2arm_transform.euler2quaternion([Rx, Ry, Rz])
    T_g = eye2arm_transform.mm2m([55, 0, 203])
    T_c = realsense_camera.get_coordinate_3d(x, y, depth_intrin, depth_frames) + [1]
    target_index = eye2arm_transform.get_target_index(R_g, T_g, T_c).tolist()
    # print(target_index)
    target_index = eye2arm_transform.m2mm(target_index)
    target_index[0] -= 100
    target_index[2] = 210
    robot_arm.end_pose_control(target_index + [0, 90, 0, 80])
    time.sleep(3)
    position = [55, 0, 203, 0, 90, 0, 20]
    robot_arm.end_pose_control(position)
    time.sleep(3)
    position = [0] * 6 + [20]
    robot_arm.joint_control(position)
    time.sleep(10)
    position[-1] = 80
    robot_arm.joint_control(position)
