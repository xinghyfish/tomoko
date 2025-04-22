import logging
import math
import time

import numpy as np
from numpy.ma.core import arctan

import environment
from algorithm import end_pose_transform, drop_water
from environment import GraspInfo
from environment.entity import *
from jarvis import Jarvis
from perception.vision.image_utils import center_of_mask
from perception.vision.lang_sam_demo import show_masks_on_image
from robot_arm.arm_controller import gripper_length
from robot_arm.command import EndPoseMoveCommand, BottomTurnCommand, WristRollCommand, GraspCommand, \
    JointControlCommand, LiftMoveCommand, TimerCommand, GripperCommand, HeadUpCommand
from robot_arm.pipeline import Pipeline
from test_utils import time_measurement

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

    @time_measurement
    def detect_teapot(self):
        # detect the whole teapot
        prompt = "yellow circle in teapot"
        target_index = []

        for i in range(3):
            obj_lst, color_image, depth_image, depth_intrin, depth_frame = self.jarvis.segment(prompt)
            assert obj_lst, len(obj_lst) == 1
            xe, ye, dis, teapot_mask, _ = obj_lst[0]

            # segment the part of handle from the whole
            teapot_depth = depth_image * teapot_mask
            replaced_teapot_depth = np.where(teapot_depth == 0, 10000, teapot_depth)
            # find the index of the nearest part
            min_index = np.unravel_index(np.argmin(replaced_teapot_depth), replaced_teapot_depth.shape)
            min_depth = teapot_depth[min_index]
            thickness = 5
            # so this is the mask of handle
            wooden_mask = np.where(abs(teapot_depth - min_depth) <= thickness, 1, 0)[np.newaxis, :]
            show_masks_on_image(color_image, wooden_mask)
            # time to check it out

            y, x = center_of_mask(wooden_mask)[0]
            print("handle x = ", x)
            distance = self.jarvis.realsense_camera.try_get_object_distance(x, y, depth_frame)
            target_index = self.jarvis.pixel_to_3d(x, y, distance, depth_intrin)
            self.jarvis.set_camera_view(target_index)
            time.sleep(0.2)

        end_pose = end_pose_transform.middle_transform(target_index, Teapot().radius, Teapot().polar_angle)
        prompt = "teapot"
        self.grasp_info[prompt] = GraspInfo(target_index, end_pose)
        logging.info(f"{prompt} position: {target_index}")

        # 茶壶盖目标识别
        x_handle, y_handle, _ = target_index
        theta = arctan(y_handle / x_handle)
        delta = (Teapot().radius + Teapot().inner_radius + Teapot().cover_radius * 2) * math.sin(math.radians(Teapot().polar_angle))
        x_center = x_handle + math.cos(theta) * delta - Teapot().cover_radius
        y_center = y_handle + math.sin(theta) * delta - Teapot().cover_radius
        z_center = environment.front_table_height + Teapot().height - environment.under_board

        end_pose_cover = end_pose_transform.bottom_transform([x_center, y_center, z_center], Teapot().cover_radius * 2, 135)
        self.grasp_info["teapot cover"] = GraspInfo([x_center, y_center, z_center], end_pose_cover)

        return target_index

    @time_measurement
    def detect_can(self, color):
        prompt = f"{color} can"
        entity = TeaCan()
        target_index = [None] * 3

        for i in range(3):
            objects_position, _, _, depth_intrin, _ = self.jarvis.segment(prompt)
            assert objects_position, len(objects_position) == 1
            x, y, distance, mask, _ = objects_position[0]
            print("can xxx = ", x)
            target_index = self.jarvis.pixel_to_3d(x, y, distance, depth_intrin)
            self.jarvis.set_camera_view(target_index)
            time.sleep(0.5)

        end_pose = end_pose_transform.middle_transform(target_index, entity.radius, entity.polar_angle)
        self.grasp_info[prompt] = GraspInfo(target_index, end_pose)
        logging.info(f"{prompt} position: {target_index}")

        cover_height = environment.side_table_height + TeaCan().height - environment.under_board
        cover_pos = target_index[:2] + [cover_height]
        end_pose_cover = end_pose_transform.bottom_transform(cover_pos, TeaCan().radius, TeaCan().polar_angle)
        self.grasp_info[prompt + " cover"] = GraspInfo(target_index, end_pose_cover)

        return target_index

    @time_measurement
    def detect_water_dispenser(self):
        prompt = "small black rectangle on the bottom"
        target_index = [None] * 3
        entity = Faucet()

        for key in ['hot', 'cold']:
            for i in range(3):
                objects_position, _, _, depth_intrin, _ = self.jarvis.segment(prompt)
                boxes = [_[-1] for _ in objects_position]
                left_idx = 0 if boxes[0][0] < boxes[1][0] else 1
                right_idx = 1 - left_idx
                faucets = {'hot': left_idx, 'cold': right_idx}
                x, y, distance, mask, _ = objects_position[faucets[key]]
                target_index = self.jarvis.pixel_to_3d(x, y, distance, depth_intrin)
                self.jarvis.set_camera_view(target_index)
                time.sleep(0.5)
            # 如果直接用机械臂接触龙头表面的 end pose
            end_pose_temp = end_pose_transform.middle_transform(target_index, entity.radius, entity.polar_angle)
            X_faucet, Y_faucet, Z_faucet, RX_faucet, RY_faucet, RZ_faucet = end_pose_temp
            # 需要考虑茶壶前端点体积和夹爪和壶身的距离，实验发现夹爪和
            delta_teapot = -(Teapot().inner_radius * 2) - (Teapot().radius - gripper_length)
            # 综上用茶壶取水需要调整的距离，为负数
            distance_push_faucet = delta_teapot + Faucet().push_distance
            # 龙头偏移角度
            theta_faucet = math.atan(Y_faucet / X_faucet)
            # 计算机械臂夹爪夹取茶壶并能从龙头取水时的末端坐标
            X_push_faucet, Y_push_faucet, Z_push_faucet = (
                X_faucet + distance_push_faucet * math.cos(theta_faucet),
                Y_faucet + distance_push_faucet * math.sin(theta_faucet),
                Z_faucet - 10
            )
            end_pose_push_faucet = [X_push_faucet, Y_push_faucet, Z_push_faucet, RX_faucet, RY_faucet, RZ_faucet]
            self.grasp_info[key] = GraspInfo(target_index, end_pose_push_faucet)

    @time_measurement
    def detect_cup(self):
        class_name = "cup"
        target_index = [0] * 3

        # 眼在手上纯视觉方法迭代式求解对称物体质心
        for i in range(3):
            objects_position, color_image, depth_image, depth_intrin, depth_frame = self.jarvis.detect(class_name)
            assert len(objects_position) >= 1
            x, y, distance, box = objects_position[0]
            y_bottom = box[-1]
            # 检测茶杯
            print("cup xxx = ", x)
            target_index = self.jarvis.pixel_to_3d(x, (y + y_bottom) >> 1, distance, depth_intrin)
            self.jarvis.set_camera_view(target_index)
            time.sleep(0.5)

        end_pose_cup = end_pose_transform.middle_transform(target_index, TeaCup().radius, TeaCup().polar_angle)
        self.grasp_info[class_name] = GraspInfo(target_index, end_pose_cup)
        # detect end pose to drop water
        x, y, z = target_index
        assert type(x) == float and type(y) == float and type(z) == float
        # 计算茶壶倾斜后壶嘴的坐标
        delta_height = 10
        xd = x + TeaCup().radius
        # 这里考虑水流的动态轨迹，取水流初始抛物线落点和自由落体点的中点
        yd = y - TeaCup().radius / 3
        zd = environment.front_table_height + TeaCup().height + delta_height - environment.under_board
        gamma = -45
        # 茶壶倒水算法：通过倒水角度、茶杯位置得到茶壶倒水前位置的解
        dest_target_index = drop_water.convert_teacup_to_teapot(xd, yd, zd, math.radians(gamma))
        drop_water_end_pose = end_pose_transform.middle_transform(dest_target_index, Teapot().radius, Teapot().polar_angle)
        self.grasp_info['drop water'] = GraspInfo(dest_target_index, drop_water_end_pose)
        return

    def scan_desk(self):
        """
        预操作：环境检查，确保茶具都在位置上，同时记录下茶具的位置和抓取信息
        """
        # TODO：补充所有茶具的 detect 方法
        teapot_detectable_joints = [0, 0, -10, 0, 30, -5]
        self.jarvis.arm_controller.joint_control(teapot_detectable_joints)
        time.sleep(2)
        self.detect_teapot()
        tea_can_detectable_joints = [40, 0, -10, 0, 30, -5]
        self.jarvis.arm_controller.joint_control(tea_can_detectable_joints)
        time.sleep(2)
        # suppose color
        # colors = ["silver", "red", "yellow"]
        colors = ["red"]
        for color in colors:
            self.detect_can(color)
            print(color)
            print(self.jarvis.arm_controller.joint_state())
        self.jarvis.arm_controller.joint_control(teapot_detectable_joints)

    def add_tea_to_teapot(self, color):
        """
        子任务1：抓取茶罐，将茶包丢进茶壶中
        :return:
        """
        self.jarvis.arm_controller.joint_state()
        grasp_target = f"{color} can"
        self.current_grasp_object = grasp_target
        end_pose_can = self.grasp_info[grasp_target].end_pose

        label = "teapot"
        wrist_turn_angle = 180
        add_tea_pipeline = Pipeline("drop tea")
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

        # 计算茶壶中心
        theta = math.radians(teapot.polar_angle)
        # 计算茶壶柄方向的单位向量
        vec = np.array([
            x_teapot_handle * math.sin(theta) / math.sqrt(x_teapot_handle ** 2 + y_teapot_handle ** 2),
            y_teapot_handle * math.sin(theta) / math.sqrt(x_teapot_handle ** 2 + y_teapot_handle ** 2),
            math.cos(theta)
        ])
        # 计算茶壶中心左边（主要需要xOy平面坐标）
        center_of_inner_space = np.array([x_teapot_handle, y_teapot_handle, z_teapot_handle]) + \
                                        (teapot.inner_radius + teapot.radius / math.sin(theta)) * vec
        x_teapot, y_teapot, z_teapot = center_of_inner_space.tolist()

        # 计算茶壶口中心坐标
        x_can, y_can, z_can = can_grasp_info.position
        # 计算茶罐和茶壶之间的角度差以计算Joint1转动角度
        angle_can = arctan(y_can / x_can)
        angle_teapot = arctan(y_teapot / x_teapot)
        # 为了防止碰撞，这里计算茶壶中心和茶罐中心距离以换算xOy平面上的目标坐标
        # 确保将物体先移动到较远的位置
        xOy_distance_teapot = math.sqrt(x_teapot ** 2 + y_teapot ** 2)
        x_can, y_can, z_can = can_grasp_info.position
        xOy_distance_can = math.sqrt(x_can ** 2 + y_can ** 2)
        delta_distance = xOy_distance_teapot - xOy_distance_can - 10
        x_target, y_target = x_can + delta_distance * math.cos(angle_can), y_can +delta_distance * math.sin(angle_can)
        # 茶包倾倒时的高度，由于深度相机安装板影响，需要抬高12cm
        drop_height = 120
        # 茶壶柄的顶端几乎和茶壶无茶盖状态下茶壶口高度一致
        end_pose = end_pose_transform.middle_transform((x_target, y_target, z_teapot_handle + drop_height), can.radius, 90)

        # 先取下茶壶盖
        self.remove_teapot_cover()

        # 茶罐抓取和放置
        # 无需再复位，只要放到回收区即可
        add_tea_pipeline.add_command(JointControlCommand(self, [55, 10, -20, 0, 15, -5], 60))
        add_tea_pipeline.add_command(GraspCommand(self, "red can cover", self.grasp_info["red can cover"].end_pose))
        add_tea_pipeline.add_command(LiftMoveCommand(self, 10))
        add_tea_pipeline.add_command(BottomTurnCommand(self, 10))
        add_tea_pipeline.add_command(GripperCommand(self, 70))
        add_tea_pipeline.run()
        add_tea_pipeline.clear()

        joints = self.jarvis.arm_controller.joint_state()
        joints[1] -= 10
        add_tea_pipeline.add_command(JointControlCommand(self, joints, 50))
        add_tea_pipeline.add_command(JointControlCommand(self, [55, 10, -20, 0, 15, -5], 60))
        add_tea_pipeline.run()
        add_tea_pipeline.clear()

        add_tea_pipeline.add_command(GraspCommand(self, grasp_target, end_pose_can))
        add_tea_pipeline.add_command(LiftMoveCommand(self, 10))
        add_tea_pipeline.add_command(EndPoseMoveCommand(self, end_pose, "linear", 20))
        add_tea_pipeline.add_command(BottomTurnCommand(self, math.degrees(angle_teapot - angle_can)))
        add_tea_pipeline.add_command(WristRollCommand(self, wrist_turn_angle))
        add_tea_pipeline.add_command(TimerCommand(self, 0.5))

        return add_tea_pipeline.run() and add_tea_pipeline.undo()

    def add_hot_water(self):
        """
        子任务2：从饮水机取热水
        """
        # 0. 验证先验条件
        assert self.grasp_info.get('teapot') is not None    # detect_teapot is done
        assert self.grasp_info.get('hot') is not None       # detect_faucet is done

        # 1. 抓茶壶
        add_water_pipeline = Pipeline('add water')
        joints = [-15, 10, -20, 0, 30, -5]
        add_water_pipeline.add_command(JointControlCommand(self, joints, 20))
        teapot_grasp_end_pose = self.grasp_info['teapot'].end_pose
        add_water_pipeline.add_command(GraspCommand(self, 'teapot', teapot_grasp_end_pose))

        # 2. 中间状态计算，规划轨迹
        end_pose_faucet = self.grasp_info['hot'].end_pose
        X_faucet, Y_faucet, Z_faucet, RX_faucet, RY_faucet, RZ_faucet = end_pose_faucet
        X_teapot, Y_teapot, Z_teapot, RX_teapot, RY_teapot, RZ_teapot = teapot_grasp_end_pose
        add_water_pipeline.add_command(LiftMoveCommand(self, 10))

        theta_back = math.atan(Y_teapot / X_teapot)
        theta_faucet = math.atan(Y_faucet / X_faucet)

        delta_theta = math.degrees(theta_faucet - theta_back)
        add_water_pipeline.add_command(BottomTurnCommand(self, delta_theta))
        add_water_pipeline.add_command(EndPoseMoveCommand(self, end_pose_faucet, "linear", 20))
        wait_add_water_elapse_second = 2   # waiting...
        add_water_pipeline.add_command(TimerCommand(self, wait_add_water_elapse_second))

        add_water_pipeline.run()
        add_water_pipeline.undo()
        self.cover_recover()

    def drop_water(self):
        """
        子任务3：倒茶
        :return:
        """
        self.remove_teapot_cover()
        gamma = -45
        drop_water_pipeline = Pipeline("drop water")
        joints = [0, 10, -20, 0, 40, -5]
        drop_water_pipeline.add_command(JointControlCommand(self, joints, 20))
        teapot_grasp_end_pose = self.grasp_info['teapot'].end_pose
        drop_water_pipeline.add_command(GraspCommand(self, 'teapot', teapot_grasp_end_pose))
        drop_water_pipeline.add_command(LiftMoveCommand(self, 20))
        drop_water_pipeline.add_command(EndPoseMoveCommand(self, self.grasp_info["drop water"].end_pose, "linear", 50))
        drop_water_pipeline.add_command(WristRollCommand(self, gamma))
        drop_water_pipeline.add_command(TimerCommand(self, 2))

        return drop_water_pipeline.run() and drop_water_pipeline.undo()

    def handle_cup(self):
        """
        子任务4：递茶杯
        :return:
        """
        init_joints = self.jarvis.arm_controller.joint_state()
        grasp_end_pose = self.grasp_info["cup"].end_pose
        lift_height = 10
        x, y, z, RX, RY, RZ = grasp_end_pose
        z += lift_height
        x += 40
        hand_end_pose = x, y, z, RX, RY, RZ

        hand_pipeline = Pipeline('hand cup')
        hand_pipeline.add_command(GraspCommand(self, 'cup', grasp_end_pose))
        hand_pipeline.add_command(EndPoseMoveCommand(self, hand_end_pose, "linear", 20))
        hand_pipeline.add_command(LiftMoveCommand(self, -lift_height))
        hand_pipeline.add_command(GripperCommand(self, 70))
        hand_pipeline.add_command(LiftMoveCommand(self, lift_height))

        flag = hand_pipeline.run()
        joints = self.jarvis.arm_controller.joint_state()
        joints[1] -= 15
        self.jarvis.arm_controller.joint_control(joints)
        time.sleep(0.5)
        self.jarvis.arm_controller.joint_control(init_joints)

    def recycle_cup(self):
        """
        子任务5：回收茶杯
        :return:
        """
        # TODO：检测最远处的茶杯
        # TODO：抓取茶杯，轨迹规划，放置到回收区
        pass

    def wash_teapot(self):
        """
        子任务6：清洗茶壶
        """
        wash_teapot_pipeline = Pipeline('wash teapot')
        # TODO：抓取茶壶，倒出茶水、茶包
        wash_teapot_pipeline.add_command(GraspCommand(self, 'teapot', self.grasp_info['teapot'].end_pose))
        wash_teapot_pipeline.add_command(LiftMoveCommand(self, 20))
        wash_teapot_pipeline.run()
        wash_teapot_pipeline.clear()
        init_angle = self.jarvis.arm_controller.joint_state()[0]
        wash_teapot_pipeline.add_command(BottomTurnCommand(self, -80 - init_angle))
        wash_teapot_pipeline.add_command(WristRollCommand(self, -90))
        wash_teapot_pipeline.add_command(TimerCommand(self, 3))
        wash_teapot_pipeline.add_command(WristRollCommand(self, 90))
        wash_teapot_pipeline.run()
        wash_teapot_pipeline.clear()

        # TODO：取冷水，清洗茶杯（MOVE C摇晃尝试一下）
        end_pose_faucet = self.grasp_info['cold'].end_pose
        X_faucet, Y_faucet, Z_faucet, RX_faucet, RY_faucet, RZ_faucet = end_pose_faucet
        angle_faucet = math.degrees(math.atan(Y_faucet / X_faucet))
        joints = self.jarvis.arm_controller.joint_state()
        current_angle = joints[0]
        wait_add_water_elapse_second = 2  # waiting...
        wash_teapot_pipeline.add_command(BottomTurnCommand(self, angle_faucet - current_angle))
        wash_teapot_pipeline.add_command(EndPoseMoveCommand(self, end_pose_faucet, "linear", 20))
        wash_teapot_pipeline.add_command(TimerCommand(self, wait_add_water_elapse_second))
        wash_teapot_pipeline.run()
        wash_teapot_pipeline.undo()
        wash_teapot_pipeline.clear()

        # TODO：倒出冷水
        cold_water_wait_time = 3
        wash_teapot_pipeline.add_command(BottomTurnCommand(self, 5))
        wash_teapot_pipeline.add_command(WristRollCommand(self, 160))
        wash_teapot_pipeline.add_command(TimerCommand(self, cold_water_wait_time))
        wash_teapot_pipeline.add_command(WristRollCommand(self, -160))
        wash_teapot_pipeline.run()
        wash_teapot_pipeline.clear()
        # TODO：茶壶归位，将茶壶盖盖上
        wash_teapot_pipeline.add_command(BottomTurnCommand(self, 90 + init_angle))
        wash_teapot_pipeline.add_command(LiftMoveCommand(self, -20))
        wash_teapot_pipeline.add_command(GripperCommand(self, 70))
        wash_teapot_pipeline.add_command(HeadUpCommand(self, 20))
        wash_teapot_pipeline.add_command(JointControlCommand(self, [0] * 6, 50))
        wash_teapot_pipeline.run()

        # self.cover_recover()
        self.jarvis.arm_controller.joint_control([0] * 6)

    def remove_teapot_cover(self):
        """
        取下茶壶盖
        """
        # 茶壶盖抓取和放置
        remove_cover_pipeline = Pipeline("remove teapot cover")
        remove_cover_pipeline.add_command(GraspCommand(self, "teapot cover", self.grasp_info["teapot cover"].end_pose))
        remove_cover_pipeline.add_command(LiftMoveCommand(self, 25))
        remove_cover_pipeline.add_command(BottomTurnCommand(self, 20))
        remove_cover_pipeline.add_command(LiftMoveCommand(self, -72))
        remove_cover_pipeline.add_command(GripperCommand(self, 70))
        remove_cover_pipeline.run()
        time.sleep(0.1)
        # 需要存储茶壶盖放置位置以供后续抓取
        self.grasp_info['teapot cover on desk'] = GraspInfo(None, self.jarvis.arm_controller.end_pose_state())
        remove_cover_pipeline.clear()
        remove_cover_pipeline.add_command(LiftMoveCommand(self, 70))
        remove_cover_pipeline.add_command(HeadUpCommand(self))
        remove_cover_pipeline.add_command(JointControlCommand(self, [0, 10, -20, 0, 15, -5], 50))
        remove_cover_pipeline.run()
        remove_cover_pipeline.clear()

    def cover_recover(self):
        """
        将茶壶盖放回茶壶上方
        """
        self.jarvis.arm_controller.joint_control([-25, 10, -10, 0, 30, -5])
        time.sleep(0.1)
        self.detect_teapot()
        recover_position = self.grasp_info['teapot cover'].end_pose
        lift_height = 25
        recover_position[2] += lift_height

        recover_pipeline = Pipeline("recover cover")
        recover_pipeline.add_command(GraspCommand(self, "teapot cover on desk", self.grasp_info['teapot cover on desk'].end_pose))
        recover_pipeline.add_command(LiftMoveCommand(self, lift_height + Teapot().cover_height))
        recover_pipeline.add_command(EndPoseMoveCommand(self, recover_position, "p", 50))
        recover_pipeline.add_command(LiftMoveCommand(self, -lift_height + 1))
        recover_pipeline.add_command(GripperCommand(self, 70))
        recover_pipeline.add_command(LiftMoveCommand(self, lift_height))
        recover_pipeline.add_command(HeadUpCommand(self))
        recover_pipeline.run()
        recover_pipeline.clear()




def pipeline():
    tomoko = Tomoko()
    # pos = [-40, 10, -10, 0, 10, -5]
    # tomoko.jarvis.arm_controller.joint_control(pos)
    # time.sleep(2)
    # tomoko.detect_water_dispenser()
    # time.sleep(1)

    # pos = [0, 10, -10, 0, 30, -5]
    # tomoko.jarvis.arm_controller.joint_control(pos)

    tomoko.scan_desk()
    for info in tomoko.grasp_info.items():
        print(info)

    # tomoko.jarvis.arm_controller.joint_control([0] * 6)
    # tomoko.add_water('left')
    tomoko.add_tea_to_teapot("red")

    # tomoko.drop_water()

def test_detect_cup_and_drop_water():
    tomoko = Tomoko()
    tomoko.jarvis.arm_controller.joint_control([10, 20, -20, 0, 40, -5])
    time.sleep(0.5)
    tomoko.detect_cup()
    tomoko.jarvis.arm_controller.joint_control([-15, 10, -10, 0, 30, -5])
    time.sleep(0.5)
    tomoko.detect_teapot()
    # print(tomoko.grasp_info)
    tomoko.drop_water()


def test_detect_can_and_cover():
    tomoko = Tomoko()
    tomoko.jarvis.arm_controller.joint_control([55, 10, -20, 0, 25, -5])
    time.sleep(0.5)
    tomoko.detect_can("red")
    tomoko.jarvis.arm_controller.joint_control([-20, 10, -10, 0, 30, -5])
    time.sleep(0.5)
    tomoko.detect_teapot()
    tomoko.add_tea_to_teapot("red")


def test_detect_teapot_and_cover():
    tomoko = Tomoko()
    tomoko.jarvis.arm_controller.joint_control([-10, 10, -10, 0, 30, -5])
    time.sleep(0.5)
    tomoko.detect_teapot()
    print(tomoko.grasp_info)
    tomoko.jarvis.arm_controller.set_grip_degree(70, gripper_effort=300)
    time.sleep(0.5)
    tomoko.jarvis.arm_controller.end_pose_control(tomoko.grasp_info["teapot cover"].end_pose)
    time.sleep(0.5)
    tomoko.jarvis.arm_controller.set_grip_degree(0)
    time.sleep(0.5)
    tomoko.jarvis.arm_controller.lift(30)


def test_detect_teapot_and_get_water():
    tomoko = Tomoko()
    tomoko.jarvis.arm_controller.joint_control([-20, 10, -10, 0, 30, 0])
    time.sleep(0.5)
    tomoko.detect_teapot()
    print(tomoko.grasp_info['teapot cover'])
    pos = [-50, 10, -10, 0, 10, -5]
    tomoko.jarvis.arm_controller.joint_control(pos)
    time.sleep(0.5)
    tomoko.detect_water_dispenser()
    tomoko.jarvis.arm_controller.joint_control([-10, 10, -10, 0, 30, 5])
    time.sleep(0.5)
    tomoko.remove_teapot_cover()
    tomoko.add_hot_water()
    tomoko.cover_recover()


def detect_tea_can_and_add_tea():
    tomoko = Tomoko()
    tomoko.jarvis.arm_controller.joint_control([-10, 10, -10, 0, 30, 5])
    time.sleep(0.5)
    tomoko.detect_teapot()
    tomoko.jarvis.arm_controller.joint_control([55, 10, -20, 0, 15, 5])
    time.sleep(0.5)
    tomoko.detect_can("red")
    tomoko.add_tea_to_teapot("red")
    time.sleep(0.5)

def test_wash_teapot():
    tomoko = Tomoko()
    tomoko.jarvis.arm_controller.joint_control([-15, 10, -10, 0, 30, 0])
    time.sleep(0.5)
    tomoko.detect_teapot()
    tomoko.jarvis.arm_controller.joint_control([-50, 10, -10, 0, 20, 0])
    time.sleep(0.5)
    tomoko.detect_water_dispenser()
    # tomoko.remove_teapot_cover()
    tomoko.wash_teapot()

def execute():
    # environment scanning
    tomoko = Tomoko()
    tomoko.jarvis.arm_controller.joint_control([-50, 10, -10, 0, 20, 0])
    time.sleep(0.5)
    tomoko.detect_water_dispenser()
    tomoko.jarvis.arm_controller.joint_control([-25, 10, -10, 0, 30, 0])
    time.sleep(0.5)
    tomoko.detect_teapot()
    tomoko.jarvis.arm_controller.joint_control([10, 20, -20, 0, 40, -5])
    time.sleep(0.5)
    tomoko.detect_cup()
    tomoko.jarvis.arm_controller.joint_control([55, 10, -20, 0, 25, -5])
    time.sleep(0.5)
    tomoko.detect_can("red")

    # execute task
    tomoko.add_tea_to_teapot("red")
    tomoko.add_hot_water()
    time.sleep(0.5)
    tomoko.jarvis.arm_controller.joint_control([0] * 6)
    time.sleep(5)
    tomoko.jarvis.arm_controller.joint_control([-25, 10, -10, 0, 30, 0])
    time.sleep(0.5)
    tomoko.detect_teapot()
    tomoko.drop_water()
    tomoko.jarvis.arm_controller.joint_control([-25, 10, -10, 0, 30, 0])
    time.sleep(0.5)
    tomoko.detect_teapot()
    tomoko.wash_teapot()



if __name__ == '__main__':
    # test_detect_cup_and_drop_water()
    # test_detect_teapot_and_get_water()
    # test_detect_can_and_cover()
    # detect_tea_can_and_add_tea()
    # test_wash_teapot()
    # time.sleep(15)
    A = time.time()
    # execute()
    execute()
    print("time xxx = ", B - A)