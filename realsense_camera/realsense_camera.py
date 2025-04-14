from typing import Tuple

import pyrealsense2 as rs
import numpy as np

from test_utils import time_measurement


class RealsenseCamera:
    @time_measurement
    def __init__(self, color_mode=(640, 480, rs.format.rgb8, 30),
                 depth_mode=(640, 480, rs.format.z16, 30)):
        self.pipeline = rs.pipeline()
        config = rs.config()
        # refer to `realsense-viewer` or `rs-sensor-control` for more config mode
        config.enable_stream(rs.stream.depth, *depth_mode)
        config.enable_stream(rs.stream.color, *color_mode)

        # Start streaming
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.count = 0

        while self.count < 10:
            # discard first 10 frames after turning on
            _ = self.pipeline.wait_for_frames()
            self.count += 1

    @time_measurement
    def get_aligned_images(self) -> Tuple:
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        # Show images
        return color_image, depth_image, depth_intrin, depth_frame

    @staticmethod
    def get_coordinate_3d(x, y, dist, depth_intrin):
        return rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dist)

    @staticmethod
    def get_pixel_distance(x, y, depth_frame):
        return depth_frame.get_distance(x, y)

    def try_get_object_distance(self, x, y, depth_frame):
        pixel_range = 10
        distance = self.get_pixel_distance(x, y, depth_frame)
        if distance != 0:
            return distance

        for i in range(1, pixel_range + 1):
            for dx in range(i, -i - 1, -1):
                for dy in range(i, -i - 1, -1):
                    nx, ny = x + dx, y + dy
                    distance = self.get_pixel_distance(nx, ny, depth_frame)
                    if distance != 0.0:
                        return distance
        return distance

    def get_stable_depth_image(self, sample_count=8):
        """
        As depth image is unstable, some pixel in image can be randomly invalid but actually not in most cases.
        Sample a few continuous frames, and compute average depth of each pixel in the sequence if valid.
        It may be useful to 3D scene reconstruction.
        :param sample_count: number of continuous frames to sample
        :return: RGB image and depth stable image
        """
        color_image, depth_image, _, _ = self.get_aligned_images()
        stable_depth_image, valid_pixel_counter = [np.zeros_like(depth_image).astype(np.float32)] * 2

        for i in range(sample_count):
            _, depth_image, _, _ = self.get_aligned_images()
            # valid pixels
            mask = depth_image > 0
            stable_depth_image[mask] += depth_image
            valid_pixel_counter[mask] += 1

        valid_mask = valid_pixel_counter > 0
        stable_depth_image[valid_mask] /= valid_pixel_counter[valid_mask]
        return color_image, stable_depth_image
