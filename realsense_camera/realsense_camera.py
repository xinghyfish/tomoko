from typing import Dict, Tuple

import PIL
import pyrealsense2 as rs
import numpy as np
from vlm import image_utils
from PIL.Image import Image


class RealsenseCamera:
    def __init__(self, color_mode=(640, 480, rs.format.rgb8, 15),
                 depth_mode=(640, 480, rs.format.z16, 15)):
        self.pipeline = rs.pipeline()
        config = rs.config()
        # refer to `realsense-viewer` or `rs-sensor-control` for more config mode
        config.enable_stream(rs.stream.depth, *depth_mode)
        config.enable_stream(rs.stream.color, *color_mode)

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)

        # Start streaming
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.count = 0

        while self.count < 10:
            # discard first 10 frames after turning on
            _ = self.pipeline.wait_for_frames()
            self.count += 1

    def get_aligned_images(self) -> Tuple:
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = image_utils.brightness_augment(
            PIL.Image.fromarray(np.asanyarray(color_frame.get_data()))
        )
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        # Show images
        return color_image, depth_image, depth_intrin, np.asanyarray(color_frame.get_data()), depth_frame

    @staticmethod
    def get_coordinate_3d(x, y, dist, depth_intrin):
        return rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dist)

    @staticmethod
    def get_object_distance(x, y, depth_frame):
        return depth_frame.get_distance(x, y)
