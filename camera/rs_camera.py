import math
import time
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera(object):
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        print('[RealSense] Warming up camera...')
        time.sleep(1)
        for i in range(10):
            try:
                self.pipeline.wait_for_frames(5000)
            except RuntimeError:
                time.sleep(0.5)
        print('[RealSense] Camera ready.')

    def _restart_pipeline(self):
        print('[RealSense] Restarting pipeline...')
        try:
            self.pipeline.stop()
        except Exception:
            pass
        time.sleep(2)
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        time.sleep(1)
        for _ in range(5):
            try:
                self.pipeline.wait_for_frames(5000)
            except RuntimeError:
                time.sleep(0.5)
        print('[RealSense] Pipeline restarted.')

    def stop(self):
        self.pipeline.stop()

    def get_frames(self, align=False):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                frames = self.pipeline.wait_for_frames(10000)
                if align:
                    return self.align.process(frames)
                return frames
            except RuntimeError as e:
                print(f'[RealSense] Frame timeout (attempt {attempt+1}/{max_retries}): {e}')
                if attempt < max_retries - 1:
                    self._restart_pipeline()
                else:
                    raise

    def get_intrinsics(self, align=False):
        frames = self.get_frames(align=align)
        color_frame = frames.get_color_frame()
        aligned_depth_frame = frames.get_depth_frame()
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        return color_intrinsics, depth_intrinsics

    def get_images(self, align=False):
        frames = self.get_frames(align=align)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image * 0.001
        depth_image[depth_image == 0] = math.nan
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image


if __name__ == '__main__':
    import cv2
    cam = RealSenseCamera()
    color_intrin, depth_intrin = cam.get_intrinsics()
    print(color_intrin)
    print(depth_intrin)
    while True:
        color_image, depth_image = cam.get_images()
        cv2.imshow('COLOR', color_image)
        cv2.imshow('DEPTH', depth_image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cam.stop()
            break
