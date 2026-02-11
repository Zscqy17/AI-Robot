"""
GR-ConvNet inference wrapper for click-to-grasp.
Takes RGBD (4-channel) input at 224x224.
Output format: (pos, cos, sin, width) - same as GGCNN.
"""
import os
import sys
import time
import cv2
import torch
import numpy as np
import scipy.ndimage as ndimage
from skimage.draw import disk

torch.nn.Module.dump_patches = False

GRCONVNET_INPUT_SIZE = 224


class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = False

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        if self.print_output:
            print('%s: %s' % (self.s, self.t1 - self.t0))


def process_depth_for_grconvnet(depth_crop):
    """Inpaint NaN values in depth crop"""
    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    depth_nan_mask = cv2.dilate(depth_nan_mask, kernel, iterations=1)
    depth_crop[depth_nan_mask == 1] = 0

    depth_scale = np.abs(depth_crop).max()
    if depth_scale == 0:
        depth_scale = 1.0
    depth_crop = depth_crop.astype(np.float32) / depth_scale
    depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)
    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale

    depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
    return depth_crop, depth_nan_mask


class TorchGRConvNet(object):
    """GR-ConvNet inference for click-to-grasp with RGBD input."""

    def __init__(self, config):
        self.device = torch.device('cpu')
        self.model = torch.load(config['MODEL_FILE'], map_location=self.device, weights_only=False)
        self.model.eval()
        self.input_size = GRCONVNET_INPUT_SIZE
        self.depth_cam_k = config['DEPTH_CAM_K']

        # Stored results for click-to-grasp
        self.last_ang_out = None
        self.last_width_out = None
        self.last_points_out = None
        self.last_depth_image = None
        self.last_crop_info = None
        self.grasp_confidence_threshold = 0.35  # 低于此值认为是桌面/空区域，不抓取
        self.grasp_img = None
        self.prev_mp = np.array([self.input_size // 2, self.input_size // 2])

        print('[GR-ConvNet] Model loaded, input_channels={}, input_size={}x{}'.format(
            self.model.conv1.in_channels, self.input_size, self.input_size))

    def get_grasp_img(self, depth_image, color_image, depth_cam_k, robot_z=0.4):
        """
        Run GR-ConvNet inference on RGBD input.
        
        Args:
            depth_image: [H, W] depth in meters, NaN for invalid
            color_image: [H, W, 3] BGR uint8
            depth_cam_k: 3x3 camera intrinsics
            robot_z: current robot z height in meters (unused in click mode)
            
        Returns:
            grasp_img: visualization image
            result: None (click mode doesn't auto-grasp)
        """
        imh, imw = depth_image.shape
        crop_size = min(imh, imw)
        crop_y_inx = max(0, imh - crop_size)
        crop_x_inx = max(0, imw - crop_size)

        # Crop depth and color
        y_start = crop_y_inx // 2
        x_start = crop_x_inx // 2
        depth_crop = depth_image[y_start:y_start + crop_size, x_start:x_start + crop_size].copy()
        color_crop = color_image[y_start:y_start + crop_size, x_start:x_start + crop_size, :].copy()

        # Inpaint depth
        depth_crop, depth_nan_mask = process_depth_for_grconvnet(depth_crop)

        # Resize to model input size (224x224)
        depth_resized = cv2.resize(depth_crop, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        color_resized = cv2.resize(color_crop, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        depth_nan_mask_resized = cv2.resize(depth_nan_mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        # Normalize depth: subtract mean, clip to [-1, 1]
        depth_norm = np.clip((depth_resized - depth_resized.mean()), -1, 1).astype(np.float32)

        # Normalize RGB: [0,255] -> [0,1], then subtract mean (matches training preprocessing)
        rgb_norm = color_resized.astype(np.float32) / 255.0
        rgb_norm -= rgb_norm.mean()
        # BGR -> RGB
        rgb_norm = rgb_norm[:, :, ::-1].copy()

        with TimeIt('GR-ConvNet Inference'):
            # Build RGBD tensor: [1, 4, 224, 224]
            # Channel order: D, R, G, B (matches training: depth first, then RGB)
            rgbd = np.stack([depth_norm, rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]], axis=0)
            rgbd_tensor = torch.from_numpy(rgbd.reshape(1, 4, self.input_size, self.input_size)).to(self.device)

            with torch.no_grad():
                pred_out = self.model(rgbd_tensor)

            points_out = pred_out[0].cpu().numpy().squeeze()
            points_out[depth_nan_mask_resized > 0] = 0

        with TimeIt('Post-process'):
            cos_out = pred_out[1].cpu().numpy().squeeze()
            sin_out = pred_out[2].cpu().numpy().squeeze()
            ang_out = np.arctan2(sin_out, cos_out) / 2.0
            width_out = pred_out[3].cpu().numpy().squeeze() * 150.0

            # Filter
            points_out = ndimage.gaussian_filter(points_out, 5.0)
            ang_out = ndimage.gaussian_filter(ang_out, 2.0)
            width_out = ndimage.gaussian_filter(width_out, 2.0)
            points_out = np.clip(points_out, 0.0, 1.0 - 1e-3)

        # Store for click-to-grasp
        self.last_ang_out = ang_out.copy()
        self.last_width_out = width_out.copy()
        self.last_points_out = points_out.copy()
        self.last_depth_image = depth_image
        self.last_crop_info = (crop_size, crop_y_inx, crop_x_inx, self.input_size)

        # Draw visualization
        grasp_img = cv2.applyColorMap((points_out * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        # Resize back to crop_size for display
        grasp_img = cv2.resize(grasp_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

        self.grasp_img = grasp_img
        return grasp_img, None  # Click mode: never auto-grasp

    def compute_grasp_at_pixel(self, depth_cam_k, display_row, display_col):
        """
        Compute grasp parameters at a clicked pixel location.
        
        Args:
            depth_cam_k: camera intrinsics matrix
            display_row: row in the displayed crop image (y)
            display_col: col in the displayed crop image (x)
            
        Returns:
            [x, y, z, ang, width, depth_center] in camera frame, or None
        """
        if self.last_ang_out is None or self.last_depth_image is None:
            return None

        crop_size, crop_y_inx, crop_x_inx, out_size = self.last_crop_info
        depth_image = self.last_depth_image
        imh, imw = depth_image.shape

        fx = depth_cam_k[0][0]
        fy = depth_cam_k[1][1]
        cx = depth_cam_k[0][2]
        cy = depth_cam_k[1][2]

        # Display coords -> GR-ConvNet output coords (224x224)
        grconv_row = int(display_row / crop_size * out_size)
        grconv_col = int(display_col / crop_size * out_size)
        grconv_row = max(0, min(grconv_row, out_size - 1))
        grconv_col = max(0, min(grconv_col, out_size - 1))

        # 检查点击位置周围区域的抓取置信度（取5x5邻域最大值）
        r_start = max(0, grconv_row - 2)
        r_end = min(out_size, grconv_row + 3)
        c_start = max(0, grconv_col - 2)
        c_end = min(out_size, grconv_col + 3)
        confidence = self.last_points_out[r_start:r_end, c_start:c_end].max()
        if confidence < self.grasp_confidence_threshold:
            print('[GR-ConvNet] 置信度太低 ({:.3f} < {:.3f})，该位置可能没有可抓取物体'.format(
                confidence, self.grasp_confidence_threshold))
            return None

        ang = self.last_ang_out[grconv_row, grconv_col]
        width = self.last_width_out[grconv_row, grconv_col]
        print('[GR-ConvNet] 置信度={:.3f}, 角度={:.1f}°, 宽度={:.1f}px'.format(
            confidence, np.degrees(ang), width))

        # GR-ConvNet output coords -> original image coords
        full_row = int(display_row + crop_y_inx // 2)
        full_col = int(display_col + crop_x_inx // 2)
        full_row = max(0, min(full_row, imh - 1))
        full_col = max(0, min(full_col, imw - 1))

        point_depth = depth_image[full_row, full_col]

        # Search nearby if NaN
        if np.isnan(point_depth):
            for r in range(1, 20):
                found = False
                for dr in range(-r, r + 1):
                    for dc in range(-r, r + 1):
                        nr, nc = full_row + dr, full_col + dc
                        if 0 <= nr < imh and 0 <= nc < imw and not np.isnan(depth_image[nr, nc]):
                            point_depth = depth_image[nr, nc]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        if np.isnan(point_depth):
            return None

        x = (full_col - cx) / fx * point_depth
        y = (full_row - cy) / fy * point_depth
        z = point_depth
        depth_center = z * 1000.0

        return [x, y, z, ang, width, depth_center]
