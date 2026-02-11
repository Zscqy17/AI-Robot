"""
SAM3 + GR-ConvNet 文字驱动抓取系统（独立项目）

流程：
1. 输入物体名称（文字）
2. SAM3 根据文字在画面中分割出物体 mask
3. 找到 mask 中心点
4. GR-ConvNet 在该点提供抓取角度和夹爪宽度
5. 机械臂执行抓取

运行环境：sam3 conda env (D:/miniconda3/envs/sam3/python.exe)
  - 需要: torch(CUDA), sam3, pyrealsense2, xarm-python-sdk, opencv, scipy, scikit-image

用法:
  D:/miniconda3/envs/sam3/python.exe main.py
  或双击 run.bat
"""
import os
import sys
import cv2
import time
import numpy as np
from queue import Queue
from PIL import Image as PILImage

# ===================== 路径设置 =====================
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

# SAM3 路径（按需修改）
SAM3_ROOT = r'C:\Users\admin\Desktop\ZhouProjects\SAM3_Test\sam3-main'
sys.path.insert(0, SAM3_ROOT)

# 将项目根目录加入路径，确保 inference 模块（GR-ConvNet 权重反序列化）可用
sys.path.insert(0, PROJ_ROOT)

from camera.rs_camera import RealSenseCamera
from camera.utils import get_combined_img
from grasp.grconvnet_torch import TorchGRConvNet
from grasp.robot_grasp import RobotGrasp

# ===================== 配置 =====================
WIN_NAME = 'SAM3 Grasp'
CAM_WIDTH = 640
CAM_HEIGHT = 480

# GR-ConvNet model
MODEL_FILE = os.path.join(PROJ_ROOT, 'models', 'grconvnet3_cornell_rgbd')

# 手眼标定
EULER_EEF_TO_COLOR_OPT = [0.067052239, -0.0311387575, 0.021611456, -0.004202176, -0.00848499, 1.5898775]
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]

# 机器人参数
GRASPING_RANGE = [-300, 200, -550, -50]
DETECT_XYZ = [-90, -300, 400]
RELEASE_XYZ = [-90, -800, 350]
LIFT_OFFSET_Z = 100
GRIPPER_Z_MM = 150
GRASPING_MIN_Z = 175

ROBOT_IP = "192.168.1.236"

# SAM3 配置
SAM3_CHECKPOINT = os.path.join(SAM3_ROOT, 'checkpoints', 'sam3.pt')
SAM3_CONFIDENCE = 0.3


# ===================== SAM3 加载 =====================
def load_sam3_model():
    """加载 SAM3 模型和处理器"""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print('[SAM3] 加载模型...')
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        bpe_path=None,
        checkpoint_path=SAM3_CHECKPOINT,
        load_from_HF=False,
        device='cuda',
    )
    processor = Sam3Processor(model, confidence_threshold=SAM3_CONFIDENCE)
    print('[SAM3] 模型加载完成')
    return processor


def sam3_segment(processor, color_image_bgr, text_prompt):
    """
    用 SAM3 对画面进行文字分割

    Args:
        processor: Sam3Processor
        color_image_bgr: [H, W, 3] BGR uint8
        text_prompt: 文字描述，如 "bottle", "cup"

    Returns:
        mask: [H, W] bool 或 None
        box: [x0, y0, x1, y1] 或 None
        score: float 或 None
    """
    import torch

    # BGR -> RGB -> PIL
    rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        state = processor.set_image(pil_img)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=text_prompt, state=state)

    masks = state.get('masks', None)
    boxes = state.get('boxes', None)
    scores = state.get('scores', None)

    if masks is None or len(masks) == 0:
        return None, None, None

    # 取置信度最高的结果
    if hasattr(scores, '__len__') and len(scores) > 1:
        best_idx = scores.argmax().item()
        mask = masks[best_idx][0].cpu().numpy()  # [H, W]
        box = boxes[best_idx].cpu().numpy()      # [x0, y0, x1, y1]
        score = scores[best_idx].item()
    else:
        mask = masks[0][0].cpu().numpy() if len(masks.shape) > 2 else masks[0].cpu().numpy()
        box = boxes[0].cpu().numpy() if boxes is not None else None
        score = scores[0].item() if scores is not None else 0.0

    return mask.astype(bool), box, score


def find_mask_grasp_point(mask):
    """
    找到 mask 的中心质点作为抓取位置 (fallback)

    Args:
        mask: [H, W] bool

    Returns:
        (row, col) 或 None
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    center_row = int(ys.mean())
    center_col = int(xs.mean())
    return center_row, center_col


def find_best_grasp_in_mask(mask, grconvnet, crop_y_inx, crop_x_inx, crop_size):
    """
    在 mask 内部区域找到 GR-ConvNet 置信度最高的抓取点。
    先腐蚀 mask 排除边缘，确保抓取点远离物体轮廓。
    如果 GR-ConvNet 没有推理结果，退回到质心。

    Returns:
        (row, col) in full image coordinates, confidence, or (None, 0)
    """
    if grconvnet.last_points_out is None:
        pt = find_mask_grasp_point(mask)
        return pt, 0.0 if pt is None else 1.0

    # Crop mask to same region as GR-ConvNet
    mask_crop = mask[crop_y_inx:crop_y_inx + crop_size,
                     crop_x_inx:crop_x_inx + crop_size]

    # Resize to GR-ConvNet output size (224x224)
    out_size = grconvnet.last_points_out.shape[0]
    mask_resized = cv2.resize(mask_crop.astype(np.uint8), (out_size, out_size),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

    # 腐蚀 mask，排除边缘点
    eroded_mask = None
    for ksize in [15, 7, 3]:
        kernel = np.ones((ksize, ksize), np.uint8)
        eroded = cv2.erode(mask_resized.astype(np.uint8), kernel, iterations=1).astype(bool)
        if eroded.any():
            eroded_mask = eroded
            break
    if eroded_mask is None:
        eroded_mask = mask_resized

    # Overlay confidence map with eroded mask
    confidence = grconvnet.last_points_out.copy()
    confidence[~eroded_mask] = -1

    best_conf = confidence.max()
    if best_conf < 0:
        pt = find_mask_grasp_point(mask)
        return pt, 0.0 if pt is None else 1.0

    best_idx = np.unravel_index(confidence.argmax(), confidence.shape)
    best_row_224, best_col_224 = best_idx

    crop_row = int(best_row_224 / out_size * crop_size)
    crop_col = int(best_col_224 / out_size * crop_size)
    full_row = crop_row + crop_y_inx
    full_col = crop_col + crop_x_inx

    return (full_row, full_col), float(best_conf)


def draw_sam3_overlay(display_img, mask, box, score, text_prompt, grasp_point=None):
    """在图像上叠加 SAM3 分割结果"""
    if mask is not None:
        overlay = display_img.copy()
        overlay[mask] = overlay[mask] * 0.5 + np.array([0, 200, 0], dtype=np.uint8) * 0.5
        display_img[:] = overlay

        if box is not None:
            x0, y0, x1, y1 = box.astype(int)
            cv2.rectangle(display_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            label = '{} {:.0f}%'.format(text_prompt, score * 100)
            cv2.putText(display_img, label, (x0, max(y0 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if grasp_point is not None:
            row, col = grasp_point
            cv2.circle(display_img, (col, row), 8, (0, 0, 255), -1)
            cv2.circle(display_img, (col, row), 12, (0, 0, 255), 2)


# ===================== UI 文字输入 =====================
def draw_input_bar(display_img, input_text, blink_on):
    """在画面底部绘制文字输入框"""
    h, w = display_img.shape[:2]
    bar_h = 45
    overlay = display_img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.8, display_img, 0.2, 0, display_img)
    cv2.rectangle(display_img, (8, h - bar_h + 5), (w - 8, h - 5), (100, 100, 100), 1)
    cursor = '|' if blink_on else ' '
    if input_text:
        label = 'Target: {}{}'.format(input_text, cursor)
        color = (255, 255, 255)
    else:
        label = 'Type object name, Enter to confirm{}'.format(cursor)
        color = (150, 150, 150)
    cv2.putText(display_img, label, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


# ===================== 主函数 =====================
def main():
    import torch

    print('=' * 60)
    print('  SAM3 + GR-ConvNet 文字驱动抓取系统')
    print('  1. 在画面底部输入物体英文名称')
    print('  2. 按 Enter 确认，SAM3 自动识别')
    print('  3. GR-ConvNet 计算最佳抓取姿态')
    print('  4. 机械臂自动执行抓取')
    print('  按 Esc 退出 | 按 R 重置（清除当前目标）')
    print('=' * 60)

    # --- 加载 SAM3 ---
    torch.autocast('cuda', dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()
    sam3_processor = load_sam3_model()

    # --- 加载相机 ---
    print('[Camera] 连接 RealSense D435...')
    camera = RealSenseCamera(width=CAM_WIDTH, height=CAM_HEIGHT)
    color_intrin, depth_intrin = camera.get_intrinsics()
    DEPTH_CAM_K = np.array([
        [color_intrin.fx, 0, color_intrin.ppx],
        [0, color_intrin.fy, color_intrin.ppy],
        [0, 0, 1]
    ])
    print('[Camera] 连接成功')

    # --- 加载 GR-ConvNet ---
    print('[GR-ConvNet] 加载模型...')
    grconvnet = TorchGRConvNet({
        'MODEL_FILE': MODEL_FILE,
        'DEPTH_CAM_K': DEPTH_CAM_K,
    })

    # --- 加载机器人 ---
    ggcnn_cmd_que = Queue(1)
    euler_opt = {
        'EULER_EEF_TO_COLOR_OPT': EULER_EEF_TO_COLOR_OPT,
        'EULER_COLOR_TO_DEPTH_OPT': EULER_COLOR_TO_DEPTH_OPT,
    }
    grasp_config = {
        'GRASPING_RANGE': GRASPING_RANGE,
        'DETECT_XYZ': DETECT_XYZ,
        'RELEASE_XYZ': RELEASE_XYZ,
        'LIFT_OFFSET_Z': LIFT_OFFSET_Z,
        'GRIPPER_Z_MM': GRIPPER_Z_MM,
        'GRASPING_MIN_Z': GRASPING_MIN_Z,
        'MIN_RESULT_Z_MM': 250,
    }
    grasp = RobotGrasp(ROBOT_IP, ggcnn_cmd_que, euler_opt, grasp_config, click_mode=True)
    print('[Robot] 机械臂就绪')

    # --- SAM3 已验证物体存在，降低 GR-ConvNet 置信度阈值 ---
    grconvnet.grasp_confidence_threshold = 0.15

    # --- 状态 ---
    current_prompt = None
    current_mask = None
    current_box = None
    current_score = None
    grasp_point = None
    grasp_confidence = 0.0
    is_grasping = False
    grasp_sent_time = 0
    sam3_frame_interval = 5
    frame_count = 0
    last_not_found_time = 0
    input_text = ''
    input_mode = True
    blink_timer = time.time()
    detection_time = 0
    PREVIEW_DELAY = 3.0
    grasp_fail_count = 0
    GRASP_FAIL_LOWER_THRESHOLD = 5   # 连续失败N次后降低阈值
    GRASP_FAIL_GIVE_UP = 20          # 连续失败N次后放弃该目标
    ORIGINAL_CONFIDENCE_THRESHOLD = grconvnet.grasp_confidence_threshold

    crop_y_inx = -1
    crop_x_inx = -1
    crop_size = 480

    cv2.namedWindow(WIN_NAME)

    while grasp.is_alive():
        color_image, depth_image = camera.get_images(align=True)

        if crop_y_inx < 0:
            imh, imw = depth_image.shape
            crop_size = min(imh, imw)
            crop_y_inx = max(0, imh - crop_size) // 2
            crop_x_inx = max(0, imw - crop_size) // 2

        color_crop = color_image[crop_y_inx:crop_y_inx + crop_size,
                                 crop_x_inx:crop_x_inx + crop_size, :]

        # --- SAM3 分割 ---
        if current_prompt and not is_grasping:
            frame_count += 1
            if current_mask is None or frame_count % sam3_frame_interval == 0:
                mask, box, score = sam3_segment(sam3_processor, color_image, current_prompt)
                if mask is not None:
                    current_mask = mask
                    current_box = box
                    current_score = score
                    gp, gconf = find_best_grasp_in_mask(mask, grconvnet,
                                                         crop_y_inx, crop_x_inx, crop_size)
                    if gp is not None:
                        grasp_point = gp
                        grasp_confidence = gconf
                        if detection_time == 0:
                            detection_time = time.time()
                        print('[SAM3] 识别到 "{}" (SAM3 {:.0f}%, 抓取质量 {:.3f})，抓取点: ({}, {})'.format(
                            current_prompt, score * 100, gconf, gp[1], gp[0]))
                else:
                    now = time.time()
                    if current_mask is None and now - last_not_found_time > 3.0:
                        last_not_found_time = now
                        print('[SAM3] 未找到 "{}"，请重试或换个描述'.format(current_prompt))

        # --- GR-ConvNet 推理（每帧）---
        robot_pos = grasp.get_eef_pose_m()
        grasp_img, _ = grconvnet.get_grasp_img(depth_image, color_image, DEPTH_CAM_K, robot_pos[2])

        # --- 自动抓取 ---
        if grasp_point is not None and not is_grasping and grasp.ready_grasp:
            if detection_time > 0 and time.time() - detection_time < PREVIEW_DELAY:
                pass
            else:
                row, col = grasp_point
                crop_row = row - crop_y_inx
                crop_col = col - crop_x_inx

                if 0 <= crop_row < crop_size and 0 <= crop_col < crop_size:
                    click_result = grconvnet.compute_grasp_at_pixel(DEPTH_CAM_K, crop_row, crop_col)
                    if click_result is not None:
                        is_grasping = True
                        detection_time = 0
                        grasp_sent_time = time.time()
                        grasp_fail_count = 0
                        grconvnet.grasp_confidence_threshold = ORIGINAL_CONFIDENCE_THRESHOLD
                        robot_pos = grasp.get_eef_pose_m()
                        if not ggcnn_cmd_que.empty():
                            ggcnn_cmd_que.get()
                        ggcnn_cmd_que.put([robot_pos, click_result])
                        print('[GRASP] 开始抓取 "{}"'.format(current_prompt))
                    else:
                        grasp_fail_count += 1
                        if grasp_fail_count >= GRASP_FAIL_GIVE_UP:
                            print('[GRASP] 连续 {} 次抓取失败，放弃目标 "{}"'.format(
                                grasp_fail_count, current_prompt))
                            current_prompt = None
                            current_mask = None
                            current_box = None
                            grasp_point = None
                            is_grasping = False
                            input_mode = True
                            input_text = ''
                            detection_time = 0
                            grasp_fail_count = 0
                            grconvnet.grasp_confidence_threshold = ORIGINAL_CONFIDENCE_THRESHOLD
                        elif grasp_fail_count == GRASP_FAIL_LOWER_THRESHOLD:
                            new_thresh = ORIGINAL_CONFIDENCE_THRESHOLD * 0.5
                            print('[GRASP] 连续 {} 次失败，降低置信度阈值: {:.3f} -> {:.3f}'.format(
                                grasp_fail_count, grconvnet.grasp_confidence_threshold, new_thresh))
                            grconvnet.grasp_confidence_threshold = new_thresh
                            grasp_point = None  # 让 SAM3 重新搜索
                        else:
                            print('[GRASP] 置信度或深度不足，搜索下一帧... (失败 {}/{})'.format(
                                grasp_fail_count, GRASP_FAIL_GIVE_UP))
                            grasp_point = None

        # --- 检查抓取完成 ---
        if is_grasping:
            if time.time() - grasp_sent_time > 2.0 and grasp.ready_grasp:
                is_grasping = False
                current_prompt = None
                current_mask = None
                current_box = None
                grasp_point = None
                detection_time = 0
                input_mode = True
                input_text = ''
                print('[GRASP] 抓取完成！请输入下一个目标')

        # --- 显示 ---
        display_img = color_crop.copy()

        if current_mask is not None and not is_grasping:
            mask_crop = current_mask[crop_y_inx:crop_y_inx + crop_size,
                                     crop_x_inx:crop_x_inx + crop_size]
            box_crop = None
            if current_box is not None:
                box_crop = current_box.copy()
                box_crop[0] -= crop_x_inx
                box_crop[2] -= crop_x_inx
                box_crop[1] -= crop_y_inx
                box_crop[3] -= crop_y_inx

            gp_crop = None
            if grasp_point is not None:
                gp_crop = (grasp_point[0] - crop_y_inx, grasp_point[1] - crop_x_inx)

            draw_sam3_overlay(display_img, mask_crop, box_crop, current_score or 0,
                              current_prompt or '', gp_crop)

        if is_grasping:
            cv2.putText(display_img, 'GRASPING: {}'.format(current_prompt), (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif current_prompt:
            cv2.putText(display_img, 'Target: {} [R=reset]'.format(current_prompt), (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if detection_time > 0 and not is_grasping:
                elapsed = time.time() - detection_time
                if elapsed < PREVIEW_DELAY:
                    remaining = PREVIEW_DELAY - elapsed
                    cv2.putText(display_img, 'Grasp in {:.1f}s'.format(remaining), (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        if input_mode:
            blink_on = int((time.time() - blink_timer) * 2) % 2 == 0
            draw_input_bar(display_img, input_text, blink_on)

        if grasp_img is None:
            grasp_img = depth_image[crop_y_inx:crop_y_inx + crop_size,
                                    crop_x_inx:crop_x_inx + crop_size]

        combined_img = get_combined_img(display_img, grasp_img)
        cv2.imshow(WIN_NAME, combined_img)

        key = cv2.waitKey(1)
        if key == -1:
            pass
        elif key == 27:  # Esc
            camera.stop()
            break
        elif input_mode:
            if key == 13:  # Enter
                if input_text.strip():
                    current_prompt = input_text.strip()
                    current_mask = None
                    current_box = None
                    grasp_point = None
                    input_mode = False
                    input_text = ''
                    print('[SAM3] 新目标: "{}"'.format(current_prompt))
            elif key == 8:  # Backspace
                input_text = input_text[:-1]
            elif 32 <= key < 127:
                input_text += chr(key)
        else:
            if key & 0xFF == ord('r'):
                current_prompt = None
                current_mask = None
                current_box = None
                grasp_point = None
                is_grasping = False
                input_mode = True
                input_text = ''
                detection_time = 0
                grasp_fail_count = 0
                grconvnet.grasp_confidence_threshold = ORIGINAL_CONFIDENCE_THRESHOLD
                print('[RESET] 已清除目标')
            elif 32 <= (key & 0xFF) < 127 and not is_grasping:
                current_prompt = None
                current_mask = None
                current_box = None
                grasp_point = None
                detection_time = 0
                input_mode = True
                input_text = chr(key & 0xFF)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
