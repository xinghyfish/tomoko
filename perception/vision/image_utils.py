from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt


def center_of_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Get the center of each object.
    :param mask: np.ndarray
        A 3D NumPy array of shape (n, h, w) where each mask[i] is a binary
        mask of shape (h, w).
    :return:
        center of each object(mask)
    """
    centers = []
    num_tasks = mask.shape[0]

    for i in range(num_tasks):
        center_mask = mask[i, :, :]
        y_indices, x_indices = np.where(center_mask == 1)
        if len(y_indices) > 0:
            center_y = int(np.round(np.mean(y_indices)))
            center_x = int(np.round(np.mean(x_indices)))
            centers.append((center_y, center_x))
        else:
            centers.append((-1, -1))

    return centers


def remove_outliers_iqr(data: List) -> float:
    """
    Remove abnormal data from the sequence and get the average of the rest.
    :param data:
        data sequence (sampled from the specific pixel from the depth image)
    :return:
    """
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Calculate the bounds for normal data
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify normal data
    normal_data = data[(data >= lower_bound) & (data <= upper_bound)]

    # Return the average of the normal data
    return float(np.mean(normal_data))


def brightness_augment(image: Image) -> Image:
    # 创建一个亮度增强对象
    enhancer = ImageEnhance.Brightness(image)
    # 设置亮度因子，值大于1.0时图像变亮，值小于1.0时图像变暗
    brightened_image = enhancer.enhance(1.2)  # 1.5倍亮度
    return brightened_image


def mask_diff(full_mask: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return full_mask - mask


def show_masks_on_image(image_pil, masks):
    # 假设 results 是从模型返回的预测结果
    # 通常包含一个 mask 列表，形状为 (H, W) 的布尔值（True 表示属于分割区域）
    # 提取第一个结果的 mask
    n, h, w = masks.shape
    # 将 mask 转为 NumPy 数组
    masks_np = np.array(masks).astype(np.uint8)

    # 将 mask 叠加到原图上
    alpha = 0.5  # 透明度
    image_np = np.array(image_pil)
    color_mask = np.zeros_like(image_np, dtype=np.uint8)
    mask = np.zeros_like(image_np[:,:,0], dtype=np.uint8)
    for mask_np in masks_np:
        mask |= mask_np
    color_mask[:, :, 1] += mask * 255  # 绿色通道
    overlay_image = (image_np * (1 - alpha) + color_mask * alpha).astype(np.uint8)

    # 使用 Matplotlib 展示原图、mask 和叠加结果
    plt.figure(figsize=(15, 5))

    # 显示原图
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_pil)
    plt.axis("off")

    # 显示 Mask
    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    # 显示叠加结果
    plt.subplot(1, 3, 3)
    plt.title("Overlay Image")
    plt.imshow(overlay_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_box_on_image(color_image: Image, class_name: str, detect_info: List[Dict[str, Any]]):
    """
    在图像上绘制指定类别的检测框并直接展示

    参数:
        class_name (str): 需要绘制的目标类别名称（如 'car'）
        detect_info (list): 检测结果列表，每个元素是字典，包含：
                            {
                                "conf": 置信度,
                                "box": [x_min, y_min, x_max, y_max],
                                "class_name": 类别名称  # 必须包含此字段
                            }
        image (numpy.ndarray): 原始图像（OpenCV 格式，BGR）
    """
    for info in detect_info:
        # 检查类别名称是否匹配
        x_min, y_min, x_max, y_max = info["box"]
        conf = info["conf"]

        # 绘制矩形框
        color = (1, 0, 0)  # 绿色
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)

        # 绘制标签和置信度
        label = f"{class_name} {conf:.2f}"
        plt.text(x_min, y_min - 10, label, color=color, fontsize=12)

    # 展示结果
    plt.imshow(color_image)
    plt.axis('off')
    plt.show()
