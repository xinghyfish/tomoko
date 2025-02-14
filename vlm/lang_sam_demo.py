import numpy as np
from PIL import Image
from lang_sam import LangSAM
import matplotlib.pyplot as plt


def show_masks_on_image(image_pil):
    # 假设 results 是从模型返回的预测结果
    # 通常包含一个 mask 列表，形状为 (H, W) 的布尔值（True 表示属于分割区域）
    # 提取第一个结果的 mask
    masks = results[0]['masks'] # 假设 results 是一个字典列表，'mask' 包含分割掩码
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


if __name__ == '__main__':
    model = LangSAM("sam2.1_hiera_small", "checkpoints/sam2.1/sam2.1_hiera_small.pt")
    image = Image.open("./assets/4cups.jpg").convert("RGB")
    text_prompt = "cup"
    results = model.predict([image], [text_prompt])
    print(results[0]['masks'])
    show_masks_on_image(image)
