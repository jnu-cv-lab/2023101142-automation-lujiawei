import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========== 任务1：读取图片 ==========
img = cv2.imread("test.jpg")
if img is None:
    print("图片读取失败，请检查路径！")
else:
    print("图片读取成功！")

    # ========== 任务2：输出图像基本信息 ==========
    height, width, channels = img.shape
    dtype = img.dtype
    print(f"图像尺寸：宽度={width}, 高度={height}")
    print(f"图像通道数：{channels}")
    print(f"图像数据类型：{dtype}")

    # ========== 任务3：显示原图 ==========
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure("原图")
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

    # ========== 任务4：转灰度图并显示 ==========
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure("灰度图")
    plt.imshow(gray_img, cmap="gray")
    plt.axis("off")
    plt.show()

    # ========== 任务5：保存灰度图 ==========
    cv2.imwrite("gray_test.jpg", gray_img)
    print("灰度图已保存为 gray_test.jpg")

    # ========== 任务6：NumPy 简单操作 ==========
    pixel_value = img[100, 100]
    print(f"像素 (100,100) 的 BGR 值：{pixel_value}")
    crop_img = img[0:100, 0:100]
    cv2.imwrite("crop_test.jpg", crop_img)
    print("裁剪区域已保存为 crop_test.jpg")