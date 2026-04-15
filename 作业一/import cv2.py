import cv2
import numpy as np

# 1. 读取并显示原图
img = cv2.imread("test.jpg")
if img is None:
    print("错误：找不到图片文件！请检查路径是否正确。")
    exit()

cv2.imshow("原图", img)

# 2. 分别进行高斯滤波和双边滤波，对比效果
# 高斯滤波（仅空间距离）
gaussian_blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)

# 双边滤波（空间距离 + 灰度相似度）
# 参数说明：
# - d: 滤波窗口直径（控制处理范围）
# - sigmaColor: 颜色差异的标准差（越大，允许更多不同颜色参与平滑）
# - sigmaSpace: 空间距离的标准差（越大，影响范围越广）
bilateral_blur = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)

# 3. 显示对比效果
cv2.imshow("高斯滤波（会模糊边缘）", gaussian_blur)
cv2.imshow("双边滤波（保留边缘+平滑噪点）", bilateral_blur)

# 等待按键，按任意键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()