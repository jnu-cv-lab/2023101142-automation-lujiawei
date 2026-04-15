# Python 图像处理基础实验
使用 OpenCV 实现图像读取、灰度转换、裁剪和保存

## 项目文件
├── test.jpg          # 测试图片
├── test.py           # 主程序
├── gray_test.jpg     # 生成的灰度图
├── crop_test.jpg     # 生成的裁剪图
└── README.md         # 说明文档

## 环境依赖
Python 3.x
opencv-python
numpy
matplotlib

## 安装命令
pip install opencv-python numpy matplotlib

## 运行方法
将 test.jpg 放在同一文件夹下，运行：
python test.py

## 实现功能
1. 读取图像 test.jpg
2. 输出图像尺寸、通道、数据类型
3. 显示原图
4. 转换为灰度图并保存
5. 裁剪图像并保存