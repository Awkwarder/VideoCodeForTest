from pickletools import uint8

import cv2
import numpy as np
# 打开视频文件
cap = cv2.VideoCapture('../data/videoplayback.mp4')

if not cap.isOpened():
    print("Error: Couldn't open the video.")
    exit()

index = 0
# 逐帧读取视频
while True:
    ret, frame = cap.read()

    # 如果读取成功，ret 为 True，frame 为当前帧
    if not ret:
        break

    # 将 BGR 图像转换为 YUV 色彩空间
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # 提取 Y 分量（亮度）
    y_channel = yuv_frame[:, :, 0]

    # 获取图像的高度和宽度
    height, width = y_channel.shape

    # 创建一个新的图像用于存储计算结果
    new_y_channel = y_channel.copy()
    count = 0
    index += 1
    maxV = 0
    zeroCount = 0
    # 遍历每个像素（从第二行和第二列开始，避免边界越界）
    for i in range(1, height):
        for j in range(1, width):
            # 当前像素的值
            current_pixel = y_channel[i, j]

            # 左边像素的值
            left_pixel = y_channel[i, j - 1]

            # 上面像素的值
            top_pixel = y_channel[i - 1, j]

            if left_pixel % 2 == 0 or top_pixel % 2 == 0:
                top_left_sum = left_pixel//2 + top_pixel//2
            else:
                top_left_sum = left_pixel//2 + top_pixel//2 + 1

            if top_left_sum > current_pixel:
                new_pixel = top_left_sum - current_pixel
            else:
                new_pixel = current_pixel - top_left_sum

            new_y_channel[i, j] = new_pixel
            if new_y_channel[i, j] > 63:
                count += 1
                if maxV < new_y_channel[i, j]:
                    maxV = new_y_channel[i, j]
            if new_y_channel[i, j]  == 0:
                zeroCount += 1

    print("frame number: ", index, " ", float(count) * 100 / (y_channel.shape[0] * y_channel.shape[1]) ,"% maxV = ", maxV ," zeroCount ", float(zeroCount) * 100 / (y_channel.shape[0] * y_channel.shape[1]) ,"%")


    # 显示 Y 分量
    cv2.imshow('new_y_channel', new_y_channel)

    # y_channel_copy = y_channel.copy()


    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()