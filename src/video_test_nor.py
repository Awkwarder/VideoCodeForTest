import cv2
import numpy as np


def count_zero_submatrices_no_overlap(matrix):
    rows, cols = matrix.shape
    count = 0
    visited = np.zeros_like(matrix, dtype=bool)  # 用来标记每个像素是否已访问

    for i in range(rows - 3):  # 遍历到倒数第四行
        for j in range(cols - 3):  # 遍历到倒数第四列
            # 检查 4x4 子矩阵是否全零且未访问
            if not np.any(visited[i:i+4, j:j+4]) and np.all(matrix[i:i+4, j:j+4] == 0):
                # 发现一个全零 4x4 子矩阵，计数
                count += 1
                # 标记该子矩阵内的像素为已访问
                visited[i:i+4, j:j+4] = True

    return count



# 打开视频文件
cap = cv2.VideoCapture('../data/videoplayback.mp4')

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
index = 0
while cap.isOpened():
    index += 1
    ret, frame = cap.read()

    if not ret:
        break

    # 1. 将帧转换为 YUV 色彩空间
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # 2. 提取Y分量
    y = frame_yuv[:, :, 0]

    # 3. 将Y分量的数值类型由 uint8 转为 short（有符号的两字节数值）
    y_short = np.array(y, dtype=np.int16)

    # 4. 将y_short矩阵整体向右移动一个像素单位
    y_short_right = np.roll(y_short, 1, axis=1)

    # 5. 计算y_diff = y_short - y_short_right
    y_diff = y_short - y_short_right

    # 6. 删除第一列
    y_diff = y_diff[:, 1:]

    # 7. 如果y_diff的列数为奇数，增加一列
    if y_diff.shape[1] % 2 != 0:
        y_diff = np.hstack([y_diff, np.zeros((y_diff.shape[0], 1), dtype=np.int16)])
    count_roll1 = count_zero_submatrices_no_overlap(y_diff)
    y_diff_right = np.roll(y_diff, 1, axis=1)
    y_diff = y_diff - y_diff_right
    # 7. 如果y_diff的列数为奇数，增加一列
    if y_diff.shape[1] % 2 != 0:
        y_diff = np.hstack([y_diff, np.zeros((y_diff.shape[0], 1), dtype=np.int16)])
    count_roll2 = count_zero_submatrices_no_overlap(y_diff)
    den =  float(y_diff.shape[1] * y_diff.shape[0])
    print("index = ",index ,round(float(count_roll1) *16 *100/den,2) ,"%  ",round(float(count_roll2) *16 *100/den,2) ,"%")
    '''
    # 8. 对y_diff做DCT变换
    y_diff_dct = cv2.dct(np.float32(y_diff))
    
    # 9. 取绝对值并归一化到[0, 255]范围
    y_diff_dct_abs = np.abs(y_diff_dct)
    y_diff_dct_norm = cv2.normalize(y_diff_dct_abs, None, 0, 255, cv2.NORM_MINMAX)

    # 10. 统计绝对值小于0.001的数量
    small_values_count = np.count_nonzero(np.abs(y_diff_dct_norm) < 0.01)

    print(f"Number of values with absolute less than 0.01 in y_diff_dct_norm: {small_values_count}")
    
    # 11. 显示y_diff_dct的结果
    cv2.imshow('Y Diff DCT', np.uint8(y_diff_dct_norm))

    # 按'q'键退出
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break
    '''
cap.release()
cv2.destroyAllWindows()