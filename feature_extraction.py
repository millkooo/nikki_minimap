import cv2
import numpy as np

def save_keypoints(kp, des, filename):
    # 将关键点转换为可序列化格式
    kp_data = []
    for p in kp:
        kp_data.append((
                p.pt[0], p.pt[1],
                p.size, p.angle,
                p.response, p.octave,
                p.class_id
            ))
    np.savez(filename, kp=np.array(kp_data), des=des)

# 加载大图像
train_img = cv2.imread('result100.png', cv2.IMREAD_GRAYSCALE)
if train_img is None:
    raise FileNotFoundError("大图像未找到，请检查路径是否正确！")

# 提取特征
sift = cv2.SIFT_create(nOctaveLayers = 5, contrastThreshold=0.01, edgeThreshold=15)

kp2, des2 = sift.detectAndCompute(train_img, None)

# 保存特征数据
save_keypoints(kp2, des2, 'preprocessed_features.npz')
print("预处理完成！特征数据已保存为preprocessed_features.npz")

# 渲染特征点
output_img = cv2.drawKeypoints(train_img, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 保存带有特征点的图像
cv2.imwrite('keypoints_image.png', output_img)
print("特征点渲染完成，并保存为keypoints_image.png")

