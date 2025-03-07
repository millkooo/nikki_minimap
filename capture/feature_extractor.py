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


# 需要处理的文件列表
file_names = [
    'huayanqundao',
    'huayuanzhen',
    'qiyuansenlin',
    'weifenglvye',
    'xiaoshishu',
    'panduan'
]

sift = cv2.SIFT_create(nOctaveLayers=5, contrastThreshold=0.01, sigma=1)

for name in file_names:
    # 加载图像
    img_path = f'../resources/img/{name}.png'
    train_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if train_img is None:
        print(f"警告：文件 {name}.png 未找到，已跳过")
        continue

    # 提取特征
    kp, des = sift.detectAndCompute(train_img, None)

    # 保存特征数据
    save_keypoints(kp, des, f'../resources/features/{name}_features.npz')

    # 生成并保存特征点可视化图像
    output_img = cv2.drawKeypoints(train_img, kp, None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(f'../resources/keypoints/{name}_keypoints.png', output_img)

    print(f"{name} 处理完成，特征保存至 {name}_features.npz，渲染图保存至 {name}_keypoints.png")

print("所有文件处理完成！")


