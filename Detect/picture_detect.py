from ultralytics import YOLO
import cv2
import numpy as np

# 加载YOLOv8模型（可以是预训练模型，也可以是自定义训练的模型）
model = YOLO('Player_Tracking/detect/train/weights/best.pt')  # 使用YOLOv8小型模型，你可以选择不同的模型文件

# 读取图片
image_path = r'D:\SIT220\Player_Tracking\微信图片_20241129200946.jpg'  # 替换为你自己的图片路径
image = cv2.imread(image_path)

# 使用YOLO模型进行目标检测
results = model(image)  # 识别图片

## 假设results是YOLOv8模型推理后的输出
for result in results:
        # 获取检测框、置信度和类别
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]  # 获取坐标 (x1, y1, x2, y2)
            conf = box.conf.item()  # 获取置信度
            cls = int(box.cls.item())  # 获取类别标签

            # 假设足球类别是0，如果模型类别索引不同，请修改
            if cls == 0:
                label = f'Football {conf:.2f}'

                # 绘制矩形框
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # 绘制标签
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 保存或显示标注后的图片
cv2.imwrite('output_image.jpg', image)  # 保存标注后的图片
cv2.imshow('Detected Image', image)  # 显示标注后的图片
cv2.waitKey(0)
cv2.destroyAllWindows()
