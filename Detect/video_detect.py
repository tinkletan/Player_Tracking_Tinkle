import cv2
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# 加载训练好的YOLOv8模型
model = YOLO('Player_Tracking/detect/train2/weights/best.pt')  # 替换为你训练好的模型路径

# 打开视频文件或摄像头
video_path = r"D:\SIT220\Player_Tracking\test3.mp4"  # 输入视频路径
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率和分辨率
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置输出视频文件
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO模型进行目标检测
    #results = model(frame)  # 进行推理
    results = model.track(frame, persist=True, show=False, tracker="botsort.yaml")

    # 假设results是YOLOv8模型推理后的输出
    for result in results:
        # 获取检测框、置信度和类别
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]  # 获取坐标 (x1, y1, x2, y2)
            conf = box.conf.item()  # 获取置信度
            cls = int(box.cls.item())  # 获取类别标签

            # 假设足球类别是0，如果模型类别索引不同，请修改
            if cls == 0 and conf>=0.4:
                label = f'Football {conf:.2f}'

                # 绘制矩形框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # 绘制标签
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 写入输出帧到视频
    out.write(frame)

    # 显示当前帧
    cv2.imshow('Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频资源
cap.release()
out.release()
cv2.destroyAllWindows()
