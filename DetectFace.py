import cv2
# 这里使用已存储的视频文件

cap = cv2.VideoCapture(
    r'VEDIO/shamless.mp4'
)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

print(size)

# 告诉OpenCV使用人脸识别分类器在哪里，
# 使用haarcascade_frontalface_alt2.xml，
classfier = cv2.CascadeClassifier(
    r'model_feature/haarcascades/haarcascade_frontalface_alt2.xml'
)

# 识别出人脸后要画的边框的颜色，BRG格式，openCV中与普通RGB值不同（历史遗留习惯）
color = (0, 255, 0)
fourcc = cv2.VideoWriter_fourcc(*'M','P','4','2')  # 保存视频的编码
out = cv2.VideoWriter('output.avi', fourcc, fps, size)
while cap.isOpened():
    ok, frame = cap.read()  # 读取一帧数据
   #如果读取失败，直接break
    if not ok:
        break
    # 读取成功后，将当前帧转换成灰度图像
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 人脸检测，使用detectMultiScale函数进行
    faceRects = classfier.detectMultiScale(
        grey,
        scaleFactor=1.25,
        minNeighbors=3,
        minSize=(35, 35) #设置最小检测范围，即是目标小于该值就无视
        # maxSize=(200,200)#设置最大检测范围
    )
    if len(faceRects) > 0:  # 当检测到多张脸时
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            cv2.rectangle(frame, (x , y), (x + w, y + h), color, 2)

    # 存储显示图像
    out.write(frame)
    cv2.imshow("Face Detection", frame)
    c = cv2.waitKey(1)		#数值越小播放越流畅，不可为浮点数
    if c & 0xFF == ord('q'):
        break
# 释放并销毁所有窗口
cap.release()
out.release()
cv2.destroyAllWindows()

