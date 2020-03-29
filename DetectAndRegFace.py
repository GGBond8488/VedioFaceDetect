import cv2
import dlib
import numpy as np
import pandas as pd
'''
说明：
person_1:Frank
person_2:jimmy
person_3:Lip
'''


# 这里使用已存储的视频文件

cap = cv2.VideoCapture(
    r'VEDIO/shamless.mp4'
)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

print(size)



#计算欧式距离
def return_euclidean_distance(feature_1, feature_2):

    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    print(feature_1)
    print(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print("e_distance: ", dist)
    if dist > 0.4:
        return "diff"
    else:
        return "same"

# 处理存放所有人脸特征的 CSV
path_features_known_csv = "csv/features_all.csv"
csv_rd = pd.read_csv(path_features_known_csv, header=None)
# 存储的特征人脸个数
print(csv_rd.shape[0])

# 用来存放所有录入人脸特征的数组
features_known_arr = []

for i in range(csv_rd.shape[0]):
    features_someone_arr = []
    for j in range(0, len(csv_rd.loc[i, :])):
        features_someone_arr.append(csv_rd.loc[i, :][j])
    # print(features_someone_arr)
    features_known_arr.append(features_someone_arr)
print("Faces in Database：", len(features_known_arr))


#返回一张人脸多张图象的128D特征
def get_128d_features(img_gray):
    dets = detector(img_gray, 1)
    if len(dets) != 0:
        face_des = []
        for i in range(len(dets)):
            shape = predictor(img_gray, dets[i])
            face_des.append(Classfier_ResNet.compute_face_descriptor(img_gray, shape))
    else:
        face_des = []
    return face_des


# 使用shape_predictor_68_face_landmarks.dat，
Face_68_landmarks = r'model_feature/shape_predictor_68_face_landmarks.dat'
Face_classfier_Resnet = r'model_feature/dlib_face_recognition_resnet_model_v1.dat'
Face_classfier = r'mmod_human_face_detector.dat'
detector = dlib.get_frontal_face_detector()
#若需识别人脸所用
Classfier_ResNet = dlib.face_recognition_model_v1(Face_classfier_Resnet)
predictor = dlib.shape_predictor(Face_68_landmarks)
# 识别出人脸后要画的边框的颜色，BRG格式，openCV中与普通RGB值不同（历史遗留习惯）
color = (0, 255, 0)

font = cv2.FONT_HERSHEY_COMPLEX

fourcc = cv2.VideoWriter_fourcc(*'M','P','4','2')  # 保存视频的编码

out = cv2.VideoWriter('Face_Reg_outputbydlib.avi', fourcc, fps, size)
while cap.isOpened():
    ok, frame = cap.read()  # 读取一帧数据
   #如果读取失败，直接break
    if not ok:
        break
    # 读取成功后，将当前帧转换成灰度图像
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 人脸检测，使用detectMultiScale函数进行
    dets = detector(grey,1)
    pos_namelist = []
    name_namelist = []
    # 转换人脸对象中的人脸框并保存
    if len(dets) > 0:
        features_cap_arr = []
        for i in range(len(dets)):
            shape = predictor(frame, dets[i])
            features_cap_arr.append(Classfier_ResNet.compute_face_descriptor(frame, shape))
        for k in range(len(dets)):
            name_namelist.append("unknown")
            pos_namelist.append(
            tuple([dets[k].left(), int(dets[k].bottom() + (dets[k].bottom() - dets[k].top()) / 4)]))
            for i in range(len(features_known_arr)):
                print("with person_", str(i + 1), "the ", end='')
                compare = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                if compare == "same":  # 找到了相似脸
                    name_namelist[k] = "person_" + str(i + 1)
        for face in dets:
            cv2.rectangle(frame, (face.left() , face.top()), (face.right(), face.bottom()), color, 2)
            shape = predictor(frame, face)  # 寻找人脸的68个标定点
            # 遍历所有点，打印出其坐标，并圈出
            for pt in shape.parts():
                pt_pos = (pt.x, pt.y)
                cv2.circle(frame, pt_pos, 1, (0, 255, 0), 2)
        for i in range(len(dets)):
            cv2.putText(frame, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)


    # 存储显示图像
    out.write(frame)
    cv2.imshow("Face Detection by dlib", frame)
    c = cv2.waitKey(1)		#数值越小播放越流畅，不可为浮点数
    if c & 0xFF == ord('q'):
        break
# 释放并销毁所有窗口
cap.release()
out.release()
cv2.destroyAllWindows()

