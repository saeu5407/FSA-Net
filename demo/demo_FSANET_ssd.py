import os
import cv2
import sys

import pandas as pd

import warnings
warnings.filterwarnings("ignore")
sys.path.append('..')
from math import cos, sin
from lib.FSANET_model import *
import numpy as np
from keras.layers import Average

# RGB 이미지 생성하는 함수
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50):

    # 전처리 수행(라디안 단위로 변환)
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # 빈 구문, else가 메인
    # 정 중앙을 포인팅하기 위해 /2를 넣음.
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # 시각화 구간인데 싸인코싸인등이 적혀있음.
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

# 모델 예측 및 RGB 시각화 함수로 시각화하는 함수
def draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot):

    # loop over the detections
    if detected.shape[2]>0:

        for i in range(0, detected.shape[2]):

            # 예측에 관련된 신뢰도를 추출
            confidence = detected[0, 0, i, 2]

            # confidence가 낮은 경우는 필터링하여 제외
            # 그 전 Net에서 엥간하면 하나만 0.5를 넘는 것으로 보이지만, 그래도 하나만 뽑게끔 전처리를 진행
            if confidence > 0.5:

                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                (h0, w0) = input_img.shape[:2] # H,W를 뽑아 두는 구문
                box = detected[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
                (startX, startY, endX, endY) = box.astype("int")

                x1 = startX
                y1 = startY
                w = endX - startX
                h = endY - startY
                
                x2 = x1+w
                y2 = y1+h

                xw1 = max(int(x1 - ad * w), 0) # 여기에 들어가는 ad는 0.6
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1) # img_w, img_h는 이미지의 w,h를 뜻함. 근데 w0나 img_w나 같음
                yw2 = min(int(y2 + ad * h), img_h - 1)

                # faces[i,:,:,:] 는 (64,64,3)의 이미지
                # 즉 위에서 뽑아 온 구간의 이미지를 모델에 넣기 전 resize해 두는 과정
                faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces[i,:,:,:] = cv2.normalize(faces[i,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                face = np.expand_dims(faces[i,:,:,:], axis=0) # 모델 Predict를 위해 배치 차원을 추가했다고 생각. 즉 (1,64,64,3)
                p_result = model.predict(face) # 모델의 최종 결과값
                
                face = face.squeeze() # 빈 구문
                img = draw_axis(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], p_result[0][0], p_result[0][1], p_result[0][2])

                # 해당 부분에 이미지화 한 RGB를 덮어 씌운다.
                input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img
                
    cv2.imshow("result", input_img)

# 모델 불러오고 Face Detect 모델을 불러와 탐지 수행, 위의 최종 모델 및 시각화 함수 불러와 최종 시각화하는 메인함수
def main():

    # 임시 폴더 생성 부분
    try:
        os.mkdir('./img')
    except OSError:
        pass

    # 모델을 불러오는 데 있어 설정해야하는 내용들
    img_size = 64
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1
    img_idx = 0
    detected = '' # make this not local variable
    time_detection = 0
    time_network = 0
    time_plot = 0
    skip_frame = 1 # every 5 frame do 1 detection and network forward propagation
    ad = 0.6

    # 파라미터 부분
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3,3,3]
    lambda_d = 1
    num_classes = 3
    image_size = 64
    num_primcaps = 7*3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    # 모델1, 2를 불러온다.
    # 1은 1x1 Conv, 2는 Variance 옵션.
    # 세부적인 확인은 다음에
    model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    # 모델 3을 불러온다.
    # Uniform 옵션일 것으로 보이며, num_primcaps 파라미터를 3만 따로 조정하는 것만 체크해두자.
    num_primcaps = 8*8*3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    
    print('Loading models ...')

    # 각각의 Pre-trained된 가중치를 불러온다.
    weight_file1 = '../pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    model1.load_weights(weight_file1)
    print('Finished loading model 1.')
    
    weight_file2 = '../pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    model2.load_weights(weight_file2)
    print('Finished loading model 2.')

    weight_file3 = '../pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model3.load_weights(weight_file3)
    print('Finished loading model 3.')

    # 하단의 주석에서 각 옵션을 적어둔 것으로 보인다.
    # 앙상블은 단순한 Average를 사용한다.
    inputs = Input(shape=(64,64,3))
    x1 = model1(inputs) #1x1
    x2 = model2(inputs) #var
    x3 = model3(inputs) #w/o
    avg_model = Average()([x1,x2,x3])
    model = Model(inputs=inputs, outputs=avg_model)

    # FACE DETECTOR를 불러오는 부분
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # 비디오 캡쳐.
    cap = cv2.VideoCapture(0) # 0번 카메라의 영상 캡쳐
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024*1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768*1)

    print('Start detecting pose ...')
    detected_pre = np.empty((1,1,1)) # np.empty는 zeros와 비슷한데 메모리 할당만 받고 초기화는 진행하지 않는 방법

    while True:

        # 비디오 읽기
        ret, input_img = cap.read()

        img_idx += 1 # 인덱스 및 h,w 설정
        img_h, img_w, _ = np.shape(input_img)

        # 각 skip_frame횟수 마다 반복하는 부분
        # 여기서는 skip_frame이 1이므로 매 회 반복
        if img_idx==1 or img_idx%skip_frame == 0:

            # 초기화
            # SSD 에서는 사용하지 않는 옵션
            time_detection = 0
            time_network = 0
            time_plot = 0
            
            # 얼굴 탐지 진행.
            # GLAY SCALE도 SSD에서는 사용하지 않는 옵션
            gray_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)

            # DNN 모듈의 blob 함수를 통과하며 탐지 및 예측을 수행.
            """
            기존 모델 대비 성능이 좋아서 SDD 기반으로 많이들 사용한다. opencv는 순전파만 가능해서 caffe등의 프레임워크에서 학습된 모델을 불러온다.
            caffe framework에서는 4차원의 blob(binary large object)을 output한다.
            
            결과물인 detection에서,
            detected.shape[2]가 이 모델을 통해 얻을 수 있는 최대 검출 박스 수를 말한다. 이 차원을 i로 두고 각 박스의 값을 뽑을 수 있다. 
            detection[0,0,i,2] : 얼굴을 인식한 박스의 신뢰도(confidence)
            detection[0,0,i,3] : 전체 폭 중 박스 시작점의 x좌표 상대위치(왼쪽 맨 위 시작점)
            detection[0,0,i,4] : 전체 높이 중 박스 시작점의 y좌표 상대위치
            detection[0,0,i,5] : 전체 폭 중 박스 끝점의 x좌표 상대위치(오른쪽 맨 아래 끝점)
            detection[0,0,i,6] : 전체 높이 중 박스 끝점의 y좌표 상대위치
            를 뜻한다.
            
            모델의 파라미터를 확인해보면,
            여기서는 res10_300x300_ssd_iter_140000.caffemodel을 사용했기에 input size가 300,300이다.
            1.0은 scaler 부분인데 밑의 mean subtraction으로 뺀 후 값을 얼마로 나눌지의 값이다. 여기서는 안나누겠다는 뜻.
            (104.0, 177.0, 123.0)은 mean subtraction의 경험적 최적값으로, dnn이 계산하기 쉽게 이미지에서 빼주는 RGB값이다.
            """
            blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)),
                                         1.0,
                                         (300, 300),
                                         (104.0, 177.0, 123.0))
            net.setInput(blob) # 만든 blob 객체를 모델의 input으로 지정
            detected = net.forward() # forward 수행

            # blobFromImage으로 나온 결과값의 shape[2]는 이 모델을 통해 얻을 수 있는 최대 검출 박스 수
            # detected했는데 박스가 아예 안 나올 경우 이전의 데이터를 유지해서 넣는다는 뜻
            if detected_pre.shape[2] > 0 and detected.shape[2] == 0:
                detected = detected_pre

            # RGB를 그리기 위해 생성
            faces = np.empty((detected.shape[2], img_size, img_size, 3))

            # 모델링 + RGB를 그리는 함수를 실행하고 그린다.
            draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot)
        # skip_frame횟수 사이에 일어나는 구문으로 현재는 죽은 구문
        else:
            draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot)

        # skip_frame이 1이므로 매 3회마다 반복하는 구문
        # 얼굴 탐지가 실패했을 경우 이전 탐지 결과를 쓰기 위한 보험
        if detected.shape[2] > detected_pre.shape[2] or img_idx%(skip_frame*3) == 0:
            detected_pre = detected

        key = cv2.waitKey(100) # ms만큼 기다리는 함수.
        
if __name__ == '__main__':

    # 메인 함수 실행
    main()
