import cv2
import mediapipe as mp
import random as rd
import os
import hand_detection as hd

detector = hd.handDetector()
cam = cv2.VideoCapture(0)

def draw_results(frame, user_draw):
    # Cho máy sinh ra lựa chọn ngẫu nhiên
    com_draw = rd.randint(0,2)

    # Vẽ hình, viết chữ theo user_dram
    frame = cv2.putText(frame,'You', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    s_img = cv2.imread(os.path.join("pix", str(user_draw) + ".png"))
    print(s_img)
    x_offset = 50
    y_offset = 100
    frame[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img

    # Vẽ hình, viết chữ theo com_dram
    frame = cv2.putText(frame,'Com', (400,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    s_img = cv2.imread(os.path.join("pix", str(com_draw) + ".png"))
    x_offset = 400
    y_offset = 100
    frame[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img

    # Kiểm tra và hiển thị kết quả
    if user_draw == com_draw:
        result = "DRAM!!!"
    elif ((user_draw == 0) and (com_draw==1)) or ((user_draw == 1) and (com_draw==2)) or ((user_draw == 2) and (com_draw==0)):
        result = "YOU WIN!!!"
    else:
        result = "YOU LOSE!!"
    
    frame = cv2.putText(frame,result,(50,550),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2,cv2.LINE_AA)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1) # Dùng để lật lại camera

    # Đưa hình ảnh vào detector
    frame, hand_ls = detector.findHands(frame)
    n_fingers = detector.count_finger(frame)
 
    user_draw = -1 # Lưu lại các giá trị mà người dùng đưa ra
    # 0: Lá, 1: Đấm, 2: Kéo

    if n_fingers == 0:
        user_draw = 1
    elif n_fingers == 2:
        user_draw = 2
    elif n_fingers == 5:
        user_draw = 0
    elif n_fingers != -1:
        print("Chỉ chấp nhận Đám Lá Kéo")
    else:
        print("Không có bàn tay trong khung hình")


    key =  cv2.waitKey(1)
    cv2.imshow("Game", frame)
    if  key == ord("q"):
        break
    elif key == ord(" "):
        draw_results(frame, user_draw)
        cv2.imshow("Game", frame)
        cv2.waitKey()

cam.release()
cv2. destroyAllWindows()