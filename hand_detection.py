import cv2
import mediapipe as mp

# Viết theo hướng đối tượng để thuận tiện cho việc quản lí code

class handDetector():
    # Khởi tạo -  định nghĩa đối tượng
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        # Vẽ ra các đốt ngón tay
        self.mpDraw = mp.solutions.drawing_utils 
    
    # Hàm tìm đối tượng (tay) trong ảnh
    def findHands(self, img):
        # Chuyển từ BGR sang RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Đưa qua thư viện mediapipe
        results = self.hands.process(imgRGB)
        
        # Lưu các giá trị của đốt ngón tay
        hand_lm_list = []
        # Trả về kết quả để nhận biết xem có đối tượng (tay) ở trong ảnh hay không
        if results.multi_hand_landmarks:
            # Vẽ landmark cho các bàn tay
            for handlm in results.multi_hand_landmarks:
                # Vẽ các đường nối giữa các đốt
                self.mpDraw.draw_landmarks(img, handlm, self.mpHands.HAND_CONNECTIONS)

                # Trích ra các tọa độ của các khớp ngón tay
                # Để tiện cho việc quan sát nên chỉ lấy ra tọa độ của bàn tay đầu tiên
                firstHand = results.multi_hand_landmarks[0]

                # Duyệt trong landmark cảu bàn tay đầu tiên để nhìn xem có bao nhiêu ngón tay được xòe ra
                # Lấy chiểu dài, chiều rộng của ảnh
                h, w, _ = img.shape
                for id, lm in enumerate(firstHand.landmark):
                    # Lấy các đốt ngón tay
                    # Tìm real_x, real_y
                    real_x, real_y = int(lm.x * w), int(lm.y * h)
                    hand_lm_list.append([id, real_x, real_y])
        
        return img, hand_lm_list
    
    # Đếm số lượng các đốt ngón tay
    def count_finger(self, hand_lm_list):
        finger_start_index = [4,8,12,16,20]
        n_fingers = 0 # Số lượng fingers từ 0 đế 5

        if len(hand_lm_list)>0:
            # Kiểm tra ngón cái Thump - số 0 là ngón cai
            # Có thể sử dụng ảnh HandLandmarks.pnd để có thể hình dung dễ hơn
            if (hand_lm_list[finger_start_index[0]][1]<hand_lm_list[finger_start_index[0]-1][1]).all():
                n_fingers += 1
            
            # Kiểm tra các ngón còn lại
            for idx in range(1,5):
                if (hand_lm_list[finger_start_index[idx]][2] < hand_lm_list[finger_start_index[idx]-2][2]).all():
                    n_fingers += 1
        
            return n_fingers
        else: 
            return -1               