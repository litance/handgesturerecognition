import cv2
import os
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np

# 创建数据集文件夹
save_path = "dataset"
os.makedirs(save_path, exist_ok=True)
for i in range(6):
    os.makedirs(os.path.join(save_path, str(i)), exist_ok=True)

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

processed_frame = None  # 保存当前骨架图

# 创建 Tkinter 窗口
root = tk.Tk()
root.title("手指数量数据采集 - 骨架图")
root.geometry("800x600")
label = Label(root)
label.pack()

def update_frame():
    global processed_frame
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)
        black_image = np.zeros(frame.shape, dtype=np.uint8)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    black_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )
        processed_frame = black_image.copy()
        display_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_image)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    label.after(10, update_frame)

def capture_image(finger_count):
    global processed_frame
    if processed_frame is not None:
        img_path = os.path.join(save_path, str(finger_count),
                                f"{len(os.listdir(os.path.join(save_path, str(finger_count))))}.jpg")
        cv2.imwrite(img_path, processed_frame)
        print(f"Saved {img_path}")
    else:
        print("No image to save!")

button_frame = tk.Frame(root)
button_frame.pack()
for i in range(6):
    btn = tk.Button(button_frame, text=f"手指 {i}", command=lambda i=i: capture_image(i), width=10, height=2)
    btn.grid(row=0, column=i, padx=10, pady=10)
exit_button = tk.Button(root, text="退出", command=root.quit, width=10, height=2, bg="red")
exit_button.pack(pady=20)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
