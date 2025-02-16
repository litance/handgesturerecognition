import cv2
import os
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# 数据集存储路径
save_path = "dataset"
os.makedirs(save_path, exist_ok=True)

# 创建 0-5 的文件夹
for i in range(6):
    os.makedirs(os.path.join(save_path, str(i)), exist_ok=True)

# 初始化 OpenCV 摄像头
cap = cv2.VideoCapture(0)

# 创建 Tkinter 窗口
root = tk.Tk()
root.title("手指数量数据采集")

# 设定窗口大小
root.geometry("800x600")

# 创建标签显示 OpenCV 画面
label = Label(root)
label.pack()

# 处理 OpenCV 图像并更新 Tkinter 界面
def update_frame():
    ret, frame = cap.read()
    if ret:
        # 转换 BGR 为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.after(10, update_frame)

# 采集图像函数
def capture_image(finger_count):
    ret, frame = cap.read()
    if ret:
        img_path = os.path.join(save_path, str(finger_count), f"{len(os.listdir(os.path.join(save_path, str(finger_count))))}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")

# 创建按钮
button_frame = tk.Frame(root)
button_frame.pack()

for i in range(6):
    btn = tk.Button(button_frame, text=f"手指 {i}", command=lambda i=i: capture_image(i), width=10, height=2)
    btn.grid(row=0, column=i, padx=10, pady=10)

# 退出按钮
exit_button = tk.Button(root, text="退出", command=root.quit, width=10, height=2, bg="red")
exit_button.pack(pady=20)

# 启动 OpenCV 画面更新
update_frame()

# 运行 Tkinter 事件循环
root.mainloop()

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
