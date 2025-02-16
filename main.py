import cv2
import numpy as np
import tensorflow as tf

# 加载训练好的模型
model = tf.keras.models.load_model("finger_count_mobilenetv2.h5")

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理图像
    img = cv2.resize(frame, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0  # 归一化

    # 预测
    prediction = model.predict(img)
    finger_count = np.argmax(prediction)

    # 显示结果
    cv2.putText(frame, f'Fingers: {finger_count}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Finger Count", frame)
    
    #按q结束
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
