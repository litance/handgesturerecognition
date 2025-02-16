import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# 加载模型
model = tf.keras.models.load_model("finger_count_mobilenetv2.h5")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

# 处理手部检测
with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        black_image = np.zeros(frame.shape, dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制骨架
                mp_drawing.draw_landmarks(
                    black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

                # 提取 ROI
                h, w, _ = frame.shape
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w)
                y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h)

                hand_roi = frame[y_min:y_max, x_min:x_max]

                # 预测手指数
                if hand_roi.size > 0:
                    hand_roi = cv2.resize(hand_roi, (128, 128)) / 255.0
                    prediction = model.predict(np.expand_dims(hand_roi, axis=0))
                    finger_count = np.argmax(prediction)

                    # 显示预测结果
                    cv2.putText(black_image, f'Fingers: {finger_count}',
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 0), 2)

        cv2.imshow("Hand Skeleton & Finger Count", black_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
