import cv2
import numpy as np
import mediapipe as mp
import math
import os
import pyaudio
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# カメラキャプチャの設定
camera_no = 0
video_capture = cv2.VideoCapture(camera_no)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                frames_per_buffer=1024,
                output=True)

if __name__ == '__main__':
    with mp_hands.Hands(static_image_mode=True,
            max_num_hands=2, # 検出する手の数（最大2まで）
            min_detection_confidence=0.5) as hands:        

        try:
            late_pos=[0,0]
            while video_capture.isOpened():
                # カメラ画像の取得
                ret, frame = video_capture.read()
                if ret is False:
                    print("カメラの取得できず")
                    break

                frame = cv2.flip(frame, 1)

                # CV2:BGR → MediaPipe:RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # 推論処理
                hands_results = hands.process(image)

                # 前処理の変換を戻しておく。
                image.flags.writeable = True
                write_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 有効なランドマーク（今回で言えば手）が検出された場合、
                # ランドマークを描画します。
                if hands_results.multi_hand_landmarks:
                    for hand in hands_results.multi_hand_landmarks:
                        now_pos=[hand.landmark[8].x,hand.landmark[8].y]

                        if late_pos!=[0,0]:
                            v=math.sqrt((now_pos[0]-late_pos[0])**2+(now_pos[1]-late_pos[1])**2)
                            if v<0.05:
                                pass
                            else:
                                samples = np.sin(np.arange(int(0.1 * 44100)) * v*10000 * np.pi * 2 / 44100)
                                stream.write(samples.astype(np.float32).tostring())
                        late_pos=now_pos
                    for landmarks in hands_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            write_image,
                            landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                # ディスプレイ表示
                cv2.imshow('hands', write_image)

                key = cv2.waitKey(1)
                if key == 27: # ESCが押されたら終了
                    print("終了")
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()