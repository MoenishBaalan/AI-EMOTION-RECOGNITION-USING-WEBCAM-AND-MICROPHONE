import cv2
import numpy as np
import sounddevice as sd
from fer import FER
from pyAudioAnalysis import ShortTermFeatures
import threading
from collections import deque


def video_emotion(shared):
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion_text = "Video: None"
        emotion, score = detector.top_emotion(frame)
        if emotion:
            emotion_text = f"Video: {emotion} ({round(score,2)})"
            shared["video"] = emotion

        audio_text = f"Audio: {shared.get('audio','None')}"

        cv2.putText(frame, emotion_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, audio_text, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Emotion Fusion System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            shared["stop"] = True
            break

    cap.release()
    cv2.destroyAllWindows()


def audio_emotion(shared):
    fs = 16000
    chunk = 1
    energy_q = deque(maxlen=5)
    zcr_q = deque(maxlen=5)

    while not shared.get("stop", False):
        audio = sd.rec(int(chunk * fs),
                        samplerate=fs,
                        channels=1,
                        dtype='int16')
        sd.wait()

        audio = audio.flatten().astype(float) / 32768.0

        features, _ = ShortTermFeatures.feature_extraction(
            audio, fs, 0.05 * fs, 0.025 * fs)

        energy = np.mean(features[1])
        zcr = np.mean(features[0])

        energy_q.append(energy)
        zcr_q.append(zcr)

        avg_energy = np.mean(energy_q)
        avg_zcr = np.mean(zcr_q)

        if avg_energy < 0.01:
            emotion = "calm"
        elif avg_zcr > 0.15:
            emotion = "happy/excited"
        else:
            emotion = "sad/angry"

        shared["audio"] = emotion


if __name__ == "__main__":
    shared = {}

    t1 = threading.Thread(target=video_emotion, args=(shared,))
    t2 = threading.Thread(target=audio_emotion, args=(shared,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
