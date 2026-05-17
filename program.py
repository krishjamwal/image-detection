import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

THRESHOLD = 0.90
WINDOW = 5      
pred_window = []
marked = set()


embeddings = {}
for f in os.listdir("embeddings"):
    if f.endswith(".npy"):
        name = f[:-4]
        emb = np.load(os.path.join("embeddings", f))
        embeddings[name] = emb


def cos_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def lm_to_emb(landmarks):
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks]).flatten()
    return pts / np.linalg.norm(pts)

def mark_attendance(name):
    if name not in marked:
        with open("attendance.csv", "a") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{name},{now}\n")
        marked.add(name)


mp_face = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

with mp_face.FaceMesh(max_num_faces=1) as fm:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = fm.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            emb = lm_to_emb(lm)

            name, best = "Unknown", 0
            for n, e in embeddings.items():
                score = cos_sim(emb, e)
                if score > best:
                    best = score
                    name = n

         
            if best < THRESHOLD:
                name = "Unknown"

           
            pred_window.append(name)
            if len(pred_window) > WINDOW:
                pred_window.pop(0)

          
            final = max(set(pred_window), key=pred_window.count)

            if final != "Unknown":
                mark_attendance(final)
                text, color = f"{final} ({best:.2f})", (0, 255, 0)
            else:
                text, color = "Unknown", (0, 0, 255)

        else:
            text, color = "No Face", (0, 0, 255)

        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow("Face Attendance", frame)
        if cv2.waitKey(1) == ord('a'):
            break

cap.release()
cv2.destroyAllWindows()





