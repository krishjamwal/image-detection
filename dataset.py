import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh

name = input("Enter your name: ")

if not os.path.exists("embeddings"):
    os.makedirs("embeddings")

cap = cv2.VideoCapture(0)

embeddings = []


def landmarks_to_embedding(landmarks):
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
    pts = pts.flatten()
    pts = pts / np.linalg.norm(pts)
    return pts

print("Show your face to the camera...")

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while len(embeddings) < 10:
        success, frame = cap.read()
        if not success:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
    
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            emb = landmarks_to_embedding(lm)
            embeddings.append(emb)
            print(f"Captured embedding {len(embeddings)}/10")

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) == ord('a'):
            break

cap.release()
cv2.destroyAllWindows()


final_embedding = np.mean(np.array(embeddings), axis=0)
np.save(f"embeddings/{name}.npy", final_embedding)

print(f"Registration complete! Saved embedding as {name}.npy")