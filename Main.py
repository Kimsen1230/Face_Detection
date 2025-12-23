import cv2
import streamlit as st

# Streamlit page setup
st.set_page_config(page_title="Real-Time Face, Eyes & Smile Detection", layout="wide")
st.title("😄 Real-Time Face, Eyes, and Smile Detection")
st.markdown("""
This web app uses *OpenCV Haar Cascade Classifiers* to detect *faces, **eyes, and **smiles* in real time via your webcam.
""")

# Load Haar Cascade Models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Start webcam stream
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("⚠️ Unable to access webcam")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
         cv2.putText(frame, 'Hi i am Saif!', (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0 , 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Detect smiles
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)

    # Convert frame from BGR to RGB (Streamlit uses RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show frame in web app
    FRAME_WINDOW.image(frame)

camera.release()