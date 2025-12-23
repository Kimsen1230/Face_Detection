from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

camera = None

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        face_detected = False
        eyes_detected = False
        smile_detected = False

        for (x, y, w, h) in faces:
            face_detected = True
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Eyes (upper face)
            eyes = eye_cascade.detectMultiScale(roi_gray[0:int(h/2), :], 1.1, 8)
            if len(eyes) > 0:
                eyes_detected = True

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Smile (lower face)
            smiles = smile_cascade.detectMultiScale(roi_gray[int(h/2):h, :], 1.7, 22)
            if len(smiles) > 0:
                smile_detected = True

            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(
                    roi_color,
                    (sx, sy + int(h/2)),
                    (sx+sw, sy+sh + int(h/2)),
                    (0, 255, 255), 2
                )

        # Status text at bottom
        y = frame.shape[0] - 20

        if face_detected:
            cv2.putText(frame, "Face Detected", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y -= 30

        if eyes_detected:
            cv2.putText(frame, "Eyes Detected", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y -= 30

        if smile_detected:
            cv2.putText(frame, "Smile Detected", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop")
def stop():
    global camera
    if camera:
        camera.release()
    return "Camera stopped"


if __name__ == "__main__":
    app.run(debug=True)
