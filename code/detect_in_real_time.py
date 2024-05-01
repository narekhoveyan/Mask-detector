import config
import cv2
import numpy as np
from keras.models import load_model

if __name__ == "__main__":
    # loading the classifier for detection face
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    # accessing webcam
    video_capture = cv2.VideoCapture(0)


    # function for detecting faces in frame
    def detect_face(frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, 1.1, 5, minSize=(40, 40))
        x, y, w, h = 0, 0, 0, 0
        for (x, y, w, h) in faces:
            faces = frame[y:y + h, x:x + w, :]
        return faces, x, y, w, h

    def draw_rectangle(frame, label, color, x, y, w, h):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(video_frame, f'{label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    while True:

        result, video_frame = video_capture.read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfully

        model = load_model(config.MODEL_PATH)
        faces, x, y, w, h = detect_face(video_frame)

        faces = np.resize(faces, config.INPUT_SHAPE)
        faces = np.array([faces])  # Convert single image to a batch.
        predictions = model.predict(faces)
        state = predictions[0].argmax()
        if state == 0:
            draw_rectangle(video_frame, "Mask worn correct, thanks",(0,255,0), x, y, w, h)
        elif state == 1:
            draw_rectangle(video_frame, "Please wear mask",(0,0,255), x, y, w, h)
        elif state == 2:
            draw_rectangle(video_frame, "Mask worn incorrect",(0,0,255), x, y, w, h)
        # display the processed frame in a window
        cv2.imshow("Mask detector", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# %%
