from multiprocessing import Value, Lock
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
import imutils
import numpy as np
import time
import dlib
import cv2


def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear



def facial_recognition(orientation, blink, lock):

    # Initialize constants for EAR threshold and consecutive frames
    EYE_AR_THRESH = 0.2 
    EYE_AR_CONSEC_FRAMES = 3

    COUNTER = 0
    TOTAL = 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    vs = VideoStream(src=0).start()
    time.sleep(1.0)


    while True:

        frame = vs.read()
        frame = cv2.flip(frame, 1)  # mirrors image

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        ear = 0
        orientation_value = 0 # 0 centro, 1 destra, 2 sinistra

        blink_event = False

        for rect in rects:

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # average the eye aspect ratio together for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Check if the eyes are closed (blink detection)
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    blink_event = True
                    TOTAL += 1
                COUNTER = 0

            leftEyeCenter = np.mean(leftEye, axis=0)
            rightEyeCenter = np.mean(rightEye, axis=0)
            nosePoint = shape[33]

            TOLERANCE = 10

            distLeft = np.linalg.norm(nosePoint - leftEyeCenter)
            distRight = np.linalg.norm(nosePoint - rightEyeCenter)

            if abs(distLeft - distRight) < TOLERANCE:
                orientation_value = 0 # CENTRO
            elif distLeft < distRight:
                orientation_value = 1 # DESTRA
            else:
                orientation_value = 2 # SINISTRA

        with lock:
            orientation.value = orientation_value
            blink.value = blink_event
        
        # Display the blink count
        cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Orientamento: {orientation_value}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
