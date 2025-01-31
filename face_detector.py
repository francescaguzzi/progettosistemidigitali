from multiprocessing import Value, Lock
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
import imutils
import numpy as np
import time
import dlib
import cv2

# ------------------------------------ #

def calculate_histograms(image, histograms, w, h):

    # nel caso sequenziale, consideriamo l'intera immagine
    for y in range(h):
        for x in range(w):
            idx = y * w + x  # indice del pixel nell'immagine
            pixel_value = image[idx]
            histograms[pixel_value] += 1 


# ------------------------------------ #

def calculate_cdfs(histograms, cdfs, clipLimit):

    # clipping dell'istogramma
    excess = 0
    if clipLimit > 0:
        for i in range(256):
            excess += max(histograms[i] - clipLimit, 0)
            histograms[i] = min(histograms[i], clipLimit)
        
        # ridistribuzione dell'eccesso
        binIncr = excess // 256
        
        for i in range(256):
            histograms[i] += binIncr
        
        if excess > 0:
            stepSz = 1 + (excess // 256)
            for i in range(256):
                if excess <= 0:
                    break
                excess -= stepSz
                histograms[i] += stepSz
    
    # calcolo delle CDF
    cdfs[0] = histograms[0]
    for i in range(1, 256):
        cdfs[i] = histograms[i] + cdfs[i - 1]

# ------------------------------------ #

def apply_cdfs(image, outputImage, cdfs, w, h):

    for y in range(h):
        for x in range(w):
            idx = y * w + x
            pixelValue = image[idx]
            equalizedPixel = (cdfs[pixelValue] - cdfs[0]) * 255 // (w * h - cdfs[0])
            outputImage[idx] = np.uint8(equalizedPixel)

# ------------------------------------ #

def histogram_equalization_sequential(input_image):

    height, width = input_image.shape

    histograms = np.zeros(256, dtype=np.int32)
    cdfs = np.zeros(256, dtype=np.int32)

    calculate_histograms(input_image.flatten(), histograms, width, height)

    clipLimit = 40
    calculate_cdfs(histograms, cdfs, clipLimit)

    output_image = np.empty_like(input_image)
    apply_cdfs(input_image.flatten(), output_image.flatten(), cdfs, width, height)

    return output_image


# ------------------------------------ #

def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

# ------------------------------------ #

def facial_recognition(orientation, blink, lock):

    EYE_AR_THRESH = 0.2
    EYE_AR_CONSEC_FRAMES = 3 # frame consecutivi in cui l'occhio deve rimanere chiuso 

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
        frame = cv2.flip(frame, 1)  

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        enhanced_gray = histogram_equalization_sequential(gray)

        rects = detector(enhanced_gray, 0)

        ear = 0
        orientation_value = 0 # 0 centro, 1 destra, 2 sinistra

        blink_event = False

        for rect in rects:

            # estrazione dei facial landmarks 
            shape = predictor(enhanced_gray, rect)
            shape = face_utils.shape_to_np(shape)

            # estrazione delle coordinate degli occhi e calcolo EAR
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # media dell'EAR per entrambi gli occhi
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # display del contorno degli occhi
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # blink detection
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
        

        cv2.putText(enhanced_gray, f"Blinks: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(enhanced_gray, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(enhanced_gray, f"Orientamento: {orientation_value}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


        # cv2.imshow("Frame", frame)
        cv2.imshow("Histogram Equalization", enhanced_gray)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
