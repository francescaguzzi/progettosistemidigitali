from multiprocessing import Value, Lock
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
import imutils
import numpy as np
import time
import dlib
import cv2

# ------------------------------------------ #

def log_performance(func, report, *args, **kwargs):
    """
    Esegue la funzione 'func' con gli argomenti forniti, misura il tempo di esecuzione e scrive (in append)
    i risultati nel file 'performance_report.txt'.

    Parametri:
      - func: funzione da eseguire e misurare.
      - *args, **kwargs: argomenti posizionali e keyword da passare alla funzione.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Crea la stringa di log
    log_line = f"{func.__name__} eseguita in {elapsed_time:.6f} secondi\n"
    
    # Scrive (in append) il log sul file performance_report.txt
    with open(report, "a") as f:
        f.write(log_line)
    
    return result

# ------------------------------------------ #

def apply_clahe_opencv(image, clip_limit=2.0, tile_grid_size=(32, 32)):
    
    # Crea l'oggetto CLAHE con i parametri specificati
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Applica CLAHE all'immagine e restituisce il risultato
    output_image = clahe.apply(image)
    return output_image

# ------------------------------------------ #

def clahe(image, clip_limit=20, window_size=(32, 32)):
    """
    Applica il CLAHE (Contrast Limited Adaptive Histogram Equalization) calcolando,
    per ogni pixel, l'istogramma locale basato su un intorno di dimensione window_size (default 32x32).

    L'algoritmo procede come segue per ciascun pixel:
      1. Viene definito un intorno di dimensione window_size centrato sul pixel. Per gestire i bordi,
         l'immagine viene opportunamente "padded" replicando i bordi.
      2. Si calcola l'istogramma dell'intorno.
      3. Si applica il clipping: per ogni bin, se il conteggio supera clip_limit, l'eccesso viene conteggiato
         e il bin viene limitato a clip_limit.
      4. L'eccesso totale viene ridistribuito uniformemente (con eventuale gestione del resto) su tutti i bin.
      5. Si calcola la Cumulative Distribution Function (CDF) a partire dall'istogramma modificato.
      6. La CDF viene normalizzata per mappare i valori nel range [0, 255].
      7. Il pixel originale viene trasformato in base al valore della LUT derivante dalla CDF locale.

    Parametri:
      - image: numpy.ndarray in scala di grigi (dtype=np.uint8) contenente l'immagine di input.
      - clip_limit: valore intero che definisce il limite massimo per ciascun bin dell'istogramma.
      - window_size: tupla (h, w) che definisce la dimensione dell'intorno locale (default (32, 32)).

    Ritorna:
      - output_image: numpy.ndarray contenente l'immagine equalizzata.
    
    NOTA: Questo approccio è computazionalmente oneroso perché per ogni pixel viene
          ricalcolato l'istogramma della sua finestra locale.
    """

    win_h, win_w = window_size
    pad_h = win_h // 2
    pad_w = win_w // 2

    # esegue il padding sull'immagine per gestire i bordi (replicando il valore dei bordi)
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    height, width = image.shape
    output_image = np.zeros_like(image)

    # per ogni pixel dell'immagine 
    for i in range(height):
        for j in range(width):

            # estrae l'intorno locale 32x32 corrispondente
            local_window = padded_image[i:i + win_h, j:j + win_w]

            # calcola l'istogramma locale (256 bin per valori 0-255)
            hist, _ = np.histogram(local_window, bins=256, range=(0, 256))
            hist = hist.astype(np.uint32)

            # clipping dell'istogramma: se un bin supera clip_limit, si accumula l'eccesso
            if clip_limit > 0:
                excess = 0
                for k in range(256):
                    if hist[k] > clip_limit:
                        excess += hist[k] - clip_limit
                        hist[k] = clip_limit

                # ridistribuzione uniforme dell'eccesso
                bin_incr = excess // 256
                hist += bin_incr
                remainder = excess % 256
                for k in range(remainder):
                    hist[k] += 1

            # calcola la Cumulative Distribution Function (CDF)
            cdf = np.zeros(256, dtype=np.uint32)
            cdf[0] = hist[0]
            for i in range(1, 256):
                cdf[i] = cdf[i - 1] + hist[i]
            
            # normalizza la CDF per mappare i valori in [0,255]
            cdf_min = cdf.min()
            cdf_max = cdf.max()
            if cdf_max != cdf_min:
                cdf_normalized = ((cdf - cdf_min) * 255) / (cdf_max - cdf_min)
            else:
                cdf_normalized = np.zeros(256, dtype=np.uint8)

            # mappa il pixel corrente usando il valore corrispondente nella Look-Up Table
            pixel_val = image[i, j]
            new_val = np.clip(round(cdf_normalized[pixel_val]), 0, 255)
            output_image[i, j] = new_val

    return output_image

# ------------------------------------------ #

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

        enhanced_gray = log_performance(clahe, "report-sequenziale.txt", gray)
        # enhanced_gray = clahe(gray)

        rects = detector(enhanced_gray, 0)

        ear = 0
        orientation_value = 0 # 0 centro, 1 destra, 2 sinistra

        blink_event = False

        for rect in rects:

            # estrazione dei facial landmarks 
            shape = predictor(gray, rect)
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
