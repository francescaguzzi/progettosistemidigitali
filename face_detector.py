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

    Ritorna:
      - Il risultato della funzione 'func'.
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

def clahe_sequenziale(image, clip_limit=20, tile_size=(32, 32)):
    """
    Applica il metodo CLAHE (Contrast Limited Adaptive Histogram Equalization) in modo sequenziale
    sull'immagine in scala di grigi, suddividendola in tile (blocchi) di dimensioni specificate (default 32x32).

    Per ogni tile:
      - Viene calcolato l'istogramma dei pixel.
      - Si applica il clipping dell'istogramma: per ciascun livello di intensità, se il conteggio supera
        il valore clip_limit, l'eccesso viene conteggiato e successivamente ridistribuito uniformemente.
      - Si calcola la Cumulative Distribution Function (CDF) a partire dall'istogramma modificato.
      - La CDF viene normalizzata per ottenere una mappatura dei valori originali (0-255) a valori equalizzati.
      - Infine, la trasformazione viene applicata al tile e il risultato viene copiato nell'immagine di output.

    Parametri:
      - image: numpy.ndarray in scala di grigi (dtype=np.uint8) contenente l'immagine di input.
      - clip_limit: valore intero che definisce il limite massimo per ciascun bin dell'istogramma prima di effettuare il clipping.
      - tile_size: tupla (tile_height, tile_width) che definisce la dimensione di ciascun blocco (tile).

    Ritorna:
      - output_image: numpy.ndarray contenente l'immagine equalizzata.
    """
    
    height, width = image.shape
    tile_height, tile_width = tile_size
    
    num_tiles_y = height // tile_height
    num_tiles_x = width // tile_width
    
    output_image = np.zeros_like(image)
    

    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            # estrazione delle coordinate
            y_start = ty * tile_height
            y_end = y_start + tile_height
            x_start = tx * tile_width
            x_end = x_start + tile_width
            
            tile = image[y_start:y_end, x_start:x_end]
            
            # calcolo degli istogrammi
            hist = np.zeros(256, dtype=np.uint32)
            for y in range(tile_height):
                for x in range(tile_width):
                    pixel_value = tile[y, x]
                    hist[pixel_value] += 1
            
            # clipping 
            if clip_limit > 0:
                excess = 0
                for i in range(256):
                    if hist[i] > clip_limit:
                        excess += hist[i] - clip_limit
                        hist[i] = clip_limit
                
                # ridistribuzione degli eccessi
                bin_incr = excess // 256
                for i in range(256):
                    hist[i] += bin_incr
                remaining = excess % 256
                for i in range(remaining):
                    hist[i] += 1
            
            # calcolo della CDF
            cdf = np.zeros(256, dtype=np.uint32)
            cdf[0] = hist[0]
            for i in range(1, 256):
                cdf[i] = cdf[i - 1] + hist[i]
            
            # normalizzazione della CDF
            cdf_min = cdf.min()
            cdf_max = cdf.max()
            if cdf_max != cdf_min:
                cdf_normalized = ((cdf - cdf_min) * 255) / (cdf_max - cdf_min)
            else:
                cdf_normalized = np.zeros(256, dtype=np.uint8)
            
            # applica la trasformazione al tile
            for y in range(tile_height):
                for x in range(tile_width):
                    pixel_value = tile[y, x]
                    equalized_pixel = cdf_normalized[pixel_value]
                    output_image[y_start + y, x_start + x] = equalized_pixel

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

        # enhanced_gray = histogram_equalization_sequential(gray)
        enhanced_gray = log_performance(clahe_sequenziale, "report-sequenziale.txt", gray)

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
