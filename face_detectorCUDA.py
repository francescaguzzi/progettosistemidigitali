from multiprocessing import Value, Lock
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
import imutils
import numpy as np
import time
import dlib
import cv2

import pycuda.driver as cuda 
import pycuda.autoinit
from pycuda.compiler import SourceModule

# ------------------------------------ #

# KERNEL CUDA #

# kernel calcolo degli istogrammi

mod_hi = SourceModule("""
                   
__global__ void calculate_histograms(unsigned char *image, unsigned int *histograms, unsigned int w, unsigned int h) {   

                   
    //ogni blocco corrisponde ad un pixel, un thread all'interno del blocco per un pixel della finestra
                               
    //indice del blocco
    unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x ;
                   
    //indice all'interno della finestra
    //unsigned int bid = threadIdx.y * blockDim.x + threadIdx.x ;
                   
    //indice globale
    // unsigned int tid = bid * (blockDim.x * blockDim.y) + bid ;
                   
    //indice di scorrimento dell'immagine
    unsigned int idx = bid + threadIdx.x + threadIdx.y * gridDim.x;
        
    if ( idx < w*h ){
        //viene aggiornato il counter per il livello di luminosità di image[idx] della finestra di indice bid
        atomicAdd( &(histograms[(bid*256)+image[idx]]) , 1 ) ;
    }
}


""")

# kernel clipping e calcolo delle cdf 

mod_cdf = SourceModule("""

__global__ void calculate_cdfs(unsigned int *histograms, int *cdfs, unsigned int clipLimit) {

    //un thread per finestra               
    unsigned idx = threadIdx.x + blockIdx.x*blockDim.x;                   

    //clipping degli istogrammi
    int excess = 0 ; 
     
    if ( clipLimit > 0 ) {
            for ( unsigned int i = 0 ; i < 256 ; i++ ) {
            unsigned int excess = histograms[(idx*256)+i] - clipLimit ;
            if (excess > 0) { excess+=excess; }
        }    

        unsigned int binIncr = (int)(excess / 256) ;
        unsigned int upper = clipLimit - binIncr ;
                       
        //distribuzione degli eccessi
        for ( unsigned int i = 0 ; i < 256 ; i++ ) {
            if (histograms[(idx*256)+i] > clipLimit) { 
                histograms[(idx*256)+i] = clipLimit; 
            } else {
                excess -= binIncr ;
                histograms[(idx*256)+i] += binIncr ;
            }
        }  

        if ( excess > 0 ) {
            //ridistribuisco l'eccesso
            unsigned int stepSz = 1 + (int)(excess / 256) ;
            for ( unsigned int i = 0 ; i < 256 && excess > 0; i++ ) {
                excess -= stepSz ; 
                histograms[(idx*256)+i] += stepSz;
            }                   
        }
    }   
                       
    //calcolo della cdf
    cdfs[idx*256] = histograms[idx*256] ;            
    
    for ( unsigned int i = 1 ; i < 256 ; i++ ) {
        cdfs[(idx*256)+i] = histograms[(idx*256)+ i ] + cdfs[(idx*256)+ i - 1] ;
    }
}
        
""")

# kernel applicazione cdf

mod_apply = SourceModule("""
                   
__global__ void apply_cdfs(unsigned char *image, unsigned char *outputImage, int *cdfs, unsigned int w, unsigned int h) {   

                   
    //ogni blocco corrisponde ad una finestra, un thread all'interno del blocco per un pixel della finestra
                               
    //indice del blocco
    unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x ;
                   
    //indice all'interno del blocco
    unsigned int lid = threadIdx.y * blockDim.x + threadIdx.x ;
                   
    //indice globale
    // unsigned int tid = bid * (blockDim.x * blockDim.y) + bid ;
                   
    //indice di scorrimento dell'immagine
    unsigned int idx = blockIdx.y * ((blockDim.y * blockDim.x) * gridDim.x ) + threadIdx.y * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x ;
                                         
    
    if ( idx < w*h ){
        int pixelValue = image[idx];
        int equalizedPixel = (cdfs[(idx*256)+pixelValue] - cdfs[idx*256]) * 255 / ((blockDim.x * blockDim.y) - cdfs[idx*256]);
        outputImage[idx] = static_cast<unsigned char>(equalizedPixel);
    }
}
""")

# ------------------------------------ #


def histogram_equalization_cuda(input_image):

    # Ottieni le dimensioni dell'immagine
    height, width = input_image.shape

    blockDimX = 32
    blockDimY = 16

    gridDimX = width
    gridDimY = height 

    nrBlocks = gridDimX * gridDimY

    clipLimit = 40

    print(blockDimX)
    print(blockDimY)
    print(gridDimX)
    print(gridDimY)
    print(nrBlocks)

    # Allocazione della memoria sulla GPU
    d_image = cuda.mem_alloc(input_image.nbytes)
    d_output_image = cuda.mem_alloc(input_image.nbytes)
    d_histograms = cuda.mem_alloc(256 * np.int32().nbytes * nrBlocks)
    d_cdfs = cuda.mem_alloc(256 * np.int32().nbytes * nrBlocks)

    # Inizializzazione dell'istogramma a zero
    cuda.memset_d8(d_histograms, 0, 256 * np.int32().nbytes * nrBlocks)
    

    # Copia l'immagine sulla GPU
    cuda.memcpy_htod(d_image, input_image)

    # lancio del kernel per calcolare gli istogrammi
    starttime = time.time()
    calculate_histogram = mod_hi.get_function("calculate_histograms")
    calculate_histogram(d_image, d_histograms, np.int32(width), np.int32(height), block=(blockDimX, blockDimY, 1), grid=(width, height))
    endtime = time.time()
    elapsed = endtime - starttime
    print(elapsed)

    # Copia hist dalla GPU alla CPU per la visualizzazione
    hist = np.zeros(256*nrBlocks, dtype=np.int32)
    cuda.memcpy_dtoh(hist, d_histograms)
    print(hist)



    # Calcolo della cdf

    nrThreads = 512 # ogni SN può contenere 512x3 e ogni blocco 1024 -> occupancy al massimo
    blockSplit = (width * height) // 512

    calculate_cdfs = mod_cdf.get_function("calculate_cdfs")
    starttime1 = time.time()
    calculate_cdfs(d_histograms, d_cdfs, np.int32(clipLimit), block=(nrThreads, 1, 1), grid=(blockSplit, 1))
    endtime1 = time.time()
    elapsed1 = endtime1 - starttime1
    print(elapsed1)



    # Copia la cdf dalla GPU alla CPU per la visualizzazione
    cdf = np.zeros(256*nrBlocks, dtype=np.int32)
    cuda.memcpy_dtoh(cdf, d_cdfs)
    print(cdf)

    # Kernel per applicare l'equalizzazione dell'istogramma
    starttime = time.time()
    apply_cdfs = mod_apply.get_function("apply_cdfs")
    apply_cdfs(d_image, d_output_image, d_cdfs, np.int32(width), np.int32(height), block=(blockDimX, blockDimY, 1), grid=(width//blockDimX, height//blockDimY))
    endtime = time.time()
    elapsed = endtime - starttime
    print(elapsed)

    # Copia l'immagine equalizzata dalla GPU alla CPU
    output_image = np.empty_like(input_image)
    cuda.memcpy_dtoh(output_image, d_output_image)

    # Rilascia la memoria sulla GPU
    d_image.free()
    d_output_image.free()
    d_histograms.free()
    d_cdfs.free()

    #necessario???? 
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

        enhanced_gray = histogram_equalization_cuda(gray)

        rects = detector(enhanced_gray, 0)

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
        cv2.imshow("CUDA face detection", enhanced_gray)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
