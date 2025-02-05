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
                   
__global__ void calculate_histograms(unsigned char const *image, unsigned int *histograms, unsigned int const w, unsigned int const h) {   

                   
    //ogni blocco corrisponde ad un pixel, un thread all'interno del blocco per un pixel della finestra
                               
    //indice del blocco
    unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x ;
                      
    //indice all'interno della finestra
    unsigned int lid = threadIdx.y * blockDim.x + threadIdx.x ;
                   
    //indice all'interno della finestra
    //unsigned int bid = threadIdx.y * blockDim.x + threadIdx.x ;
                   
    //indice globale
    // unsigned int tid = bid * (blockDim.x * blockDim.y) + bid ;
                      
    __shared__ unsigned int s_hist[256];
                      
    for (unsigned int i = lid ; i<256 ; i += blockDim.x*blockDim.y) {
        s_hist[i] = 0u ;                  
    } 
                      
    __syncthreads();
          
                      
    // Outer loop iterazione i = 0

    unsigned int idx0 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 0 * blockDim.x + 0 * gridDim.x * blockDim.y;
    unsigned int idx1 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 0 * blockDim.x + 1 * gridDim.x * blockDim.y;
    unsigned int idx2 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 0 * blockDim.x + 2 * gridDim.x * blockDim.y;
    unsigned int idx3 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 0 * blockDim.x + 3 * gridDim.x * blockDim.y;

    if (idx0 < w * h && idx0 > 0) atomicAdd(&(s_hist[image[idx0]]), 1);
    if (idx1 < w * h && idx1 > 0) atomicAdd(&(s_hist[image[idx1]]), 1);
    if (idx2 < w * h && idx2 > 0) atomicAdd(&(s_hist[image[idx2]]), 1);
    if (idx3 < w * h && idx3 > 0) atomicAdd(&(s_hist[image[idx3]]), 1);


    // Outer loop iterazione i = 1

    idx0 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 1 * blockDim.x + 0 * gridDim.x * blockDim.y;
    idx1 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 1 * blockDim.x + 1 * gridDim.x * blockDim.y;
    idx2 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 1 * blockDim.x + 2 * gridDim.x * blockDim.y;
    idx3 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 1 * blockDim.x + 3 * gridDim.x * blockDim.y;

    if (idx0 < w * h && idx0 > 0) atomicAdd(&(s_hist[image[idx0]]), 1);
    if (idx1 < w * h && idx1 > 0) atomicAdd(&(s_hist[image[idx1]]), 1);
    if (idx2 < w * h && idx2 > 0) atomicAdd(&(s_hist[image[idx2]]), 1);
    if (idx3 < w * h && idx3 > 0) atomicAdd(&(s_hist[image[idx3]]), 1);


    // Outer loop iterazione = 2

    idx0 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 2 * blockDim.x + 0 * gridDim.x * blockDim.y;
    idx1 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 2 * blockDim.x + 1 * gridDim.x * blockDim.y;
    idx2 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 2 * blockDim.x + 2 * gridDim.x * blockDim.y;
    idx3 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 2 * blockDim.x + 3 * gridDim.x * blockDim.y;

    if (idx0 < w * h && idx0 > 0) atomicAdd(&(s_hist[image[idx0]]), 1);
    if (idx1 < w * h && idx1 > 0) atomicAdd(&(s_hist[image[idx1]]), 1);
    if (idx2 < w * h && idx2 > 0) atomicAdd(&(s_hist[image[idx2]]), 1);
    if (idx3 < w * h && idx3 > 0) atomicAdd(&(s_hist[image[idx3]]), 1);
    

    // Outer loop iterazione i = 3
    idx0 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 3 * blockDim.x + 0 * gridDim.x * blockDim.y;
    idx1 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 3 * blockDim.x + 1 * gridDim.x * blockDim.y;
    idx2 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 3 * blockDim.x + 2 * gridDim.x * blockDim.y;
    idx3 = bid + (threadIdx.x - 16) + (threadIdx.y - 16) * gridDim.x + 3 * blockDim.x + 3 * gridDim.x * blockDim.y;

    if (idx0 < w * h && idx0 > 0) atomicAdd(&(s_hist[image[idx0]]), 1);
    if (idx1 < w * h && idx1 > 0) atomicAdd(&(s_hist[image[idx1]]), 1);
    if (idx2 < w * h && idx2 > 0) atomicAdd(&(s_hist[image[idx2]]), 1);
    if (idx3 < w * h && idx3 > 0) atomicAdd(&(s_hist[image[idx3]]), 1);
                      
    __syncthreads();
                      
    for ( unsigned int i = lid ; i < 256 ; i += blockDim.x*blockDim.y) {
        histograms[i + (bid<<8)]=s_hist[i];
    }
    
}
""")

# kernel clipping e calcolo delle cdf 

mod_cdf = SourceModule("""

__global__ void apply_clipping(unsigned int *histograms, unsigned int const clipLimit) {

    //un thread per livello, un blocco per pixel
                       
    //indice del blocco
    unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x ;
                       
    //indice del livello
    unsigned int hid = (bid<<8) + threadIdx.x;

    //clipping degli istogrammi
    __shared__ int nrExcess; 
                       
    if (threadIdx.x == 0) { nrExcess = 0u; }
     
    if ( clipLimit > 0 ) {
        //calcolo degli eccessi              
        int excess = histograms[hid]-clipLimit ;
        if(excess > 0) { atomicAdd(&(nrExcess),excess); } 

        __syncthreads();  
                       
        //distribuzione degli eccessi
        unsigned int binIncr = (int)(nrExcess / 256);
        unsigned int upper = clipLimit - binIncr;

        __syncthreads();
        if (histograms[hid] > clipLimit) { 
            histograms[hid] = clipLimit; 
        } else if (histograms[hid] > upper) {
            atomicSub(&(nrExcess),histograms[hid] - upper);
            histograms[hid] = clipLimit;              
        } else {
            atomicSub(&(nrExcess),binIncr);
            histograms[hid] += binIncr ;
        }
                       
        __syncthreads();
                       
        //eventuale ridistribuzione sequenziale, basso carico di lavoro
            if ( nrExcess > 0 ) {
                //ridistribuisco l'eccesso
                unsigned int stepSz = (int)(nrExcess >> 8) ;
                histograms[hid] += stepSz;                  
            }               
    }

    //uso della scan di Kogge-Stone

    // stride = 1
    if (threadIdx.x >= 1) {
        __syncthreads();
        histograms[hid] += histograms[hid - 1];
    }

    // stride = 2
    __syncthreads();
    if (threadIdx.x >= 2) {
        __syncthreads();
        histograms[hid] += histograms[hid - 2];
    }

    // stride = 4
    if (threadIdx.x >= 4) {
        __syncthreads();
        histograms[hid] += histograms[hid - 4];
    }

    // stride = 8
    if (threadIdx.x >= 8) {
        __syncthreads();
        histograms[hid] += histograms[hid - 8];
    }

    // stride = 16
    if (threadIdx.x >= 16) {
        __syncthreads();
        histograms[hid] += histograms[hid - 16];
    }

    // stride = 32
    if (threadIdx.x >= 32) {
        __syncthreads();
        histograms[hid] += histograms[hid - 32];
    }

    // stride = 64
    if (threadIdx.x >= 64) {
        __syncthreads();
        histograms[hid] += histograms[hid - 64];
    }

    // stride = 128
    if (threadIdx.x >= 128) {
        __syncthreads();
        histograms[hid] += histograms[hid - 128];
    }                                             
}
        
""")

# kernel applicazione cdf

mod_apply = SourceModule("""
                   
__global__ void apply_cdfs(unsigned char const *image, unsigned char *outputImage, int const *cdfs, unsigned int const w, unsigned int const h) {   

                   
    //ogni blocco corrisponde ad una finestra, un thread all'interno del blocco per un pixel della finestra
                               
    //indice del blocco
    unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x ;
                   
    //indice di scorrimento dell'immagine
    unsigned int idx = blockIdx.y * ((blockDim.y * blockDim.x) * gridDim.x ) + threadIdx.y * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x ;
                                         
    
    if ( idx < w*h ){
        int equalizedPixel = (cdfs[(idx<<8)+image[idx]] - cdfs[idx<<8]) * 255 / (1024 - cdfs[idx<<8]);
        outputImage[idx] = static_cast<unsigned char>(equalizedPixel);
    }
}
""")

# ------------------------------------ #


def histogram_equalization_cuda(input_image):

    height, width = input_image.shape
    image_np = np.array(input_image,dtype=np.uint8)

    blockDimX = 8
    blockDimY = 8

    gridDimX = width
    gridDimY = height 
    nrBlocks = gridDimX * gridDimY

    clipLimit = 20

    # Allocazione della memoria sulla GPU
    d_image = cuda.mem_alloc(image_np.nbytes)
    d_output_image = cuda.mem_alloc(image_np.nbytes)
    d_histograms = cuda.mem_alloc(256 * np.int32().nbytes * nrBlocks)
    d_cdfs = cuda.mem_alloc(256 * np.int32().nbytes * nrBlocks)

    # Inizializzazione dell'istogramma a zero
    cuda.memset_d8(d_histograms, 0, 256 * np.int32().nbytes * nrBlocks)

    # Copia l'immagine sulla GPU
    cuda.memcpy_htod(d_image, image_np)

    # lancio del kernel per calcolare gli istogrammi
    calculate_histogram = mod_hi.get_function("calculate_histograms")
    calculate_histogram(d_image, d_histograms, np.int32(width), np.int32(height), block=(blockDimX, blockDimY, 1), grid=(width, height))

    # lancio del kernel per il calcolo della cdf e clipping
    apply_clipping = mod_clp.get_function("apply_clipping")
    apply_clipping(d_histograms, np.int32(clipLimit), block=(256, 1, 1), grid=(width, height))
   
    # kernel per applicare l'equalizzazione dell'istogramma
    apply_cdfs = mod_apply.get_function("apply_cdfs")
    apply_cdfs(d_image, d_output_image, d_histograms, np.int32(width), np.int32(height), block=(blockDimX, blockDimY, 1), grid=(width//blockDimX, height//blockDimY))

    # copia l'immagine equalizzata dalla GPU alla CPU
    output_image = np.empty_like(image_np)
    cuda.memcpy_dtoh(output_image, d_output_image)

    # rilascia la memoria sulla GPU
    d_image.free()
    d_output_image.free()
    d_histograms.free()
    d_cdfs.free()
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
        # cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, f"Orientamento: {orientation_value}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # cv2.imshow("Frame", frame)
        cv2.imshow("CUDA face detection", enhanced_gray)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
