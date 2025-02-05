#!/usr/bin/env python3

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
import cv2

# ------------------------------------------ #

# VERSIONE 2 - USO DELLA SHARED MEMORY

# ------------------------------------------ #

#kernel calcolo degli istogrammi

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
    
                   
                      
    for (unsigned int i = 0 ; i<32/blockDim.x ; i++) {
        for(unsigned int j = 0 ; j<32/blockDim.y ; j++) {
            //indice di scorrimento dell'immagine
            unsigned int idx = bid + (threadIdx.x-((int)32/2)) + (threadIdx.y-(32/2)) * gridDim.x + i * blockDim.x + j * gridDim.x * blockDim.y;
            if ( idx < w*h  && idx > 0){
                //viene aggiornato il counter per il livello di luminosità di image[idx] della finestra di indice bid
                atomicAdd( &(s_hist[(image[idx])]) , 1 ) ;
            }             
        }
    }
                      
    __syncthreads();
                      
    for ( unsigned int i = lid ; i < 256 ; i += blockDim.x*blockDim.y) {
        histograms[i + (bid<<8)]=s_hist[i];
    }
    
}


""")

# ------------------------------------------ #

#kernel clipping e calcolo delle cdf 

mod_clp = SourceModule("""

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

    for ( unsigned int stride = 1 ; stride < blockDim.x ; stride <<= 1 ) {
        __syncthreads();
        if ( threadIdx.x >= stride ) {
            histograms[hid] += histograms[hid - stride];
        }
    }                                             
}

""")

# ------------------------------------------ #

#kernel applicazione cdf

mod_apply = SourceModule("""
                   
__global__ void apply_cdfs(unsigned char const *image, unsigned char *outputImage, int const *cdfs, unsigned int const w, unsigned int const h) {   

                   
    //ogni blocco corrisponde ad una finestra, un thread all'interno del blocco per un pixel della finestra
                               
    //indice del blocco
    unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x ;
                   
    //indice di scorrimento dell'immagine
    unsigned int idx = blockIdx.y * ((blockDim.y * blockDim.x) * gridDim.x ) + threadIdx.y * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x ;                                     
    
    int equalizedPixel = (cdfs[(idx<<8)+image[idx]] - cdfs[idx<<8]) * 255 / (1024 - cdfs[idx<<8]);
    outputImage[idx] = static_cast<unsigned char>(equalizedPixel);
    
}
""")

# ------------------------------------------ #

def histogram_equalization_cuda(input_image):

    height, width = input_image.shape
    image_np = np.array(input_image,dtype=np.uint8)

    blockDimX = 8
    blockDimY = 8

    gridDimX = width
    gridDimY = height 

    nrBlocks = gridDimX * gridDimY

    clipLimit = 20

    # allocazione della memoria sulla GPU
    d_image = cuda.mem_alloc(image_np.nbytes)
    d_output_image = cuda.mem_alloc(image_np.nbytes)
    d_histograms = cuda.mem_alloc(256 * np.int32().nbytes * nrBlocks)
    d_cdfs = cuda.mem_alloc(256 * np.int32().nbytes * nrBlocks)

    # inizializzazione dell'istogramma a zero
    cuda.memset_d8(d_histograms, 0, 256 * np.int32().nbytes * nrBlocks)
    
    # copia l'immagine sulla GPU
    cuda.memcpy_htod(d_image, image_np)

    # lancio del kernel per calcolare gli istogrammi
    calculate_histogram = mod_hi.get_function("calculate_histograms")
    calculate_histogram(d_image, d_histograms, np.int32(width), np.int32(height), block=(blockDimX, blockDimY, 1), grid=(width, height))

    # kernel per il clipping e calcolo della cdf
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

# ------------------------------------------ #

def main() :

    image = cv2.imread('input-image.jpg', cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Errore nel caricamento dell'immagine.")
    else:
        equalized_image = histogram_equalization_cuda(image)

    #cv2.imshow("Originale", image)
    cv2.imshow("Equalizzato", equalized_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main()