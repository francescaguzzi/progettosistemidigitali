#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


// COMPILARE FORMATO PTX
// nvcc -ptx cuda_kernels.cu -o cuda_kernels.ptx


/* --------------------------------------------------- */

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

/* --------------------------------------------------- */


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


/* --------------------------------------------------- */


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


/* --------------------------------------------------- */