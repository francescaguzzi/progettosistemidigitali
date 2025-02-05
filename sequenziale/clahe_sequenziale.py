import numpy as np
import time
import cv2


# ------------------------------------------ #

def CLAHE(image, clip_limit=20, window_size=(32, 32)):
    """
    Applica il CLAHE (Contrast Limited Adaptive Histogram Equalization) calcolando,
    per ogni pixel, l'istogramma locale basato su un intorno di dimensione window_size (default 32x32).
    """

    win_h, win_w = window_size
    pad_h = win_h // 2
    pad_w = win_w // 2

    # esegue il padding sull'immagine per gestire i bordi (replicando il valore dei bordi)
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    height, width = image.shape
    output_image = np.zeros_like(image)

    # print("Inizio\n")
    start_time = time.time()

    # per ogni pixel dell'immagine 
    for i in range(height):
        for j in range(width):

            # estrae l'intorno locale 32x32 corrispondente
            local_window = padded_image[i:i + win_h, j:j + win_w]

            # calcola l'istogramma locale (256 bin per valori 0-255)
            hist, _ = np.histogram(local_window, bins=256, range=(0, 256))
            hist = hist.astype(np.uint32)

            histogram_end = time.time()

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
            for k in range(1, 256):
                cdf[k] = cdf[k - 1] + hist[k]

            clipping_cdf_end = time.time()
            
            # normalizza la CDF per mappare i valori in [0,255]
            cdf_min = cdf.min()
            cdf_max = cdf.max()
            if cdf_max != cdf_min:
                cdf_normalized = np.empty(256, dtype=np.uint8)
                for k in range(1, 256):
                    cdf_normalized[k] = np.clip(round((cdf[k] - cdf_min) * 255 / (cdf_max - cdf_min)), 0, 255)
            else:
                cdf_normalized = np.zeros(256, dtype=np.uint8)

            # mappa il pixel corrente usando il valore corrispondente nella Look-Up Table
            pixel_val = image[i, j]
            # new_val = np.clip(round(cdf_normalized[pixel_val]), 0, 255)
            output_image[i, j] = cdf_normalized[pixel_val]

            apply_cdf_end = time.time()

    end_time = time.time()

    elapsed_time = end_time - start_time
    print("\nFunzione eseguita in {elapsed_time} secondi\n")
    print(elapsed_time)

    hist_time = histogram_end - start_time
    print("\nFunzione istogrammi: {hist_time} secondi\n")
    print(hist_time)

    clipping_time = clipping_cdf_end - histogram_end
    print("\nFunzione clipping: {clipping_time} secondi\n")
    print(clipping_time)

    equalization_time = apply_cdf_end - clipping_cdf_end
    print("\nFunzione equalizzazione: {equalization_time} secondi\n")
    print(equalization_time)

    return output_image

# ------------------------------------------ #

def main():
    # carica l'immagine 
    image = cv2.imread('input-image.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Errore nel caricamento dell'immagine.")
    else:
        enhanced_gray = CLAHE(image)

    cv2.imshow("CLAHE Equalization", enhanced_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0 

if __name__ == "__main__":
    main()
