import cv2
import numpy as np

# Imposta valori iniziali per luminosità e contrasto
contrast = 1.9  # Aumenta o diminuisci per regolare il contrasto
brightness = 10  # Aumenta o diminuisci per regolare la luminosità

def adjust_contrast_brightness(image, contrast, brightness):

    midvalue = np.mean(image)

    # Applica la formula per ogni pixel
    new_image = contrast * (image - midvalue) + midvalue + brightness
    # Assicura che i valori siano tra 0 e 255
    new_image = np.clip(new_image, 0, 255)
    return new_image.astype(np.uint8)

# Avvia il feed della webcam
cap = cv2.VideoCapture(0)

while True:
    # Cattura il frame
    ret, frame = cap.read()
    if not ret:
        break

    # Converte il frame in scala di grigi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applica la regolazione di contrasto e luminosità
    adjusted = adjust_contrast_brightness(gray, contrast, brightness)

    # Mostra il risultato
    cv2.imshow("Original Grayscale", gray)
    cv2.imshow("Adjusted Contrast and Brightness", adjusted)

    # Premi 'q' per uscire dal ciclo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la videocamera e chiudi le finestre
cap.release()
cv2.destroyAllWindows()
