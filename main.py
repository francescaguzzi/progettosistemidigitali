import pygame
# import sys
from multiprocessing import Process, Queue
import time

# sys.path.append("game/")
from face_detector import facial_recognition

# Script Pygame
def game(queue):
    pygame.init()
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    running = True
    camera_x = 0  # Posizione della "telecamera"
    light_on = True  # Stato della luce

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Ricevi eventi dalla coda
        if not queue.empty():
            data = queue.get()
            orientation = data["orientation"]
            blink = data["blink"]

            # Muove la telecamera in base all'orientamento
            if orientation == "DESTRA":
                camera_x -= 3
            elif orientation == "SINISTRA":
                camera_x += 3

            # Spegne/accende la luce quando sbatte le palpebre
            if blink:
                light_on = not light_on

        # Riempie lo schermo con colore diverso a seconda della luce
        screen.fill((255, 255, 255) if light_on else (0, 0, 0))

        # Qui puoi aggiungere logica per disegnare oggetti in base a `camera_x`
        pygame.draw.circle(screen, (0, 128, 255), (400 + camera_x, 300), 50)  # Esempio di oggetto

        pygame.display.flip()
        pygame.time.delay(30)

    pygame.quit()

if __name__ == "__main__":
    queue = Queue()
    recognition_process = Process(target=facial_recognition, args=(queue,))
    game_process = Process(target=game, args=(queue,))
    recognition_process.start()
    game_process.start()
    recognition_process.join()
    game_process.join()
