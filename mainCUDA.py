import os
import pygame
# import sys
from multiprocessing import Process, Value, Lock, Manager
import time
import psutil
import random 

# sys.path.append("game/")
from face_detectorCUDA import facial_recognition

# ---------------------------- #

def generate_performance_report(data):

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"performance_report_{timestamp}.txt"

    with open(filename, "w") as file:
        file.write("Performance Report\n")
        file.write("="*30 + "\n")
        file.write(f"Tempo totale di esecuzione: {data['total_time']:.2f} secondi\n")
        file.write(f"Media degli FPS: {data['average_fps']:.2f}\n")
        file.write(f"Media consumo GPU: {data['cpu_usage']:.2f}%\n")
        file.write(f"Media memoria utilizzata: {data['memory_usage']:.2f} MB\n")
        # file.write(f"Tempo processo di gioco: {data['game_time']:.2f} secondi\n")
        # file.write(f"Tempo processo di rilevamento facciale: {data['recognition_time']:.2f} secondi\n")

def monitor_performance(start_time, game_process, recognition_process, stop_flag, shared_data):
    
    fps_list = []
    cpu_usage = []
    memory_usage = []

    while not stop_flag.value:
        # calcolo FPS approssimato
        elapsed_time = time.time() - start_time.value
        if elapsed_time > 0:
            fps_list.append(1 / elapsed_time)

        cpu_usage.append(psutil.cpu_percent())
        memory_usage.append(psutil.virtual_memory().used / (1024 ** 2))  # Converti in MB

        time.sleep(1)  # monitora ogni secondo

    shared_data["total_time"] = time.time() - start_time.value
    shared_data["average_fps"] = sum(fps_list) / len(fps_list) if fps_list else 0
    shared_data["cpu_usage"] = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
    shared_data["memory_usage"] = sum(memory_usage) / len(memory_usage) if memory_usage else 0
    # shared_data["game_time"] = game_process.exitcode if game_process.exitcode is not None else 0
    # shared_data["recognition_time"] = recognition_process.exitcode if recognition_process.exitcode is not None else 0


# ---------------------------- #




def load_frames(folder_path, screen_height):
    frames = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".png"):  # Controlla solo i file PNG
            frame = pygame.image.load(os.path.join(folder_path, file_name))
            # Ottieni le dimensioni originali del frame
            original_width, original_height = frame.get_size()
            # Calcola il rapporto di scala basato sull'altezza
            scale_ratio = screen_height / original_height
            scaled_width = int(original_width * scale_ratio)
            scaled_height = screen_height  # Altezza uguale allo schermo
            # Ridimensiona mantenendo le proporzioni
            frame = pygame.transform.scale(frame, (scaled_width, scaled_height))
            frames.append(frame)
    return frames

def load_random_frames(screen_height):

    # prendo tutte le sottocartelle della directory
    base_path = "frames"
    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

    random_folder = random.choice(subfolders)

    print(f"Carico i frame dalla cartella: {random_folder}")
    return load_frames(random_folder, screen_height)

# Script Pygame
def game(orientation, blink, lock):

    pygame.init()
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    running = True

    # Stato iniziale
    light_off = False  
    frames = load_frames("frames/default", SCREEN_HEIGHT)
    current_frame = 0
    frame_delay = 10
    frame_count = 0

    move_speed = 5
    camera_x = -(frames[0].get_width() // 2 - SCREEN_WIDTH // 2)

    min_camera_x = 0
    max_camera_x = frames[0].get_width() - SCREEN_WIDTH

    overlay_image = pygame.image.load("images/overlay.png")
    overlay_original_width, overlay_original_height = overlay_image.get_size()
    overlay_scale_ratio = SCREEN_HEIGHT / overlay_original_height
    overlay_scaled_width = int(overlay_original_width * overlay_scale_ratio)
    overlay_scaled_height = SCREEN_HEIGHT
    overlay_image = pygame.transform.scale(overlay_image, (overlay_scaled_width, overlay_scaled_height))
    overlay_position = (SCREEN_WIDTH // 2 - overlay_image.get_width() // 2, 
                        SCREEN_HEIGHT // 2 - overlay_image.get_height() // 2)

    while running:

        frame_count += 1
        if frame_count >= frame_delay:
            current_frame = (current_frame + 1) % len(frames)  # Passa al frame successivo
            frame_count = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Usa il lock per accedere alla memoria condivisa
        with lock:
            orientation_value = orientation.value 
            blink_value = blink.value

        # Muove la telecamera in base all'orientamento
        if orientation_value == 1 and camera_x < max_camera_x: # DESTRA
            camera_x += move_speed
        elif orientation_value == 2 and camera_x > min_camera_x: # SINISTRA
            camera_x -= move_speed
        elif orientation_value == 0: # CENTRO
            pass

        # Spegne/accende la luce quando sbatte le palpebre
        if blink_value:
            light_off = not light_off

            if not light_off:
                frames = load_random_frames(SCREEN_HEIGHT)
                current_frame = 0

        if light_off:
            screen.fill((0,0,0)) 
        else: 
            # Calcola lo spostamento orizzontale per centrare il frame
            frame_width = frames[current_frame].get_width()
            max_offset_x = frame_width - SCREEN_WIDTH  # Limite massimo dello scorrimento

            # Limita lo spostamento ai bordi dell'immagine
            if camera_x < 0:
                camera_x = 0
            elif camera_x > max_offset_x:
                camera_x = max_offset_x

            # Renderizza il frame
            screen.blit(frames[current_frame], (-camera_x, 0))
            screen.blit(overlay_image, overlay_position)

        # Riempie lo schermo con colore diverso a seconda della luce
        pygame.display.flip()
        pygame.time.delay(30)

    pygame.quit()



if __name__ == "__main__":

    try: 
    
        # Creazione della memoria condivisa per l'orientamento e lo stato delle palpebre
        orientation = Value('i', 0)  # intero: 0 centro, 1 destra, 2 sinistra
        blink = Value('b', False)  # 'b' è per booleani
        lock = Lock()

        # flag per interrompere il benchmark
        stop_flag = Value('b', False)
        # variabile per il tempo di avvio
        start_time = Value('d', time.time())

        # dati condivisi per le performance
        manager = Manager()
        shared_data = manager.dict()

        # avvio dei processi
        recognition_process = Process(target=facial_recognition, args=(orientation, blink, lock))
        game_process = Process(target=game, args=(orientation, blink, lock))
        monitor_process = Process(target=monitor_performance, args=(start_time, game_process, recognition_process, stop_flag, shared_data))
        
        recognition_process.start()
        game_process.start()
        monitor_process.start()
        
        # recognition_process.join()
        # game_process.join()

        print("Premi 'q' seguito da INVIO per uscire dal gioco.")

        # Monitora l'input per l'interruzione
        while True:
            user_input = input()  
            if user_input.lower() == 'q':  
                print("Chiusura dell'applicazione...")
                break

        # Termina i processi
        stop_flag.value = True
        recognition_process.terminate()
        game_process.terminate()
        monitor_process.join()

        # genera il report
        generate_performance_report(shared_data)
        print("Report delle prestazioni generato")

    except KeyboardInterrupt:

        # Gestione di Ctrl+C per sicurezza
        print("\nInterruzione manuale rilevata. Arresto dei processi...")
        stop_flag.value = True
        recognition_process.terminate()
        game_process.terminate()
        monitor_process.join()
        generate_performance_report(shared_data)
        print("Report delle prestazioni generato. Uscita.")
