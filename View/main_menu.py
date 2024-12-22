import pygame
import sys
import subprocess
import time
import os
from animasi.button import Button

pygame.init()

# Menentukan ukuran layar
SCREEN_WIDTH, SCREEN_HEIGHT = 1800, 900
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Menu")

# Memuat background dan menyesuaikan ukurannya
BG = pygame.image.load("../Assets/image/Backgroud_menu.jpg")
BG = pygame.transform.scale(BG, (SCREEN_WIDTH, SCREEN_HEIGHT))  # Menyesuaikan ukuran background

def get_font(size):  # Returns Press-Start-2P in the desired size
    return pygame.font.Font("../Assets/font/font.ttf", size)

# Loading Screen Function
def loading_screen():
    # Colors for the progress bar
    bar_color = (0, 255, 255)
    bar_background_color = (50, 50, 50)
    text_color = (255, 255, 255)

    # Font
    font = pygame.font.Font(None, 60)

    # Loading progress simulation
    progress = 0
    bar_width = 400
    bar_height = 50
    bar_x = (SCREEN_WIDTH - bar_width) // 2
    bar_y = SCREEN_HEIGHT // 2 - bar_height // 2

    while progress < 1:
        SCREEN.fill("black")
        
        # Draw progress bar
        pygame.draw.rect(SCREEN, bar_background_color, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(SCREEN, bar_color, (bar_x, bar_y, bar_width * progress, bar_height))
        
        # Draw loading text
        loading_text = font.render(f"Loading... {int(progress * 100)}%", True, text_color)
        text_rect = loading_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100))
        SCREEN.blit(loading_text, text_rect)

        progress += 0.01  # Increase progress
        time.sleep(0.05)  # Simulate loading delay
        
        pygame.display.update()

    time.sleep(1)  # Optional delay to let the user see the "loading complete" state before proceeding

# Fungsi untuk main menu
def main_menu():
    """Fungsi untuk tampilan utama menu"""
    # Memuat suara
    hover_sound = pygame.mixer.Sound("../Assets/sound/sound_button.mp3")
    click_sound = pygame.mixer.Sound("../Assets/sound/sound_click.mp3")

    # Menentukan jarak antar tombol
    button_spacing = 150  # Jarak antar tombol dalam piksel

    while True:
        SCREEN.blit(BG, (0, 0))  # Menampilkan background yang sudah disesuaikan ukuran layar

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_font(100).render("MAIN MENU", True, "#b68f40")
        MENU_RECT = MENU_TEXT.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))  # Posisikan teks di tengah
        SCREEN.blit(MENU_TEXT, MENU_RECT)

        # Mengatur posisi tombol dengan jarak antar tombol
        PLAY_PLAYER_VS_PLAYER_BUTTON = Button(image=pygame.image.load("../Assets/image/Options Rect.png"), 
                             pos=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2.5),
                             text_input="PLAYER VS PLAYER", font=get_font(25), base_color="#d7fcd4", hovering_color="White", hover_sound=hover_sound, click_sound=click_sound)

        PLAY_PLAYER_VS_BOT_BUTTON = Button(image=pygame.image.load("../Assets/image/Options Rect.png"), 
                                    pos=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2.5 + button_spacing),
                                    text_input="PLAYER VS BOT", font=get_font(25), base_color="#d7fcd4", hovering_color="White", hover_sound=hover_sound, click_sound=click_sound)

        OPTIONS_BUTTON = Button(image=pygame.image.load("../Assets/image/Options Rect.png"), 
                                pos=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2.5 + 2 * button_spacing),
                                text_input="OPTIONS", font=get_font(25), base_color="#d7fcd4", hovering_color="White", click_sound=click_sound)

        QUIT_BUTTON = Button(image=pygame.image.load("../Assets/image/Options Rect.png"), 
                            pos=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2.5 + 3 * button_spacing),
                            text_input="QUIT", font=get_font(25), base_color="#d7fcd4", hovering_color="White", click_sound=click_sound)
        
        
        # Update tombol sesuai posisi mouse
        for button in [PLAY_PLAYER_VS_BOT_BUTTON, PLAY_PLAYER_VS_PLAYER_BUTTON, OPTIONS_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)  # Ubah warna tombol saat hover
            button.update(SCREEN)  # Gambar tombol
        

        # Menangani event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_PLAYER_VS_PLAYER_BUTTON.checkForInput(MENU_MOUSE_POS):
                    click_sound.play()  # Putar suara klik
                    loading_screen()  # Pindah ke layar loading
                    play()
                if PLAY_PLAYER_VS_BOT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    click_sound.play()  # Putar suara klik
                    loading_screen()  # Pindah ke layar loading
                    pygame.quit()
                    subprocess.run(["python3", "../Controller/player_vs_bot_cont.py"])
                if OPTIONS_BUTTON.checkForInput(MENU_MOUSE_POS):
                    click_sound.play()  # Putar suara klik
                    options()
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    click_sound.play()  # Putar suara klik
                    pygame.quit()
                    sys.exit()

        pygame.display.update()

def play():
    while True:
        PLAY_MOUSE_POS = pygame.mouse.get_pos()

        SCREEN.fill("black")

        PLAY_TEXT = get_font(45).render("This is the PLAY screen.", True, "White")
        PLAY_RECT = PLAY_TEXT.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
        SCREEN.blit(PLAY_TEXT, PLAY_RECT)

        PLAY_BACK = Button(image=None, pos=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 1.75),
                           text_input="BACK", font=get_font(75), base_color="White", hovering_color="Green")

        PLAY_BACK.changeColor(PLAY_MOUSE_POS)
        PLAY_BACK.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BACK.checkForInput(PLAY_MOUSE_POS):
                    main_menu()

        pygame.display.update()

def options():
    while True:
        OPTIONS_MOUSE_POS = pygame.mouse.get_pos()

        SCREEN.fill("white")

        OPTIONS_TEXT = get_font(45).render("This is the OPTIONS screen.", True, "Black")
        OPTIONS_RECT = OPTIONS_TEXT.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
        SCREEN.blit(OPTIONS_TEXT, OPTIONS_RECT)

        OPTIONS_BACK = Button(image=None, pos=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 1.75),
                              text_input="BACK", font=get_font(75), base_color="Black", hovering_color="Green")

        OPTIONS_BACK.changeColor(OPTIONS_MOUSE_POS)
        OPTIONS_BACK.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if OPTIONS_BACK.checkForInput(OPTIONS_MOUSE_POS):
                    main_menu()

        pygame.display.update()

# Start the loading screen before main menu
loading_screen()
main_menu()
