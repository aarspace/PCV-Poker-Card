import pygame
import sys
from pygame.locals import *
from button import Button

pygame.init()

# Menentukan ukuran layar
SCREEN_WIDTH, SCREEN_HEIGHT = 1800, 800
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Options")

# Memuat background dan menyesuaikan ukurannya
BG = pygame.image.load("/home/aarspace/Documents/PCV/Tugas Game/CNN/GUI/display/assets/Backgroud_menu.jpg")
BG = pygame.transform.scale(BG, (SCREEN_WIDTH, SCREEN_HEIGHT))  # Menyesuaikan ukuran background

def get_font(size):  # Returns Press-Start-2P in the desired size
    return pygame.font.Font("assets/font.ttf", size)

# Slider class untuk pengaturan suara
class Slider:
    def __init__(self, x, y, width, min_value, max_value, initial_value):
        self.rect = pygame.Rect(x, y, width, 20)  # Posisi dan ukuran slider
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.handle_rect = pygame.Rect(self.rect.x + self.value * (self.rect.width / (self.max_value - self.min_value)),
                                       self.rect.y, 20, 20)  # Posisi handle slider

    def update(self, mouse_pos, mouse_pressed):
        if self.rect.collidepoint(mouse_pos):
            if mouse_pressed:
                # Mengubah nilai slider berdasarkan posisi mouse
                new_value = (mouse_pos[0] - self.rect.x) / (self.rect.width / (self.max_value - self.min_value))
                self.value = min(max(new_value, self.min_value), self.max_value)
                self.handle_rect.x = self.rect.x + self.value * (self.rect.width / (self.max_value - self.min_value))

    def draw(self, screen):
        pygame.draw.rect(screen, (200, 200, 200), self.rect)  # Gambar background slider
        pygame.draw.rect(screen, (100, 100, 100), self.handle_rect)  # Gambar handle slider

# Dropdown class untuk pengaturan kamera
class Dropdown:
    def __init__(self, x, y, options, font, base_color):
        self.x = x
        self.y = y
        self.options = options
        self.font = font
        self.base_color = base_color
        self.selected_option = options[0]
        self.is_open = False
        self.rect = pygame.Rect(x, y, 200, 50)

    def update(self, mouse_pos, mouse_pressed):
        if self.rect.collidepoint(mouse_pos):
            if mouse_pressed:
                self.is_open = not self.is_open
        if self.is_open:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.x, self.y + (i + 1) * 50, 200, 50)
                if option_rect.collidepoint(mouse_pos):
                    if mouse_pressed:
                        self.selected_option = option
                        self.is_open = False

    def draw(self, screen):
        pygame.draw.rect(screen, (100, 100, 100), self.rect)  # Gambar dropdown
        option_text = self.font.render(self.selected_option, True, self.base_color)
        screen.blit(option_text, (self.x + 10, self.y + 10))

        if self.is_open:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.x, self.y + (i + 1) * 50, 200, 50)
                pygame.draw.rect(screen, (150, 150, 150), option_rect)
                option_text = self.font.render(option, True, self.base_color)
                screen.blit(option_text, (self.x + 10, self.y + (i + 1) * 50 + 10))

def options():
    """Halaman opsi dengan slider dan dropdown untuk pengaturan suara dan kamera"""
    # Membuat slider untuk volume suara
    slider = Slider(600, 300, 400, 0, 100, 50)
    # Membuat dropdown untuk pengaturan kamera
    camera_options = ["Default", "720p", "1080p", "4K"]
    dropdown = Dropdown(600, 400, camera_options, get_font(30), "Black")

    # Memuat suara
    hover_sound = pygame.mixer.Sound("/home/aarspace/Documents/PCV/Tugas Game/CNN/GUI/display/click-button-140881.mp3")
    click_sound = pygame.mixer.Sound("/home/aarspace/Documents/PCV/Tugas Game/CNN/GUI/display/sound_weapon_sniperriffle_l115a1_1.mp3")

    while True:
        SCREEN.fill("white")  # Latar belakang putih untuk halaman opsi
        SCREEN.blit(BG, (0, 0))  # Menampilkan background

        OPTIONS_MOUSE_POS = pygame.mouse.get_pos()

        OPTIONS_TEXT = get_font(45).render("OPTIONS", True, "Black")
        OPTIONS_RECT = OPTIONS_TEXT.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
        SCREEN.blit(OPTIONS_TEXT, OPTIONS_RECT)

        # Menambahkan slider untuk pengaturan suara
        slider.update(OPTIONS_MOUSE_POS, pygame.mouse.get_pressed()[0])
        slider.draw(SCREEN)

        # Menambahkan dropdown untuk pengaturan kamera
        dropdown.update(OPTIONS_MOUSE_POS, pygame.mouse.get_pressed()[0])
        dropdown.draw(SCREEN)

        # Tombol Back
        OPTIONS_BACK = Button(image=None, pos=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 1.75),
                              text_input="BACK", font=get_font(75), base_color="Black", hovering_color="Green")

        OPTIONS_BACK.changeColor(OPTIONS_MOUSE_POS)
        OPTIONS_BACK.update(SCREEN)

        # Menangani event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if OPTIONS_BACK.checkForInput(OPTIONS_MOUSE_POS):
                    click_sound.play()
                    pygame.quit()  # Keluarkan aplikasi

        pygame.display.update()

# Menjalankan halaman Options
options()
