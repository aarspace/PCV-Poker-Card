import os
import pygame

# Inisialisasi Pygame
pygame.init()

# Konfigurasi layar
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Card Viewer")

# Warna
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Jalur folder kartu
CARD_FOLDER = "/home/aarspace/Documents/PCV/Tugas Game/Final_Game/Assets/image/cards"

# Ukuran kartu
CARD_WIDTH = 100
CARD_HEIGHT = 150
CARD_SPACING = 20  # Jarak antar kartu

# Font untuk teks
font = pygame.font.Font(None, 24)

# Fungsi untuk memuat gambar kartu yang diminta
def load_card_images(requested_cards, card_folder):
    """Memuat gambar kartu yang diminta dari folder."""
    card_images = {}
    try:
        files_in_folder = {f.lower(): f for f in os.listdir(card_folder)}  # Case-insensitive
        for card_name in requested_cards:
            card_filename = card_name.strip().lower().replace(' ', '_') + '.png'

            if card_filename in files_in_folder:
                card_path = os.path.join(card_folder, files_in_folder[card_filename])
                try:
                    image = pygame.image.load(card_path).convert_alpha()
                    card_images[card_name] = image
                    print(f"Loaded {card_filename} successfully.")
                except pygame.error as e:
                    print(f"Error loading {card_filename}: {e}")
            else:
                print(f"Image for {card_name} not found.")
    except FileNotFoundError:
        print(f"Error: Folder {card_folder} not found.")
    return card_images

# Fungsi untuk menampilkan kartu
def display_cards(card_images, screen):
    """Menampilkan semua kartu di layar."""
    screen.fill(WHITE)  # Latar belakang putih
    x, y = CARD_SPACING, CARD_SPACING  # Posisi awal

    for card_name, card_image in card_images.items():
        # Tampilkan gambar kartu
        scaled_image = pygame.transform.scale(card_image, (CARD_WIDTH, CARD_HEIGHT))
        screen.blit(scaled_image, (x, y))

        # Tampilkan nama kartu di bawahnya
        card_text = font.render(card_name.replace('_', ' ').title(), True, BLACK)
        text_rect = card_text.get_rect(center=(x + CARD_WIDTH // 2, y + CARD_HEIGHT + 15))
        screen.blit(card_text, text_rect)

        # Pindah ke posisi berikutnya
        x += CARD_WIDTH + CARD_SPACING
        if x + CARD_WIDTH > SCREEN_WIDTH:  # Jika melebihi lebar layar, pindah ke baris berikutnya
            x = CARD_SPACING
            y += CARD_HEIGHT + CARD_SPACING + 20  # Tambahkan jarak lebih besar untuk teks

    pygame.display.flip()  # Perbarui layar

# Daftar kartu yang diminta
requested_cards = [
    'ten of Diamonds', 'nine of Hearts', 'ten of Spades', 'seven of Diamonds',
    'seven of Hearts', 'ace of Diamonds', 'three of Hearts', 'four of Hearts',
    'four of Clubs', 'two of Diamonds', 'jack of Clubs', 'nine of Spades',
    'eight of Diamonds', 'eight of Hearts', 'six of Clubs', 'ace of Spades',
    'jack of Hearts', 'queen of Diamonds', 'six of Hearts', 'queen of Spades',
    'two of Hearts', 'queen of Clubs', 'two of Spades', 'king of Spades',
    'eight of Clubs', 'six of Spades'
]

# Memuat kartu yang diminta
card_images = load_card_images(requested_cards, CARD_FOLDER)

# Loop utama Pygame
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Tampilkan kartu di layar
    display_cards(card_images, screen)

# Keluar dari Pygame
pygame.quit()
