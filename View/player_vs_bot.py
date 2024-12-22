import pygame
import sys
import os

# Inisialisasi pygame
pygame.init()

# Set ukuran layar
screen_width = 1500
screen_height = 900
screen = pygame.display.set_mode((screen_width, screen_height))

# Set title window
pygame.display.set_caption("Game Player Vs Bot")

# Warna
white = (255, 255, 255)
gray = (200, 200, 200)
black = (0, 0, 0)
blue = (0, 0, 255)
button_color = (0, 128, 255)
button_hover_color = (0, 100, 255)
center_box_color = (50, 50, 50)  # Warna kotak tengah

# Score dan variabel permainan
player_score = 0
bot_score = 0
turn = "player | bot"
round_number = 1

# Font
font = pygame.font.Font(None, 36)

# Properti kartu
card_width = 70  # Lebar kartu
card_height = 100  # Tinggi kartu
card_spacing = -70  # Spacing antara kartu
card_offset_y = 10  # Offset untuk efek stacking (vertikal)

# Jumlah maksimum kartu per kolom
cards_per_column = 26

# Posisi area kartu pemain
player_card_x = 70
player_card_y = 70
player_card_slots = []

# Posisi area kartu bot (di sebelah kanan layar)
bot_card_x = screen_width - card_width - 60  # Posisi kartu bot di sebelah kanan
bot_card_y = 70
bot_card_slots = []

# Fungsi untuk mengambil font tertentu
def get_font(size):
    return pygame.font.Font("../Assets/font/font.ttf", size)

# Fungsi untuk memuat gambar kartu
def load_card_images(card_folder):
    """Memuat gambar kartu dari folder tertentu."""
    card_images = {}
    for filename in os.listdir(card_folder):
        if filename.endswith(".png"):
            card_name = filename.lower().replace(".png", "")  # Nama kartu seperti ace_of_hearts
            try:
                image = pygame.image.load(os.path.join(card_folder, filename))
                card_images[card_name] = image
            except pygame.error as e:
                print(f"Error loading {filename}: {e}")
    return card_images

# Folder tempat gambar kartu
card_folder = "/home/aarspace/Documents/PCV/Tugas Game/Final_Game/Assets/image/cards"
card_images_user = load_card_images(card_folder)
card_images_bot = load_card_images(card_folder)

# Membuat slot kartu untuk pemain
for i in range(cards_per_column):
    slot_x = player_card_x
    slot_y = player_card_y + i * (card_height + card_spacing) - (i * card_offset_y)  # Efek stacking dengan offset
    player_card_slots.append(pygame.Rect(slot_x, slot_y, card_width, card_height))

# Membuat slot kartu untuk bot
for i in range(cards_per_column):
    slot_x = bot_card_x
    slot_y = bot_card_y + i * (card_height + card_spacing) - (i * card_offset_y)  # Efek stacking dengan offset
    bot_card_slots.append(pygame.Rect(slot_x, slot_y, card_width, card_height))

center_box_width = 900
center_box_height = 500
center_box_x = (screen_width - center_box_width) // 2
center_box_y = (screen_height - center_box_height) // 2  # Posisi kotak tengah

# Fungsi untuk menggambar tombol
def draw_button(button_rect, text, color, hover_color):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    # Ganti warna jika mouse berada di atas tombol
    if button_rect.collidepoint(mouse_x, mouse_y):
        pygame.draw.rect(screen, hover_color, button_rect)  # Gambar tombol dengan warna hover
    else:
        pygame.draw.rect(screen, color, button_rect)  # Warna default tombol
    
    # Gambar teks tombol
    text_surf = font.render(text, True, white)
    text_rect = text_surf.get_rect(center=button_rect.center)
    screen.blit(text_surf, text_rect)

# Fungsi untuk menggambar skor
def draw_scores():
    player_score_text = get_font(20).render(f"Player Score: {player_score}", True, "#b68f40")
    bot_score_text = get_font(20).render(f"Bot Score: {bot_score}", True, "#b68f40")
    turn_text = get_font(20).render(f"Turn : {turn}", True, "#b68f40")

    player_score_rect = player_score_text.get_rect(center=(screen_width // 2 - 300, center_box_y - 50))
    bot_score_rect = bot_score_text.get_rect(center=(screen_width // 2 + 300, center_box_y - 50))
    turn_text_rect = turn_text.get_rect(center=(screen_width // 2 + 30, center_box_y - 150))

    screen.blit(player_score_text, player_score_rect)
    screen.blit(bot_score_text, bot_score_rect)
    screen.blit(turn_text, turn_text_rect)

# Fungsi untuk menggambar kartu
def draw_cards(available_cards, card_x, card_y, card_slots, card_width, card_height, card_spacing, card_offset_y, card_images, screen):
    """Menampilkan kartu yang masih ada di layar."""
    for i, card in enumerate(available_cards):
        card_name = card.strip().lower().replace(' ', '_')  # Format nama kartu untuk mencari di dictionary

        # Hitung posisi kartu
        card_rect = pygame.Rect(
            card_x, 
            card_y + i * (card_height + card_spacing) - (i * card_offset_y), 
            card_width, 
            card_height
        )

        # Jika gambar kartu ditemukan, tampilkan
        if card_name in card_images:
            scaled_image = pygame.transform.scale(card_images[card_name], (card_width, card_height))
            screen.blit(scaled_image, (card_rect.x, card_rect.y))
        else:
            pygame.draw.rect(screen, black, card_rect)  # Bingkai kartu
            pygame.draw.rect(screen, white, card_rect.inflate(-10, -10))  # Isi kartu

# Loop utama permainan
running = True
while running:
    # Menangkap event dari kamera atau tombol
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Menggambar latar belakang
    screen.fill(white)

    # Menggambar kartu pemain
    draw_cards(
        ['three of Spades', 'six of Diamonds', 'queen of Hearts', 'ten of Clubs', 'ace of Hearts', 'seven of Spades', 'nine of Clubs', 'ten of Diamonds', 'ace of Diamonds', 'ace of Spades', 'four of Diamonds', 'two of Hearts', 'three of Hearts', 'king of Hearts', 'six of Clubs', 'nine of Spades', 'six of Spades', 'seven of Diamonds', 'four of Clubs', 'nine of Diamonds', 'ten of Hearts', 'queen of Clubs', 'five of Hearts', 'king of Spades', 'two of Clubs', 'ten of Spades'],  # Contoh kartu
        player_card_x, player_card_y, player_card_slots, 
        card_width, card_height, card_spacing, card_offset_y, card_images_user, screen
    )

    # Menggambar kartu bot
    draw_cards(
        ['three of Spades', 'six of Diamonds', 'queen of Hearts', 'ten of Clubs', 'ace of Hearts', 'seven of Spades', 'nine of Clubs', 'ten of Diamonds', 'ace of Diamonds', 'ace of Spades', 'four of Diamonds', 'two of Hearts', 'three of Hearts', 'king of Hearts', 'six of Clubs', 'nine of Spades', 'six of Spades', 'seven of Diamonds', 'four of Clubs', 'nine of Diamonds', 'ten of Hearts', 'queen of Clubs', 'five of Hearts', 'king of Spades', 'two of Clubs', 'ten of Spades'],  # Contoh kartu
        bot_card_x, bot_card_y, bot_card_slots, 
        card_width, card_height, card_spacing, card_offset_y, card_images_bot, screen
    )

    # Menggambar skor
    draw_scores()

    # Menggambar tombol
    draw_button(pygame.Rect(100, 800, 200, 50), "Start Game", button_color, button_hover_color)
    draw_button(pygame.Rect(350, 800, 200, 50), "Pause", button_color, button_hover_color)
    
    # Update tampilan
    pygame.display.flip()

# Menutup pygame
pygame.quit()
sys.exit()
