import pygame
import time
import os

# Initialize Pygame
pygame.init()

# Set screen dimensions
screen_width = 1800
screen_height = 900
screen = pygame.display.set_mode((screen_width, screen_height))

# Set title
pygame.display.set_caption("Progress Bar Loading")

# Colors
bar_color = (0, 255, 255)
bar_background_color = (50, 50, 50)
text_color = (255, 255, 255)

# Font
font = pygame.font.Font(None, 60)

# Load background image (ensure you have a valid file path)
background_image_path = "background.jpg"  # or background.png
if os.path.exists(background_image_path):
    background_image = pygame.image.load(background_image_path)
    background_image = pygame.transform.scale(background_image, (screen_width, screen_height))
else:
    # If the background image is not found, use a solid color
    background_color = (30, 30, 30)

# Function to draw the progress bar
def draw_progress_bar(surface, x, y, width, height, progress):
    pygame.draw.rect(surface, bar_background_color, (x, y, width, height))  # Background
    pygame.draw.rect(surface, bar_color, (x, y, width * progress, height))  # Progress

# Function to draw the loading text
def draw_loading_text(surface, progress):
    loading_text = font.render(f"Loading... {int(progress * 100)}%", True, text_color)
    text_rect = loading_text.get_rect(center=(screen_width // 2, screen_height // 2 + 100))
    surface.blit(loading_text, text_rect)

# Main game loop
running = True
progress = 0  # Progress for loading bar (0 to 1)

# Progress bar dimensions
bar_width = 400
bar_height = 50
bar_x = (screen_width - bar_width) // 2  # Center the bar horizontally
bar_y = screen_height // 2 - bar_height // 2  # Center the bar vertically

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with the background image or solid color
    if os.path.exists(background_image_path):
        screen.blit(background_image, (0, 0))
    else:
        screen.fill(background_color)

    # Draw progress bar in the center of the screen
    draw_progress_bar(screen, bar_x, bar_y, bar_width, bar_height, progress)

    # Draw loading text in the center of the screen
    draw_loading_text(screen, progress)

    # Simulate loading by incrementing the progress
    if progress < 1:
        progress += 0.01  # Increase the progress by a small amount
        time.sleep(0.05)  # Simulate loading delay

    # Update the display with a smooth transition
    pygame.display.flip()

# Quit pygame
pygame.quit()
