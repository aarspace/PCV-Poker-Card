import pygame


class Button():
    def __init__(self, image, pos, text_input, font, base_color,
                 hovering_color, hover_sound=None, click_sound=None):
        self.image = image
        self.x_pos = pos[0]
        self.y_pos = pos[1]
        self.font = font
        self.base_color = base_color
        self.hovering_color = hovering_color
        self.text_input = text_input

        # Render teks tombol
        self.text = self.font.render(self.text_input, True, self.base_color)
        self.text_rect = self.text.get_rect(center=(self.x_pos, self.y_pos))

        # Jika tidak ada gambar, gunakan teks
        if self.image is None:
            self.image = self.text
        self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))

        # Menambahkan suara hover dan klik
        self.hover_sound = hover_sound
        self.click_sound = click_sound

        # Status untuk mendeteksi apakah suara hover sudah diputar
        self.hover_played = False
        self.was_hovered = False  # Untuk mendeteksi apakah kursor baru pertama kali masuk

    def update(self, screen):
        """Menggambar tombol pada layar"""
        if self.image is not None:
            screen.blit(self.image, self.rect)
        screen.blit(self.text, self.text_rect)

    def checkForInput(self, position):
        """Memeriksa apakah kursor berada di atas tombol dan klik terdeteksi"""
        if self.rect.collidepoint(position):
            return True
        return False

    def changeColor(self, position):
        is_hovered = self.rect.collidepoint(position)

        if is_hovered:
            if not self.was_hovered:
                # Suara hover diputar hanya sekali saat kursor pertama kali
                # memasuki tombolc
                if self.hover_sound and not self.hover_played:
                    self.hover_sound.play()  # Putar suara hover
                    self.hover_played = True  # Tandai bahwa suara sudah diputar
                self.was_hovered = True  # Menandakan kursor sudah memasuki tombol
            # Ubah warna teks tombol ke warna hover
            self.text = self.font.render(
                self.text_input, True, self.hovering_color)
        else:
            # Reset saat kursor keluar dari tombol
            self.was_hovered = False
            self.hover_played = False  # Reset agar suara hover bisa diputar lagi saat kursor masuk
            # Kembali ke warna teks dasar
            self.text = self.font.render(
                self.text_input, True, self.base_color)
