from gtts import gTTS
import os

# Daftar "Bot card" yang ingin dikonversi menjadi teks ke suara
cards = [
    "Bot Win",
    "Player Win"
]

# Fungsi untuk mengkonversi teks ke file MP3
def text_to_speech_mp3(text, filename):
    tts = gTTS(text=text, lang='en')  # Bahasa Inggris
    tts.save(filename)  # Menyimpan suara sebagai file MP3
    print(f"File MP3 telah disimpan sebagai {filename}")

# Loop untuk mengonversi setiap elemen dalam list 'cards' menjadi file MP3
for card in cards:
    # Membuat nama file berdasarkan teks
    filename = f"{card.replace(' ', '_').replace(':', '')}.mp3"
    
    # Mengonversi teks ke MP3
    text_to_speech_mp3(card, filename)

print("Konversi selesai!")
