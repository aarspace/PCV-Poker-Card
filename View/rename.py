import os

folder_path = '../Assets/image/cards'  # Ganti dengan path yang sesuai

# Ambil semua file dalam folder
file_list = os.listdir(folder_path)

# Cetak nama file di folder untuk memeriksa
for filename in file_list:
    print(filename)
