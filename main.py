import cv2

for i in range(10):  # Coba hingga 5 perangkat yang berbeda
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Kamera ditemukan pada perangkat {i}")
        break
    cap.release()