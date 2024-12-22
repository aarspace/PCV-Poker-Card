import random
import cv2
import numpy as np
from PIL import Image
import timm
import torch
from torchvision import transforms
import torch.nn as nn
import threading
import time
from collections import Counter


# Peta nilai kartu
card_values = {
    'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
    'eight': 8, 'nine': 9, 'ten': 10, 'jack': 11, 'queen': 12, 'king': 13, 'ace': 14
}

# Peta simbol kartu
suit_order = {
    'spades': 4, 'hearts': 3, 'clubs': 2, 'diamonds': 1
}

# Representasi dek kartu
suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
ranks = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'ace']

# Membuat dek kartu
deck = [f'{rank} of {suit}' for suit in suits for rank in ranks]

# Fungsi untuk load model
def load_model():
    class SimpleCardClassifier(nn.Module):
        def __init__(self, num_classes=53):
            super(SimpleCardClassifier, self).__init__()
            self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
            self.features = nn.Sequential(*list(self.base_model.children())[:-1])

            enet_out_size = 1280
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(enet_out_size, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            output = self.classifier(x)
            return output

    model = SimpleCardClassifier(num_classes=53)
    model.load_state_dict(torch.load('/home/aarspace/Documents/PCV/Tugas Game/CNN/Model_DariAwal/Model_1/model/trained_model.pth', map_location=torch.device('cpu')))
    model.to(torch.device('cpu'))
    model.eval()
    return model
# Fungsi untuk membaca class labels
def read_class_labels(file_path):
    with open(file_path, 'r') as file:
        class_labels = [line.strip() for line in file]
    return class_labels
# Fungsi untuk memproses gambar
def preprocess_frame(frame, transform):
    image = Image.fromarray(frame).convert("RGB")
    return image, transform(image).unsqueeze(0)
# Fungsi untuk mengurutkan titik-titik dalam urutan yang benar
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # Titik kiri atas
    rect[2] = pts[np.argmax(s)]  # Titik kanan bawah
    rect[1] = pts[np.argmin(diff)]  # Titik kanan atas
    rect[3] = pts[np.argmax(diff)]  # Titik kiri bawah

    return rect

# Fungsi untuk melakukan perspektif transformasi pada gambar
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    return warped

# Fungsi untuk prediksi kartu
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

# Fungsi untuk mendeteksi kartu pengguna
def detect_user_cards(frame, model, transform, class_labels, min_confidence=50.0, min_area=1000):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_cards = []  # Menyimpan kartu yang terdeteksi

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        if len(approx) == 4:  # Deteksi kontur berbentuk segi empat (kartu)
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            
            if w > 100 and h > 100 and area > min_area:  # Cek ukuran kartu dan area kontur
                rect = np.array([point[0] for point in approx], dtype="int")
                warped = four_point_transform(frame, rect)

                # Preprocess dan prediksi
                image, preprocessed_frame = preprocess_frame(warped, transform)
                probabilities = predict(model, preprocessed_frame, torch.device('cpu'))
                predicted_class = np.argmax(probabilities)
                predicted_label = class_labels[predicted_class]
                confidence = probabilities[predicted_class] * 100

                if confidence >= min_confidence:  # Hanya menerima deteksi dengan confidence tinggi
                    detected_cards.append((predicted_label, confidence, rect))

                # Hentikan jika sudah mendeteksi 5 kartu
                if len(detected_cards) >= 5:
                    break

    return detected_cards, frame

# Fungsi untuk membagi kartu (untuk bot dan user)
def deal_cards(deck):
    random.shuffle(deck)
    user_cards = deck[:26]  # Player gets 26 cards
    bot_cards = deck[26:52]  # Bot gets 26 cards
    return user_cards, bot_cards




def bot_choose_card_single(bot_cards, used_cards, hand_combination, last_player_card):
    # Mendapatkan nilai dari kartu pemain
    player_card_value = card_value_from_text(hand_combination)

    # Jika kartu pemain tidak terdeteksi (nilai kartu pemain = 0), bot tidak memilih kartu
    if player_card_value == 0:
        print("Kartu pemain tidak terdeteksi, bot tidak memilih kartu.")
        # Hanya update available_cards berdasarkan used_cards yang sudah ada
        available_cards = [card for card in bot_cards if card not in used_cards]
        print(f"Kartu yang masih bisa digunakan oleh bot: {available_cards}")
        print(f"Kartu yang sudah digunakan oleh bot: {used_cards}")
        return None, last_player_card, used_cards, available_cards

    # Menghapus kartu yang sudah digunakan dari daftar kartu bot
    available_cards = [card for card in bot_cards if card not in used_cards]

    # Jika tidak ada kartu yang tersedia, kembalikan None
    if not available_cards:
        print("Tidak ada kartu yang tersedia untuk dipilih.")
        return None, last_player_card, used_cards, bot_cards

    print(f"Nilai kartu pemain: {player_card_value} (Kartu: {hand_combination})")

    selected_card = None

    # Cek kartu yang tersedia dan pilih kartu yang lebih besar atau sama dengan kartu pemain
    for card in available_cards:
        card_value = card_value_from_text(card)
        print(f"Nilai kartu bot: {card_value} (Kartu: {card})")

        if card_value >= player_card_value:
            selected_card = card
            break

    # Jika bot memilih kartu
    if selected_card:
        used_cards.append(selected_card)  # Tambahkan kartu yang dipilih ke daftar used_cards
        available_cards.remove(selected_card)  # Hapus kartu yang dipilih dari daftar available_cards
        # card_value_bot_selected = card_value_from_text(selected_card)
        # print(f"Nilai Kartu Pilihan Bot : {card_value_bot_selected}")
        print(f"Bot memilih kartu: {selected_card}")
    else:
        print("Bot tidak memilih kartu karena tidak ada kartu yang lebih besar atau sama dengan kartu pemain.")

    return selected_card, last_player_card, used_cards, available_cards



# def evaluate_bot_hand(bot_cards):
#     # Cek jika bot_cards adalah None atau kosong

#     # Pisahkan rank dan suit dengan pengecekan validitas format kartu
#     ranks = []
#     suits = []
    
#     # Memastikan kartu dalam format yang benar
#     for card in bot_cards:
#         if ' of ' in card:  # Format yang benar adalah 'Rank of Suit'
#             try:
#                 rank, suit = card.split(' of ')
#                 ranks.append(rank)
#                 suits.append(suit)
#             except ValueError:
#                 # Jika format kartu salah, kembalikan pesan kesalahan
#                 print(f"Error: Invalid card format detected: {card}")
#                 return
#         else:
#             print(f"Error: Invalid card format detected: {card}")
#             return

#     # Identifikasi kombinasi dengan pengecekan hasil kembalian
#     pair = is_pair_bot(ranks, bot_cards) if is_pair_bot(ranks, bot_cards) else []
#     two_pair = is_two_pair_bot(ranks, bot_cards) if is_two_pair_bot(ranks, bot_cards) else []
#     three_of_a_kind = is_three_of_a_kind_bot(ranks, bot_cards) if is_three_of_a_kind_bot(ranks, bot_cards) else []
#     full_house = is_full_house_bot(ranks, bot_cards) if is_full_house_bot(ranks, bot_cards) else []
#     flush = is_flush_bot(suits, bot_cards) if is_flush_bot(suits, bot_cards) else []
#     straight = is_straight_bot(ranks) if is_straight_bot(ranks) else []

#     combinations = []

#     # Menambahkan kombinasi yang valid
#     if full_house:
#         combinations.append(f"Full House: {', '.join(full_house)}")
#     if straight:
#         combinations.append(f"Straight: {', '.join(straight)}")
#     if flush:
#         combinations.append(f"Flush: {', '.join(flush)}")
#     if three_of_a_kind:
#         combinations.append(f"Three of a Kind: {', '.join(three_of_a_kind)}")
#     if two_pair:
#         combinations.append(f"Two Pair: {', '.join(two_pair)}")
#     if pair:
#         combinations.append(f"Pair: {', '.join(pair)}")

#     # Tambahkan pengecekan tambahan untuk mencetak kombinasi yang ada
#     if combinations:
#             # print(combination)
#             return combinations
#     else:
#         # print("Bot combin None")  # Pastikan hanya dicetak jika tidak ada kombinasi
#         return None


def evaluate_bot_hand(bot_cards):
    # Cek jika bot_cards adalah None atau kosong
    if not bot_cards:
        return "", "", "", "", "", ""

    # Pisahkan rank dan suit
    ranks = []
    suits = []
    
    # Memastikan kartu dalam format yang benar
    for card in bot_cards:
        if ' of ' in card:  # Format yang benar adalah 'Rank of Suit'
            try:
                rank, suit = card.split(' of ')
                ranks.append(rank)
                suits.append(suit)
            except ValueError:
                print(f"Error: Invalid card format detected: {card}")
                return "", "", "", "", "", ""
        else:
            print(f"Error: Invalid card format detected: {card}")
            return "", "", "", "", "", ""

    # Identifikasi kombinasi dengan pengecekan hasil kembalian
    pair = is_pair_bot(ranks, bot_cards)
    two_pair = is_two_pair_bot(ranks, bot_cards)
    three_of_a_kind = is_three_of_a_kind_bot(ranks, bot_cards)
    full_house = is_full_house_bot(ranks, bot_cards)
    flush = is_flush_bot(suits, bot_cards)
    straight = is_straight_bot(ranks)

    # Convert hasil ke string jika ada kombinasi yang ditemukan
    pair_str = f"Pair: {', '.join(pair)}" if pair else ""
    two_pair_str = f"Two Pair: {', '.join(two_pair)}" if two_pair else ""
    three_of_a_kind_str = f"Three of a Kind: {', '.join(three_of_a_kind)}" if three_of_a_kind else ""
    full_house_str = f"Full House: {', '.join(full_house)}" if full_house else ""
    flush_str = f"Flush: {', '.join(flush)}" if flush else ""
    straight_str = f"Straight: {', '.join(straight)}" if straight else ""

    return full_house_str, three_of_a_kind_str, pair_str, two_pair_str, flush_str, straight_str




def is_pair_bot(ranks, bot_cards):
    """
    Mengidentifikasi apakah ada Pair dalam ranks dan mengembalikan kombinasi Pair.
    """
    rank_counts = Counter(ranks)
    pair = [rank for rank, count in rank_counts.items() if count == 2]

    if pair:
        # Ambil kartu lengkap dari Pair
        return [card for card in bot_cards if card.split(' of ')[0] == pair[0]]
    
    return None


def is_two_pair_bot(ranks, bot_cards):
    """
    Mengidentifikasi apakah ada Two Pair dalam ranks dan mengembalikan pasangan-pasangan yang ditemukan.
    """
    rank_counts = Counter(ranks)
    pairs = [rank for rank, count in rank_counts.items() if count == 2]
    
    if len(pairs) == 2:
        # Ambil kartu yang cocok dengan dua pasangan tersebut
        two_pair_cards = []
        for pair in pairs:
            two_pair_cards.extend([card for card in bot_cards if card.split(' of ')[0] == pair])
        return two_pair_cards
    return None


def is_three_of_a_kind_bot(ranks, bot_cards):
    """
    Mengidentifikasi apakah ada Three of a Kind dalam ranks dan mengembalikan kombinasi Three of a Kind.
    """
    rank_counts = Counter(ranks)
    three_of_a_kind = [rank for rank, count in rank_counts.items() if count == 3]

    if three_of_a_kind:
        # Ambil kartu lengkap dari Three of a Kind
        return [card for card in bot_cards if card.split(' of ')[0] == three_of_a_kind[0]]
    
    return None


def is_full_house_bot(ranks, bot_cards):
    """
    Mengidentifikasi apakah ada Full House dalam ranks dan mengembalikan kombinasi Full House.
    Full House adalah kombinasi 3 kartu dengan rank yang sama (Three of a Kind) dan 2 kartu dengan rank yang sama (Pair).
    """
    # Urutan rank yang benar sesuai dengan nilai kartu
    rank_order = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'ace']

    
    # Menghitung jumlah kemunculan setiap rank
    rank_counts = Counter(ranks)
    
    # Mencari Three of a Kind (3 kartu dengan rank yang sama)
    three_of_a_kind = [rank for rank, count in rank_counts.items() if count == 3]
    # Mencari Pair (2 kartu dengan rank yang sama)
    pair = [rank for rank, count in rank_counts.items() if count == 2]
    
    if three_of_a_kind and pair:
        # Mengurutkan Three of a Kind dan Pair sesuai dengan urutan rank (dari yang lebih besar)
        three_of_a_kind.sort(key=lambda x: rank_order.index(x), reverse=True)
        pair.sort(key=lambda x: rank_order.index(x), reverse=True)

        # Ambil Three of a Kind yang lebih besar terlebih dahulu
        three_rank = three_of_a_kind[0]
        # Ambil Pair yang lebih kecil
        pair_rank = pair[0]

        # Ambil kartu lengkap untuk Three of a Kind
        three_cards = [card for card in bot_cards if card.split(' of ')[0] == three_rank]
        # Ambil kartu lengkap untuk Pair
        pair_cards = [card for card in bot_cards if card.split(' of ')[0] == pair_rank]

        # Gabungkan hasil Three of a Kind dan Pair, pastikan urutan dimulai dari yang lebih besar
        full_house = three_cards[:3] + pair_cards[:2]
        
        # Urutkan kartu-kartu hasil Full House berdasarkan urutan rank
        full_house.sort(key=lambda card: rank_order.index(card.split(' of ')[0]), reverse=True)
        
        return full_house  # Mengembalikan kombinasi Full House yang terurut
    
    return None

def is_straight_bot(ranks):
    if len(ranks) < 5:  # Straight membutuhkan setidaknya 5 kartu
        return None

    ranks = ['A' if x.lower() == 'ace' else x for x in ranks]

    rank_order = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    try:
        ranks_sorted = sorted(ranks, key=lambda x: rank_order.index(x))
    except ValueError as e:
        return None

    if rank_order.index(ranks_sorted[-1]) - rank_order.index(ranks_sorted[0]) == len(ranks_sorted) - 1:
        return ranks_sorted
    return None

def is_flush_bot(suits, bot_cards):
    # Cek apakah semua kartu memiliki suit yang sama
    if len(set(suits)) == 1:  # Jika hanya ada satu jenis suit, maka Flush
        return bot_cards
    return None



def is_four_of_a_kind(ranks):
    """
    Mengidentifikasi apakah ada Four of a Kind dalam ranks dan mengembalikan empat kartu yang membentuk Four of a Kind.
    """
    rank_counts = Counter(ranks)
    four_of_a_kind = [rank for rank, count in rank_counts.items() if count == 4]
    
    if four_of_a_kind:
        # Ambil empat kartu yang cocok dengan Four of a Kind
        four_cards = [card for card in ranks if card == four_of_a_kind[0]]
        return four_cards
    return None



def check_hand_combination(detected_cards):
    # Cek jika detected_cards kosong
    if not detected_cards:
        return "No cards detected"

    # Cek format dan pisahkan rank dan suit
    try:
        ranks = [card[0].split(' of ')[0] for card in detected_cards]
        suits = [card[0].split(' of ')[1] for card in detected_cards]
    except IndexError:
        return "Error: Invalid card format detected."

    # Pemetaan nama kartu ke format standar untuk dibandingkan
    rank_mapping = {
        '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
        'jack': 'jack', 'queen': 'queen', 'king': 'king', 'ace': 'ace',
        'J': 'jack', 'Q': 'queen', 'K': 'king', 'A': 'ace'
    }

    # Menyesuaikan rank ke format standar yang sesuai dengan rank_order
    ranks = [rank_mapping.get(rank.lower(), rank.lower()) for rank in ranks]

    # Identifikasi kombinasi
    pair = is_pair(ranks, detected_cards)
    two_pair = is_two_pair(ranks, detected_cards)
    three_of_a_kind = is_three_of_a_kind(ranks, detected_cards)
    full_house = is_full_house(ranks, detected_cards)
    straight = is_straight(ranks)
    flush = is_flush(suits, detected_cards)

    # if full_house:
    #     return ', '.join(full_house)
    # elif flush:
    #     return ', '.join(flush)
    # elif straight:
    #     return ', '.join(straight)
    # elif three_of_a_kind:
    #     return ', '.join(three_of_a_kind)
    # elif two_pair:
    #     return ', '.join(two_pair)
    # elif pair:
    #     return ', '.join(pair)
    # else:
    #     # Jika tidak ada kombinasi (hanya kartu tunggal), kembalikan kartu tertinggi yang terdeteksi
    #     return detected_cards[0][0]  # Ambil kartu pertama yang terdeteksi tanpa label
    
    
    # Kembalikan kombinasi kartu yang terdeteksi
    if full_house:
        return f"Full House: {', '.join(full_house)}"
    elif flush:
        return f"Flush: {', '.join(flush)}"
    elif straight:
        return f"Straight: {', '.join(straight)}"
    elif three_of_a_kind:
        return f"Three of a Kind: {', '.join(three_of_a_kind)}"
    elif two_pair:
        return f"Two Pair: {', '.join(two_pair)}"
    elif pair:
        return f"Pair: {', '.join(pair)}"
    else:
        return "Single Card: " + detected_cards[0][0]  # Jika tidak ada kombinasi, kartu tertinggi


# Fungsi Kombinasi Kartu
def is_pair(ranks, detected_cards):
    rank_counts = {rank: ranks.count(rank) for rank in ranks}
    pairs = [rank for rank, count in rank_counts.items() if count == 2]
    if len(pairs) == 1:
        pair_cards = [card[0] for card in detected_cards if card[0].split(' of ')[0] == pairs[0]]
        return pair_cards
    return None

def is_two_pair(ranks, detected_cards):
    rank_counts = {rank: ranks.count(rank) for rank in ranks}
    pairs = [rank for rank, count in rank_counts.items() if count == 2]
    if len(pairs) == 2:
        two_pair_cards = []
        for pair in pairs:
            two_pair_cards.extend([card[0] for card in detected_cards if card[0].split(' of ')[0] == pair])
        return two_pair_cards
    return None

def is_three_of_a_kind(ranks, detected_cards):
    rank_counts = {rank: ranks.count(rank) for rank in ranks}
    threes = [rank for rank, count in rank_counts.items() if count == 3]
    if threes:
        three_cards = [card[0] for card in detected_cards if card[0].split(' of ')[0] == threes[0]]
        return three_cards
    return None

def is_full_house(ranks, detected_cards):
    # Pemetaan Ace menjadi 'A' untuk konsistensi
    ranks = ['A' if x.lower() == 'ace' else x for x in ranks]
    
    rank_counts = {rank: ranks.count(rank) for rank in ranks}
    threes = [rank for rank, count in rank_counts.items() if count == 3]
    pairs = [rank for rank, count in rank_counts.items() if count == 2]
    
    if len(threes) == 1 and len(pairs) == 1:
        # Mengambil 3 kartu yang sama dan 2 kartu pasangan
        three_cards = [card[0] for card in detected_cards if card[0].split(' of ')[0] == threes[0]]
        pair_cards = [card[0] for card in detected_cards if card[0].split(' of ')[0] == pairs[0]]
        
        # Gabungkan tiga kartu dan dua kartu, pastikan urutan tiga kartu (Three of a Kind) muncul dulu
        return three_cards + pair_cards
    return None


def is_straight(ranks):
    if len(ranks) < 5:
        return None

    ranks = ['A' if x.lower() == 'ace' else x for x in ranks]
    rank_order = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    try:
        ranks_sorted = sorted(ranks, key=lambda x: rank_order.index(x))
    except ValueError:
        return None

    if rank_order.index(ranks_sorted[-1]) - rank_order.index(ranks_sorted[0]) == len(ranks_sorted) - 1:
        return ranks_sorted
    return None


def is_flush(suits, detected_cards):
    suit_counts = {suit: suits.count(suit) for suit in suits}
    flush_suits = [suit for suit, count in suit_counts.items() if count >= 5]
    if flush_suits:
        flush_cards = [card[0] for card in detected_cards if card[0].split(' of ')[1] == flush_suits[0]]
        return flush_cards
    return None


def single_or_combination(combination_type):
    # Menghapus bagian setelah tanda ":" dan menghilangkan spasi ekstra
    combination_type = combination_type.split(":")[0].strip().lower()

    # Memeriksa kombinasi dan menjalankan aksi yang sesuai
    if combination_type == "full house":
        print("Full House")
    elif combination_type == "two pair":
        print("Two Pair")
    elif combination_type == "three of a kind":
        print("Three of a Kind")
    elif combination_type == "flush":
        print("Flush")
    elif combination_type == "straight":
        print("Straight")
    elif combination_type == "pair":
        print("Pair")
    elif combination_type == "single card":
        print("Single Card")
    else:
        print("Kombinasi tidak dikenali")




def card_value_from_text(card_text):
    # Menghapus kata-kata yang tidak perlu (Pair:, Full House:, dll)
    card_text = card_text.lower().replace('single card: ', '').replace('full house: ', '').replace('flush: ', '').replace('straight: ', '').replace('three of a kind: ', '').replace('two pair: ', '').replace('pair: ', '').strip()
    
    # Pisahkan kartu berdasarkan ' of '
    words = card_text.split(' of ')
    if len(words) != 2:
        return 0  # Jika formatnya salah
    
    card_rank = card_values.get(words[0], 0)  # Mendapatkan nilai rank kartu
    suit_rank = suit_order.get(words[1], 0)  # Mendapatkan nilai suit kartu
    
    if card_rank == 0 or suit_rank == 0:
        return 0  # Jika kartu tidak dikenali, kembalikan 0
    
    # Mengembalikan jumlah nilai rank dan suit kartu
    return card_rank + suit_rank

def card_value_from_text_bulk(cards_text):
    # Menghapus kata-kata yang tidak perlu
    cards_text = cards_text.lower().replace('single card: ', '').replace('full house: ', '').replace('flush: ', '').replace('straight: ', '').replace('three of a kind: ', '').replace('two pair: ', '').replace('pair: ', '').strip()

    # Pisahkan kartu berdasarkan koma (jika ada lebih dari satu kartu)
    cards = cards_text.split(',')  
    total_value = 0
    
    for card in cards:
        # Pisahkan setiap kartu berdasarkan ' of '
        card = card.strip()  # Hapus spasi ekstra di sekitar kartu
        words = card.split(' of ')
        
        if len(words) != 2:
            continue  # Jika formatnya salah, lewati kartu ini
        
        card_rank = card_values.get(words[0], 0)  # Mendapatkan nilai rank kartu
        suit_rank = suit_order.get(words[1], 0)  # Mendapatkan nilai suit kartu
        
        if card_rank == 0 or suit_rank == 0:
            continue  # Jika kartu tidak dikenali, lewati kartu ini
        
        # Menambahkan nilai kartu ini ke total
        total_value += card_rank + suit_rank
    
    return total_value




# Fungsi untuk mengubah deskripsi kartu (misal: "two of spade") menjadi nilai numerik
# def card_value_from_text(card_text):
#     # print(f"Processing card: {card_text}")  # Debugging
#     words = card_text.lower().split(' of ')
#     if len(words) != 2:
#         print(f"Invalid format: {card_text}")  # Debugging
#         return 0  # Mengembalikan 0 jika format input salah
    
#     card_rank = card_values.get(words[0], 0)
#     suit_rank = suit_order.get(words[1], 0)
    
#     # print(f"Rank: {words[0]}, Suit: {words[1]}, Card Value: {card_rank + suit_rank}")  # Debugging

#     # Jika kartu tidak dikenali, kembalikan 0
#     if card_rank == 0 or suit_rank == 0:
#         # print(f"Card not recognized: {card_text}")  # Debugging
#         return 0
    
#     return card_rank + suit_rank



def evaluate_combination(cards):
    # Mendapatkan nilai kartu dari setiap kartu dalam kombinasi
    card_values_list = [card_value_from_text(card) for card in cards]
    
    # Periksa apakah ada kartu yang tidak dikenali (nilai 0)
    if any(value == 0 for value in card_values_list):
        return 0  # Jika ada kartu yang tidak dikenali, kembalikan 0
    
    # Menyusun nilai kartu tanpa simbol (hanya nilai rank kartu)
    total_value = sum(card_values_list)

    return total_value


# Fungsi untuk memproses input string
def process_input(input_text):
    # Pisahkan input berdasarkan koma dan strip spasi
    cards = [card.strip() for card in input_text.split(',')]
    
    # Evaluasi total value dari kartu yang ada
    total_value = evaluate_combination(cards)
    return total_value

def convert_cards_to_values(cards):
    # Menggunakan card_value_from_text untuk mengonversi setiap kartu menjadi nilai angka
    return [card_value_from_text(card) for card in cards]


def compare_cards(player_cards, bot_cards):
    # Misalkan compare_cards melakukan perbandingan antara player_cards dan bot_cards
    # dan mengembalikan string hasil perbandingan: "Player wins!", "Bot wins!", atau "Tie"
    player_card_value = card_value_from_text(player_cards)
    bot_card_value = card_value_from_text(bot_cards)

    if player_card_value > bot_card_value:
        return "Player wins!"
    elif bot_card_value > player_card_value:
        return "Bot wins!"
    else:
        return "Tie"


def update_scores(player_card_value, bot_card_value):
    global user_score, bot_score

    # Bandingkan nilai kartu dan perbarui skor
    if player_card_value > bot_card_value:
        user_score += 1
        print(f"Player wins this round! Score: Player {user_score} - Bot {bot_score}")
    elif player_card_value < bot_card_value:
        bot_score += 1
        print(f"Bot wins this round! Score: Player {user_score} - Bot {bot_score}")
    else:
        print(f"It's a tie! Score: Player {user_score} - Bot {bot_score}")



def use_card_user(user_cards, used_user_cards, card_text, remove_card=None):

# Jika card_text adalah string, hapus label dan teks lainnya
    # Jika card_text adalah string, hapus label dan teks lainnya
    if isinstance(card_text, str):
        card_text = card_text.lower()
        
        # Debugging: Cek teks sebelum pembersihan
        print(f"Original card text: {card_text}")
        
        # Hapus label seperti "single card:", "full house:", dll.
        card_text = card_text.replace('single card: ', '').replace('full house: ', '').replace('flush: ', '')\
            .replace('straight: ', '').replace('three of a kind: ', '').replace('two pair: ', '').replace('pair: ', '')\
            .strip()

        # Debugging: Cek teks setelah pembersihan label
        print(f"Card text after removing labels: {card_text}")

        # Proses pemisahan kartu berdasarkan koma jika ada
        if card_text:
            card_text = [card.strip() for card in card_text.split(',')]

        # Debugging: Cek kartu yang akan digunakan setelah pemisahan
        print(f"Cards to use (after splitting): {card_text}")
    
    # Jika card_text adalah tuple atau list, proses setiap elemen dan pastikan hanya kartu yang valid
    elif isinstance(card_text, (tuple, list)):
        card_text = [str(card).strip() for card in card_text if isinstance(card, str)]
    
    else:
        print(f"Invalid input: {card_text}")
        return

    # Proses penggunaan kartu
    for selected_card in card_text:
        if selected_card in user_cards:
            # Menggunakan kartu jika ada dalam daftar user_cards
            user_cards.remove(selected_card)
            used_user_cards.append(selected_card)
            print(f"Card '{selected_card}' used by player.")
        else:
            print(f"Card '{selected_card}' is not available in user's cards.")

    # Debugging: Menampilkan hasil akhir
    print(f"Remaining user cards: {user_cards}")
    print(f"Used user cards: {used_user_cards}")
    
    # Jika ada kartu yang perlu dihapus dari used_user_cards
    if remove_card:
        if remove_card in used_user_cards:
            used_user_cards.remove(remove_card)
            print(f"Card '{remove_card}' removed from used cards.")
        else:
            print(f"Card '{remove_card}' not found in used cards.")

def player_block(hand_combination, bot_cards):
    """
    Memeriksa apakah kartu yang akan dimainkan oleh player ada di dalam kartu bot.
    Jika ada, blokir kartu tersebut agar tidak bisa dimainkan.
    """
    # Bersihkan hand_combination dengan menghapus 'single card:' (tanpa memperhatikan kapitalisasi) dan ubah ke huruf kecil
    cleaned_hand_combination = hand_combination.lower().replace('single card:', '').strip()

    # Debug: Cek hasil cleaned_hand_combination
    print(f"Cleaned hand combination: '{cleaned_hand_combination}'")

    # Membuat list kartu bot dalam format lowercase dan menghilangkan spasi ekstra
    bot_cards_lower = [card.lower().strip() for card in bot_cards]

    # Debug: Cek hasil bot_cards_lower
    print(f"Bot cards (lowercase): {bot_cards_lower}")
    
    # Periksa apakah kartu yang dimaksud ada dalam kartu bot
    if cleaned_hand_combination in bot_cards_lower:
        print(f"Kartu '{cleaned_hand_combination}' bukan milik user, tidak bisa dimainkan.")
        return True  # Mengembalikan True jika kartu ada di dalam bot
    else:
        print(f"Kartu '{cleaned_hand_combination}' bisa dimainkan.")
        return False  # Mengembalikan False jika kartu tidak ada di dalam bot





def use_card_bot(bot_cards, used_bot_cards, selected_card):
    if selected_card in bot_cards:
        bot_cards.remove(selected_card)  # Menghapus kartu dari list bot_cards
        used_bot_cards.append(selected_card)  # Menambahkan kartu ke list used_bot_cards
        print(f"Card '{selected_card}' used by bot.")
    else:
        print(f"Card '{selected_card}' is not available in bot's cards.")


# Misalnya, kita menambahkan print statement untuk mendebug
# Inisialisasi variabel penting
model = load_model()  # Memuat model sebelum digunakan
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
class_labels = read_class_labels('/home/aarspace/Documents/PCV/Tugas Game/CNN/Model_DariAwal/Model_1/classes.txt')  # Pastikan jalur file class_labels benar

# Misalnya kita siapkan kartu bot untuk pengujian
# bot_cards = ["A of Hearts", "A of Spades", "A of Clubs", "K of Diamonds", "K of Hearts"]

# Evaluasi kombinasi tangan bot
# bot_hand = evaluate_bot_hand(bot_cards)
# print(f"Bot's hand combination: {bot_hand}")
# Skor dan pengaturan permainan
user_score = 0
bot_score = 0
max_score = 20
game_continue = False
round_in_progress = False

# Menunggu input pengguna untuk memulai permainan
print("Apakah Anda mau mulai? (yes/no)")
start_input = input().strip().lower()


def wait_for_input():
    input_text = input("Press 'Enter' to continue to the next round.")
    return input_text == ""  # Mengembalikan True jika input adalah Enter


# Fungsi untuk menangani input pengguna dalam thread terpisah
def input_thread():
    global input_received
    input_received = wait_for_input()

# Modifikasi pada bagian utama kode game loop:
if start_input == "yes":
    user_cards, bot_cards = deal_cards(deck)
    used_user_cards = []  # Menyimpan kartu yang sudah dikeluarkan oleh pengguna
    used_bot_cards = []  # Menyimpan kartu yang sudah dikeluarkan oleh bot
    bot_cards1 = bot_cards
    used_cards1 = []  # Kartu yang sudah digunakan oleh bot
    last_player_card = ""  # Kartu pemain sebelumnya, kosong pada awalnya
    available_cards1 = []
    last_player_hand = None  # Menyimpan kombinasi tangan pemain sebelumnya
    last_printed_available_cards = None

    # test1 = evaluate_bot_hand(bot_cards)
    # print(f"Bot's hand combination: {bot_hand_combination}")
    # print(f"Player's Cards: {user_cards}")
    # print(f"Bot's Cards: {bot_cards}")
   
    # Capture video untuk deteksi kartu pengguna
    cap = cv2.VideoCapture(0)

    input_received = False  # Flag untuk mendeteksi apakah input telah diterima

    # Membuat thread untuk menangani input, agar tidak memblokir pengolahan video
    def handle_input():
        global input_received
        while True:
            input_received = wait_for_input()

    # Menjalankan thread input secara terpisah
    input_thread_instance = threading.Thread(target=handle_input)
    input_thread_instance.daemon = True
    input_thread_instance.start()

    print("kartu Bot :", bot_cards)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detected_cards, frame = detect_user_cards(frame, model, transform, class_labels)

        # Tampilkan kartu yang terdeteksi pada frame
        for i, (card, confidence, rect) in enumerate(detected_cards):
            label_text = f"Card {i+1}: {card} ({confidence:.2f}%)"
            cv2.putText(frame, label_text, (50, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.polylines(frame, [rect], isClosed=True, color=(0, 255, 0), thickness=2)
        

        hand_combination = check_hand_combination(detected_cards)  # Pastikan fungsi ini ada dan berjalan
        if hand_combination != "No Combination":
            cv2.putText(frame, f"Player's hand: {hand_combination}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Score: Player {user_score} - Bot {bot_score}", (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # input_text12 = "eight of club, eight of spade"
        # Variabel untuk kartu yang tersedia dan yang sudah digunakan oleh bot
        combination_type = hand_combination.split(":")[0].strip().lower()
        combin_list = ["full house", "three of a kind", "straight", "pair", "two pair"]
        full_house, three_of_a_kind, pair, two_pair, flush, straight = evaluate_bot_hand(available_cards1)
        
        if player_block(hand_combination, bot_cards):  # Jika kartu ada di bot
            cv2.putText(frame, "Kartu ini milik bot! Pilih kartu lain.", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Jika kartu valid (bukan milik bot), lanjutkan ke langkah berikutnya
            print(f"Kartu '{hand_combination}' dapat dimainkan!")
            if combination_type == "single card":
        # Di dalam while loop
                if hand_combination != last_player_hand:  # Hanya memilih kartu jika kartu pemain berubah
                    last_player_hand = hand_combination
                    
                    selected_card_1, last_player_card, used_cards1, available_cards1 = bot_choose_card_single(
                        bot_cards1, used_cards1, hand_combination, last_player_card)
                    card_value_bot_selected = card_value_from_text(selected_card_1)
                    card_value_player_selected = card_value_from_text(hand_combination)
                    score_single = update_scores(card_value_player_selected, card_value_bot_selected)
                    use_card_user(user_cards, used_user_cards, hand_combination)
                    # Print hanya jika ada perubahan pada kartu yang dipilih
                    print(f"Bot memilih kartu 1: {selected_card_1}")
                    print(f"Nilai Kartu Bot {card_value_bot_selected}")
                    print(f"Nilai Kartu Player {card_value_player_selected}")
                    print(f"Kartu yang masih bisa digunakan oleh bot: {available_cards1}")
                    print(f"Kartu yang sudah digunakan oleh bot: {used_cards1}")
                    print(f"Sisa Kartu User : {used_user_cards}")
                    
                    
                # Memastikan jika kartu pemain tidak berubah, kita tetap update status kartu yang digunakan dan tersedia
                else:
                    # Update kartu yang sudah digunakan dan yang masih tersedia
                    available_cards1 = [card for card in bot_cards1 if card not in used_cards1]

                    # Cek apakah ada perubahan pada available_cards atau used_cards sebelum mencetak
                    if available_cards1 != last_printed_available_cards or used_cards1 != last_printed_used_cards:
                        print(f"Kartu yang masih bisa digunakan oleh bot: {available_cards1}")
                        print(f"Kartu yang sudah digunakan oleh bot: {used_cards1}")

                        # Simpan status terakhir untuk perbandingan pada iterasi berikutnya
                        last_printed_available_cards = available_cards1.copy()
                        last_printed_used_cards = used_cards1.copy()
            elif combination_type == "pair":
                if hand_combination != last_player_hand:  # Hanya memilih kartu jika kartu pemain berubah
                    last_player_hand = hand_combination
                    
                    # Dapatkan nilai dari pair yang ada
                    card_value_pair_bot = card_value_from_text_bulk(pair)
                    card_value_pair_player = card_value_from_text_bulk(hand_combination)
                    
                    print("Value pair player", card_value_pair_player)
                    print("Value pair bot", card_value_pair_bot)
                    
                    score_pair = update_scores(card_value_pair_player, card_value_pair_bot)
                    print("Kombinasi Pair:", pair)
                    
                    # Pindahkan kartu pair yang digunakan ke dalam used_cards1 dan update available_cards1
                    # Misalkan pair berisi list kartu yang membentuk pair
                    for card in pair:
                        if card in available_cards1:
                            available_cards1.remove(card)  # Hapus kartu yang digunakan dari available_cards1
                            used_cards1.append(card)  # Tambahkan kartu yang digunakan ke used_cards1
                    
                    # Jika kamu ingin melihat kartu yang sudah digunakan dan yang masih tersedia
                    print(f"Kartu yang masih bisa digunakan oleh bot: {available_cards1}")
                    print(f"Kartu yang sudah digunakan oleh bot: {used_cards1}")

                # Jika kartu pemain tidak berubah, cukup perbarui status kartu yang digunakan dan tersedia
                else:
                    # Update kartu yang sudah digunakan dan yang masih tersedia
                    available_cards1 = [card for card in bot_cards1 if card not in used_cards1]

                    # Cek apakah ada perubahan pada available_cards atau used_cards sebelum mencetak
                    if available_cards1 != last_printed_available_cards or used_cards1 != last_printed_used_cards:
                        print(f"Kartu yang masih bisa digunakan oleh bot: {available_cards1}")
                        print(f"Kartu yang sudah digunakan oleh bot: {used_cards1}")

                        # Simpan status terakhir untuk perbandingan pada iterasi berikutnya
                        last_printed_available_cards = available_cards1.copy()
                        last_printed_used_cards = used_cards1.copy()
            elif combination_type == "full house":
                if hand_combination != last_player_hand:  # Hanya memilih kartu jika kartu pemain berubah
                    last_player_hand = hand_combination
                    
                    # Dapatkan nilai dari full house
                    card_value_full_house_bot = card_value_from_text_bulk(full_house)
                    card_value_full_house_player = card_value_from_text_bulk(hand_combination)
                    
                    print("Value full house player", card_value_full_house_player)
                    print("Value full house bot", card_value_full_house_bot)
                    
                    score_full_house = update_scores(card_value_full_house_player, card_value_full_house_bot)
                    print("Kombinasi Full House:", full_house)
                    
                    # Pindahkan kartu full house yang digunakan ke dalam used_cards1 dan update available_cards1
                    for card in full_house:
                        if card in available_cards1:
                            available_cards1.remove(card)  # Hapus kartu yang digunakan dari available_cards1
                            used_cards1.append(card)  # Tambahkan kartu yang digunakan ke used_cards1
                    
                    print(f"Kartu yang masih bisa digunakan oleh bot: {available_cards1}")
                    print(f"Kartu yang sudah digunakan oleh bot: {used_cards1}")

            elif combination_type == "three of a kind":
                if hand_combination != last_player_hand:  # Hanya memilih kartu jika kartu pemain berubah
                    last_player_hand = hand_combination
                    
                    # Dapatkan nilai dari three of a kind
                    card_value_three_of_a_kind_bot = card_value_from_text_bulk(three_of_a_kind)
                    card_value_three_of_a_kind_player = card_value_from_text_bulk(hand_combination)
                    
                    print("Value three of a kind player", card_value_three_of_a_kind_player)
                    print("Value three of a kind bot", card_value_three_of_a_kind_bot)
                    
                    score_three_of_a_kind = update_scores(card_value_three_of_a_kind_player, card_value_three_of_a_kind_bot)
                    print("Kombinasi Three of a Kind:", three_of_a_kind)
                    
                    # Pindahkan kartu three of a kind yang digunakan ke dalam used_cards1 dan update available_cards1
                    for card in three_of_a_kind:
                        if card in available_cards1:
                            available_cards1.remove(card)  # Hapus kartu yang digunakan dari available_cards1
                            used_cards1.append(card)  # Tambahkan kartu yang digunakan ke used_cards1
                    
                    print(f"Kartu yang masih bisa digunakan oleh bot: {available_cards1}")
                    print(f"Kartu yang sudah digunakan oleh bot: {used_cards1}")

            elif combination_type == "straight":
                if hand_combination != last_player_hand:  # Hanya memilih kartu jika kartu pemain berubah
                    last_player_hand = hand_combination
                    
                    # Dapatkan nilai dari straight
                    card_value_straight_bot = card_value_from_text_bulk(straight)
                    card_value_straight_player = card_value_from_text_bulk(hand_combination)
                    
                    print("Value straight player", card_value_straight_player)
                    print("Value straight bot", card_value_straight_bot)
                    
                    score_straight = update_scores(card_value_straight_player, card_value_straight_bot)
                    print("Kombinasi Straight:", straight)
                    
                    # Pindahkan kartu straight yang digunakan ke dalam used_cards1 dan update available_cards1
                    for card in straight:
                        if card in available_cards1:
                            available_cards1.remove(card)  # Hapus kartu yang digunakan dari available_cards1
                            used_cards1.append(card)  # Tambahkan kartu yang digunakan ke used_cards1
                    
                    print(f"Kartu yang masih bisa digunakan oleh bot: {available_cards1}")
                    print(f"Kartu yang sudah digunakan oleh bot: {used_cards1}")

            elif combination_type == "two pair":
                if hand_combination != last_player_hand:  # Hanya memilih kartu jika kartu pemain berubah
                    last_player_hand = hand_combination
                    
                    # Dapatkan nilai dari two pair
                    card_value_two_pair_bot = card_value_from_text_bulk(two_pair)
                    card_value_two_pair_player = card_value_from_text_bulk(hand_combination)
                    
                    print("Value two pair player", card_value_two_pair_player)
                    print("Value two pair bot", card_value_two_pair_bot)
                    
                    score_two_pair = update_scores(card_value_two_pair_player, card_value_two_pair_bot)
                    print("Kombinasi Two Pair:", two_pair)
                    
                    # Pindahkan kartu two pair yang digunakan ke dalam used_cards1 dan update available_cards1
                    for card in two_pair:
                        if card in available_cards1:
                            available_cards1.remove(card)  # Hapus kartu yang digunakan dari available_cards1
                            used_cards1.append(card)  # Tambahkan kartu yang digunakan ke used_cards1
                    
                    print(f"Kartu yang masih bisa digunakan oleh bot: {available_cards1}")
                    print(f"Kartu yang sudah digunakan oleh bot: {used_cards1}")

            elif combination_type == "flush":
                if hand_combination != last_player_hand:  # Hanya memilih kartu jika kartu pemain berubah
                    last_player_hand = hand_combination
                    
                    # Dapatkan nilai dari flush
                    card_value_flush_bot = card_value_from_text_bulk(flush)
                    card_value_flush_player = card_value_from_text_bulk(hand_combination)
                    
                    print("Value flush player", card_value_flush_player)
                    print("Value flush bot", card_value_flush_bot)
                    
                    score_flush = update_scores(card_value_flush_player, card_value_flush_bot)
                    print("Kombinasi Flush:", flush)
                    
                    # Pindahkan kartu flush yang digunakan ke dalam used_cards1 dan update available_cards1
                    for card in flush:
                        if card in available_cards1:
                            available_cards1.remove(card)  # Hapus kartu yang digunakan dari available_cards1
                            used_cards1.append(card)  # Tambahkan kartu yang digunakan ke used_cards1
                    
                    print(f"Kartu yang masih bisa digunakan oleh bot: {available_cards1}")
                    print(f"Kartu yang sudah digunakan oleh bot: {used_cards1}")

            else :
                print("No card Detection")
        # test kartu kombinasi player
        # print(hand_combination)
        # Resize frame untuk tampilan
        frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv2.imshow("Poker Game - User Card Detection", frame_resized)
        # Mengecek input dari thread input
        
        # Menampilkan frame game info
        text_frame = np.zeros((frame.shape[0], 900, 3), dtype=np.uint8)
        cv2.putText(text_frame, "User's Cards:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for i, card in enumerate(user_cards):
            cv2.putText(text_frame, f"{i+1}: {card}", (20, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # print("USER CARDS :", user_cards)

        cv2.putText(text_frame, "Bot's Cards:", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for i, card in enumerate(bot_cards):
            cv2.putText(text_frame, f"{i+1}: {card}", (450, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Game Info", text_frame)

        # Menghentikan aplikasi jika menekan 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Permainan dibatalkan.")