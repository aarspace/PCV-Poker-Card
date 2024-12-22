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