import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Fungsi untuk membaca tangan dan mengenali tangan kanan atau kiri serta mendeteksi jumlah jari yang diangkat
def read_hands(frame):
    # Ubah gambar menjadi BGR yang diharapkan oleh MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses deteksi tangan menggunakan MediaPipe Hands
    results = mp_hands.process(image_rgb)

    # Tampilkan landmark tangan yang terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Mengenali tangan kanan atau kiri
            if hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x:
                handedness = "Kanan"
            else:
                handedness = "Kiri"

            # Tampilkan kotak di sekitar tangan
            bbox = calc_bounding_box(hand_landmarks, frame.shape[1], frame.shape[0])
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)

            # Tampilkan landmark tangan
            for landmark in mp.solutions.hands.HandLandmark:
                x = int(hand_landmarks.landmark[landmark].x * frame.shape[1])
                y = int(hand_landmarks.landmark[landmark].y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Tampilkan garis yang menghubungkan titik landmark
            connections = mp.solutions.hands.HAND_CONNECTIONS
            for connection in connections:
                start = connection[0]
                end = connection[1]
                start_x = int(hand_landmarks.landmark[start].x * frame.shape[1])
                start_y = int(hand_landmarks.landmark[start].y * frame.shape[0])
                end_x = int(hand_landmarks.landmark[end].x * frame.shape[1])
                end_y = int(hand_landmarks.landmark[end].y * frame.shape[0])
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

            # Tampilkan teks keterangan tangan
            text_pos = (bbox[0][0], bbox[0][1] - 10)
            text = f"Tangan: {handedness}"
            cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

# Fungsi untuk menghitung kotak pembatas di sekitar tangan
def calc_bounding_box(hand_landmarks, width, height):
    x = []
    y = []
    for landmark in hand_landmarks.landmark:
        x.append(int(landmark.x * width))
        y.append(int(landmark.y * height))
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    return ((min_x, min_y), (max_x, max_y))

# Fungsi utama
def main():
    # Buka webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Baca setiap frame dari webcam
        ret, frame = cap.read()

        # Jika frame berhasil dibaca
        if ret:
            # Proses frame untuk membaca tangan
            frame_with_hands = read_hands(frame)

            # Tampilkan frame dengan landmark tangan, kotak, dan keterangan tangan
            cv2.imshow('Hand Reading', frame_with_hands)

        # Hentikan program dengan menekan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Tutup webcam dan jendela tampilan
    cap.release()
    cv2.destroyAllWindows()

    # Hentikan MediaPipe Hands
    mp_hands.close()

# Jalankan program utama
if __name__ == '__main__':
    main()
