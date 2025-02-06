import numpy as np
import cv2
import matplotlib.pyplot as plt


# 1. Kaotik Harita Başlangıç Değerlerini Hesaplama
def logistic_map(x, r, iterations):
    sequence = []
    for _ in range(iterations):
        x = r * x * (1 - x)
        sequence.append(x)
    return sequence


# 2. Görüntüyü Okuma
def read_image(image_path):
    image = cv2.imread(image_path)  # Renkli okuma (BGR formatında)
    return image


# 3. Kaotik Diziyi Yer Değiştirme Pozisyonu Matrisi İçin Kullanma
def create_position_matrix(image, logistic_sequence):
    height, width, channels = image.shape
    num_pixels = height * width

    shuffled_image = np.zeros_like(image)

    for c in range(channels):
        channel_data = image[:, :, c].flatten()

        # Logistic harita çıktısını görüntü boyutuna göre genişlet
        if len(logistic_sequence) < num_pixels:
            repeat_factor = (num_pixels // len(logistic_sequence)) + 1
            logistic_sequence = np.tile(logistic_sequence, repeat_factor)[:num_pixels]

        positions = np.argsort(logistic_sequence)  # Pozisyonları al
        shuffled_channel = channel_data[positions].reshape(height, width)
        shuffled_image[:, :, c] = shuffled_channel

    return shuffled_image


# 4. Şifreli Görüntüyü Çözme
def decrypt_position_matrix(image, logistic_sequence):
    height, width, channels = image.shape
    num_pixels = height * width

    decrypted_image = np.zeros_like(image)

    for c in range(channels):
        channel_data = image[:, :, c].flatten()

        if len(logistic_sequence) < num_pixels:
            repeat_factor = (num_pixels // len(logistic_sequence)) + 1
            logistic_sequence = np.tile(logistic_sequence, repeat_factor)[:num_pixels]

        positions = np.argsort(logistic_sequence)
        original_positions = np.argsort(positions)  # Şifre çözme için ters sıralama
        decrypted_channel = channel_data[original_positions].reshape(height, width)
        decrypted_image[:, :, c] = decrypted_channel

    return decrypted_image


# 5. Görüntüyü Gösterme
def show_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye çevir
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()


# 6. Görüntüyü Kaydetme
def save_image(image, output_path):
    cv2.imwrite(output_path, image)



# 8. Şifre Çözme Fonksiyonu
def decrypt_image(image_path, output_path, r=3.99, iterations=65025):
    x_initial = 0.5  # Logistic harita için başlangıç değeri
    logistic_sequence = logistic_map(x_initial, r, iterations)

    image = read_image(image_path)
    decrypted_image = decrypt_position_matrix(image, logistic_sequence)

    show_image(decrypted_image)
    save_image(decrypted_image, output_path)


# Şifreleme ve Şifre Çözme İşlemleri
image_path = 'SONUC.png'
encrypted_output_path = 'SONUC.png'
decrypted_output_path = 'decrypt1.png'

decrypt_image(encrypted_output_path, decrypted_output_path)

import cv2
import numpy as np


def vigenere_decrypt_pixel_positions(encrypted_image, key):
    decrypted_image = np.zeros_like(encrypted_image)
    height, width, channels = encrypted_image.shape
    key_length = len(key)

    key_indices = [ord(k) % width for k in key]

    for y in range(height):
        for x in range(width):
            new_x = (x - key_indices[y % key_length]) % width
            decrypted_image[y, new_x] = encrypted_image[y, x]

    return decrypted_image


def vigenere_decrypt_pixel_values(encrypted_image, key):
    decrypted_image = np.zeros_like(encrypted_image)
    height, width, channels = encrypted_image.shape
    key_length = len(key)

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                decrypted_image[y, x, c] = (encrypted_image[y, x, c] - ord(key[(x + y) % key_length])) % 255

    return decrypted_image


# Şifreli görüntüleri yükleme
encrypted_pixel_positions2 = cv2.imread("decrypt1.png")

decrypted_pixel_positions = vigenere_decrypt_pixel_positions(encrypted_pixel_positions2, "SELCUKUNIVERSITESI")
cv2.imwrite("cozulen_piksel_konum_v2.png", decrypted_pixel_positions)
print("Piksel konumları çözüldü ve 'cozulen_piksel_konum_v2.png' olarak kaydedildi.")

decrypted_pixel_values = vigenere_decrypt_pixel_values(decrypted_pixel_positions, "SELCUKUNIVERSITESI")
cv2.imwrite("cozulen_goruntu_v2.png", decrypted_pixel_values)
print("Piksel değerleri çözüldü ve 'cozulen_goruntu_v2.png' olarak kaydedildi.")
