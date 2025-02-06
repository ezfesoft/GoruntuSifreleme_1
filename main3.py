import cv2
import numpy as np
import time
from PIL import Image
from matplotlib import pyplot as plt

# Başlangıç zamanını al
start_time = time.time()

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

# 3. Kaotik Diziyi Kullanarak Görüntüyü Çözme
def decrypt_image(image, logistic_sequence):
    height, width, channels = image.shape
    num_pixels = height * width

    decrypted_image = np.zeros_like(image)

    for c in range(channels):
        channel_data = image[:, :, c].flatten()

        # Logistic harita çıktısını görüntü boyutuna göre genişlet
        if len(logistic_sequence) < num_pixels:
            repeat_factor = (num_pixels // len(logistic_sequence)) + 1
            logistic_sequence = np.tile(logistic_sequence, repeat_factor)[:num_pixels]

        positions = np.argsort(logistic_sequence)  # Pozisyonları al
        reverse_positions = np.argsort(positions)  # Ters pozisyonları al
        decrypted_channel = channel_data[reverse_positions].reshape(height, width)
        decrypted_image[:, :, c] = decrypted_channel

    return decrypted_image

# 4. Şifreli Görüntüyü Gösterme
def show_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye çevir
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

# 5. Şifreli Görüntüyü Kaydetme
def save_image(image, output_path):
    cv2.imwrite(output_path, image)

# Ana Fonksiyon
def decrypt_image_from_path(image_path, output_path, r=3.99, iterations=65025):
    x_initial = 0.5  # Logistic harita için başlangıç değeri
    logistic_sequence = logistic_map(x_initial, r, iterations)

    # Görüntüyü oku
    image = read_image(image_path)

    # Görüntüyü çöz
    decrypted_image = decrypt_image(image, logistic_sequence)

    # Çözülmüş görüntüyü göster
    show_image(decrypted_image)

    # Çözülmüş görüntüyü kaydet
    save_image(decrypted_image, output_path)
# Şifre çözme işlemi için bir örnek görüntü yolu
image_path = 'SONUC.png'
output_path = 'decrypt1.png'
decrypt_image_from_path(image_path, output_path)
# 7. Vigenère Şifrelemesini Çözme

def vigenere_decrypt_pixel_positions(image, key):
    """
    Piksel konumları Vigenère şifre çözme yöntemiyle orijinal haline getirilir.
    """
    decrypted_image = np.zeros_like(image)
    key_length = len(key)
    h, w, _ = image.shape

    for y in range(h):
        for x in range(w):
            key_value_y = ord(key[(y * w + x) % key_length]) % h
            key_value_x = ord(key[(x * h + y) % key_length]) % w

            # Orijinal konumu geri bul
            original_y = (y - key_value_y) % h
            original_x = (x - key_value_x) % w

            decrypted_image[original_y, original_x] = image[y, x]

    return decrypted_image


def vigenere_decrypt_pixel_values(image, key):
    """
    Piksel değerlerini Vigenère şifre çözme yöntemiyle orijinal haline getirir.
    """
    decrypted_image = np.zeros_like(image, dtype=np.uint8)
    key_length = len(key)
    h, w, _ = image.shape

    for y in range(h):
        for x in range(w):
            for c in range(3):  # BGR kanalları için döngü
                encrypted_value = int(image[y, x, c])  # `int` dönüşümü ile taşma önlenir
                key_value = ord(key[(y * w + x) % key_length]) % 255
                original_value = (encrypted_value - key_value + 255) % 255  # Taşma önleme

                decrypted_image[y, x, c] = np.uint8(original_value)

    return decrypted_image


# Şifre çözme işlemi için bir örnek görüntü yolu
encrypted_image_path = 'decrypt1.png'
output_decrypted_image_path = 'decrypt2.png'
output_decrypted_image_path2 = 'decrypt3.png'

# Görüntüyü yükle
encrypted_image = cv2.imread(encrypted_image_path)

# Şifreleme anahtarı
vigenere_key = "SELCUKUNIVERSITESI"

# Piksel konumlarını çöz
decrypted_pixel_positions = vigenere_decrypt_pixel_positions(encrypted_image, vigenere_key)
cv2.imwrite(output_decrypted_image_path, decrypted_pixel_positions)
# Piksel değerlerini çöz
decrypted_image = vigenere_decrypt_pixel_values(decrypted_pixel_positions, vigenere_key)

# Çözülmüş görüntüyü kaydet
cv2.imwrite(output_decrypted_image_path2, decrypted_image)
print(f"Çözülmüş görüntü '{output_decrypted_image_path2}' olarak kaydedildi.")




# 6. Steganografi ile Gizlenmiş Metni Çıkarma
def decode_text_from_image(image_path):
    # Görüntüyü yükle
    img = Image.open(image_path)
    img = img.convert("RGB")
    data = np.array(img, dtype=np.uint8)

    # Metni çıkarmak için bitleri topla
    binary_text = ''
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(3):  # RGB kanalları
                binary_text += str(data[i, j, k] & 1)
                if binary_text[-16:] == '1111111111111110':  # Bitiş işareti
                    binary_text = binary_text[:-16]  # Bitiş işaretini kaldır
                    # Binary metni ASCII'ye çevir
                    text = ''.join(chr(int(binary_text[i:i+8], 2)) for i in range(0, len(binary_text), 8))
                    return text
    return ""

# Steganografi ile gizlenmiş metni çıkar
decoded_text = decode_text_from_image(output_path)
print(f"Çözülen Metin: {decoded_text}")


"""
# 8. Yüzü Orijinal Görüntüye Yerleştirme
def place_face_on_original(original_image, face_image, position):
    
    y, x, h, w = position  # Yüzün konumu: y, x, yükseklik, genişlik
    y, x, h, w = int(y), int(x), int(h), int(w)
    face_resized = cv2.resize(face_image, (w, h))  # Yüzü orijinal boyutuna getir

    # Orijinal görüntüye yerleştir
    result_image = original_image.copy()
    result_image[y:y+h, x:x+w] = face_resized

    return result_image

# Gerekli dosyaları yükle
original_image_path = "kare_kirpilan_goruntu.jpg"  # Orijinal görüntünün yolu
face_image_path = "kesilen_yuz.jpg"  # Kesilen yüzün yolu
output_final_image_path = "orijinal_goruntu_geri_yuklendi.jpg"  # Sonuç görüntüsünün kaydedileceği yol

# Görüntüleri yükle
original_image = cv2.imread(original_image_path)
face_image = cv2.imread(face_image_path)

if original_image is None or face_image is None:
    print("Görüntü yüklenemedi. Lütfen dosya yollarını kontrol edin.")
else:
    # Tespit edilen yüzün pozisyonu (örnek değerler, tespit edilen pozisyonla değiştirilmeli)
    split_data = decoded_text.split(";")  # Virgüle göre böl
    split_data2 = split_data[0].split(",")
    face_position = (int(split_data2[0]), int(split_data2[1]), int(split_data2[2]), int(split_data2[3]))

    # Yüzü orijinal görüntüye yerleştir
    final_image = place_face_on_original(original_image, face_image, face_position)

    # Sonuç görüntüsünü kaydet
    cv2.imwrite(output_final_image_path, final_image)
    print(f"Yüz orijinal görüntüye yerleştirildi ve '{output_final_image_path}' olarak kaydedildi.")
"""
end_time = time.time()

# Geçen süreyi hesapla
elapsed_time = end_time - start_time

# Geçen süreyi yazdır
print(f"Geçen süre: {elapsed_time} saniye")