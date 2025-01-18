import cv2
import numpy as np

def crop_to_square(image):
    """
    Görüntüyü kare boyuta kırpar (merkezi baz alarak).
    """
    height, width = image.shape[:2]
    if height == width:
        return image  # Zaten kare ise kırpma yapma

    # Kırpma işlemi
    if height > width:
        offset = (height - width) // 2
        cropped_image = image[offset:offset + width, :]
    else:
        offset = (width - height) // 2
        cropped_image = image[:, offset:offset + height]

    return cropped_image


# Görüntüyü yükle
input_image_path = "image1.jpg"  # Orijinal görüntü dosyasının yolu
output_cropped_image_path = "kare_kirpilan_goruntu.jpg"  # Kırpılmış görüntünün kaydedileceği yol

# Görüntüyü oku
image = cv2.imread(input_image_path)

if image is None:
    print("Görüntü yüklenemedi. Lütfen dosya yolunu kontrol edin.")
else:
    # Görüntüyü kare boyuta kırp
    cropped_image = crop_to_square(image)

    # Kırpılmış görüntüyü 256x256 boyutuna yeniden boyutlandır
    resized_image = cv2.resize(cropped_image, (255, 255))

    # Kırpılmış ve yeniden boyutlandırılmış görüntüyü kaydet
    cv2.imwrite(output_cropped_image_path, resized_image)
    print(f"Kırpılmış ve yeniden boyutlandırılmış görüntü '{output_cropped_image_path}' olarak kaydedildi.")

output_face_path = "kesilen_yuz.jpg"  # Kesilen yüzün kaydedileceği dosya yolu
output_blackout_path = "siyah_yuz.jpg"  # Siyah kareyle düzenlenmiş görüntünün kaydedileceği dosya yolu


# Görüntüyü yükle
image = cv2.imread(output_cropped_image_path)


# OpenCV'nin yerleşik Haar Cascade sınıflandırıcısını kullanarak yüz tespiti
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Gri tonlamaya çevir
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Yüzleri tespit et
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
image_deneme=[]
sifre=''
# Yüzleri işle
for (x, y, w, h) in faces:
    # Tespit edilen yüzün konum bilgilerini yazdır
    print(f"Yüz konumu: X={x}, Y={y}, Genişlik={w}, Yükseklik={h}")
    sifre=f"{x},{y},{w},{h};"
    # Yüzü görüntüden kes ve kaydet
    face = image[y:y+h, x:x+w]
    cv2.imwrite(output_face_path, face)
    print(f"Kesilen yüz '{output_face_path}' olarak kaydedildi.")

    # Yüz bölgesini siyaha boya
    image[y:y+h, x:x+w] = (0, 0, 0)
    image_deneme= image[y:y+h, x:x+w]


# Düzenlenmiş görüntüyü kaydet

cv2.imwrite(output_blackout_path, image)
print(f"Siyah kareyle düzenlenmiş görüntü '{output_blackout_path}' olarak kaydedildi.")

print(sifre)

#-------

import cv2
import numpy as np

def vigenere_encrypt_pixel_values(image, key):
    """
    Görüntü piksel değerlerini Vigenère şifreleme ile şifreler.
    """
    encrypted_image = np.zeros_like(image)
    key_length = len(key)

    # Şifreleme işlemi (her kanal için ayrı ayrı)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(3):  # BGR kanalları
                original_value = image[y, x, c]
                key_value = ord(key[(y * image.shape[1] + x) % key_length]) % 255
                encrypted_value = (original_value + key_value) % 255
                encrypted_image[y, x, c] = encrypted_value

    return encrypted_image


def vigenere_encrypt_pixel_positions(image, key):
    """
    Görüntüdeki piksel konumlarını Vigenère şifreleme ile şifreler.
    Şifrelenmiş konumlar görüntü sınırları içinde kalmalıdır.
    """
    encrypted_image = np.zeros_like(image)
    key_length = len(key)

    h, w, _ = image.shape

    for y in range(h):
        for x in range(w):
            key_value_y = ord(key[(y * w + x) % key_length]) % h
            key_value_x = ord(key[(x * h + y) % key_length]) % w

            # Yeni şifrelenmiş konum
            new_y = (y + key_value_y) % h
            new_x = (x + key_value_x) % w

            # Şifrelenmiş konumlara orijinal piksel değerini ata
            encrypted_image[new_y, new_x] = image[y, x]

    return encrypted_image


# Kesilen yüzü yükle
input_face_path = "kesilen_yuz.jpg"  # Kesilen yüz dosyasının yolu
output_pixel_value_encrypted = "sifreli_piksel_deger.jpg"
output_pixel_position_encrypted = "sifreli_piksel_konum.jpg"

# Görüntüyü yükle
face_image = cv2.imread(input_face_path)

# Şifreleme anahtarı
vigenere_key = "SELCUKUNIVERSITESI"

# Piksel değerlerine Vigenère şifreleme uygula
encrypted_pixel_values = vigenere_encrypt_pixel_values(face_image, vigenere_key)
cv2.imwrite(output_pixel_value_encrypted, encrypted_pixel_values)
print(f"Piksel değerlerine şifreleme uygulanmış görüntü '{output_pixel_value_encrypted}' olarak kaydedildi.")

# Piksel konumlarına Vigenère şifreleme uygula
encrypted_pixel_positions = vigenere_encrypt_pixel_positions(encrypted_pixel_values, vigenere_key)
cv2.imwrite(output_pixel_position_encrypted, encrypted_pixel_positions)
print(f"Piksel konumlarına şifreleme uygulanmış görüntü '{output_pixel_position_encrypted}' olarak kaydedildi.")


#-------

import cv2
import numpy as np

def place_encrypted_image_on_original(original_image, encrypted_image, position):
    """
    Şifrelenmiş görüntüyü, orijinal görüntü üzerinde belirtilen konuma yerleştirir.
    """
    y, x, h, w = position  # Yüzün konumu: y, x, yükseklik, genişlik
    y, x, h, w = int(y), int(x), int(h), int(w)
    encrypted_resized = cv2.resize(encrypted_image, (w,h))  # Şifreli görüntüyü yüz boyutuna getir

    # Orijinal görüntüye yerleştir
    result_image = original_image.copy()
    result_image[x:x+w,y:y+h] = encrypted_resized

    return result_image


# Gerekli dosyaları yükle
original_image_path = "kare_kirpilan_goruntu.jpg"  # Orijinal görüntünün yolu
encrypted_image_path = "sifreli_piksel_konum.jpg"  # Şifrelenmiş görüntünün yolu
output_final_image_path = "sonuc_goruntu.jpg"  # Sonuç görüntüsünün kaydedileceği yol

# Görüntüleri yükle
original_image = cv2.imread(original_image_path)
encrypted_image = cv2.imread(encrypted_image_path)

if original_image is None or encrypted_image is None:
    print("Görüntü yüklenemedi. Lütfen dosya yollarını kontrol edin.")
else:
    # Tespit edilen yüzün pozisyonu (örnek değerler, tespit edilen pozisyonla değiştirilmeli)

    split_data = sifre.split(";")  # Virgüle göre böl
    print(split_data[0])
    split_data2=split_data[0].split(",")
    face_position = (split_data2[0],split_data2[1], split_data2[2], split_data2[3])
    print(face_position)
    # Şifreli görüntüyü orijinal görüntüye yerleştir
    final_image = place_encrypted_image_on_original(original_image, encrypted_image, face_position)

    # Sonuç görüntüsünü kaydet
    cv2.imwrite(output_final_image_path, final_image)
    print(f"Sonuç görüntüsü '{output_final_image_path}' olarak kaydedildi.")





from PIL import Image
import numpy as np

def encode_text_to_image(image_path, output_path, text):
    # Görüntüyü yükle
    img = Image.open(image_path)
    img = img.convert("RGB")
    data = np.array(img, dtype=np.uint8)

    # Metni bitlere çevir
    binary_text = ''.join(format(ord(char), '08b') for char in text)
    binary_text += '1111111111111110'  # Bitiş işareti
    print(binary_text)
    # Metni görüntüye gizle
    binary_index = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(3):  # RGB kanalları
                if binary_index < len(binary_text):
                    # Piksel değerini 0-255 arasında tutacak şekilde güncelle
                    pixel_value = int(data[i, j, k])
                    data[i, j, k] = (pixel_value & ~1) | int(binary_text[binary_index])
                    binary_index += 1

    # Yeni görüntüyü kaydet
    encoded_img = Image.fromarray(data)
    encoded_img.save(output_path)
    print(f"Metin başarıyla şifrelendi ve {output_path} dosyasına kaydedildi.")


output_path = "Steganografi_sonuc.png"  # Çıktı görüntüsü
encode_text_to_image("sonuc_goruntu.jpg", output_path, sifre)

#---------

goruntuV2 = "Steganografi_sonuc.png"  # Kesilen yüz dosyasının yolu
output_pixel_value_encrypted2 = "sifreli_piksel_deger_v2.png"
output_pixel_position_encrypted2 = "sifreli_piksel_konum_v2.png"

goruntuv2_sifreli = cv2.imread(goruntuV2)

# Şifreleme anahtarı
vigenere_key = "SELCUKUNIVERSITESI"

# Piksel değerlerine Vigenère şifreleme uygula
encrypted_pixel_values2 = vigenere_encrypt_pixel_values(goruntuv2_sifreli, vigenere_key)
cv2.imwrite(output_pixel_value_encrypted2, encrypted_pixel_values2)
print(f"Piksel değerlerine şifreleme uygulanmış görüntü '{output_pixel_value_encrypted2}' olarak kaydedildi.")

# Piksel konumlarına Vigenère şifreleme uygula
encrypted_pixel_positions2 = vigenere_encrypt_pixel_positions(encrypted_pixel_values2, vigenere_key)
cv2.imwrite(output_pixel_position_encrypted2, encrypted_pixel_positions2)
print(f"Piksel konumlarına şifreleme uygulanmış görüntü '{output_pixel_position_encrypted2}' olarak kaydedildi.")

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
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Gri tonlamada okuma
    return image


# 3. Kaotik Diziyi Yer Değiştirme Pozisyonu Matrisi İçin Kullanma
def create_position_matrix(image, logistic_sequence):
    height, width = image.shape
    num_pixels = height * width

    # Logistic harita çıktısını, görüntü boyutuna göre yeniden şekillendir
    if len(logistic_sequence) < num_pixels:
        # Eğer kaotik dizi görüntü boyutuna eşit değilse, diziyi çoğaltarak genişletiyoruz
        repeat_factor = (num_pixels // len(logistic_sequence)) + 1
        logistic_sequence = np.tile(logistic_sequence, repeat_factor)[:num_pixels]

    positions = np.argsort(logistic_sequence)  # Logistic dizisini sıralayıp pozisyonları alıyoruz

    # Yeni pozisyonlar ile piksel sırasını karıştırma
    shuffled_image = image.flatten()[positions].reshape(height, width)
    return shuffled_image


# 4. Şifreli Görüntüyü Gösterme
def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()


# 5. Şifreli Görüntüyü Kaydetme
def save_image(image, output_path):
    cv2.imwrite(output_path, image)


# Ana Fonksiyon
def encrypt_image(image_path, output_path, r=3.99, iterations=65025):  # iterations'ı görüntü boyutuna göre ayarladık
    # Başlangıç değeri ve parametre
    x_initial = 0.5  # Logistic harita için başlangıç değeri
    logistic_sequence = logistic_map(x_initial, r, iterations)

    # Görüntüyü oku
    image = read_image(image_path)

    # Görüntüyü şifrele
    encrypted_image = create_position_matrix(image, logistic_sequence)

    # Şifreli görüntüyü göster
    show_image(encrypted_image)

    # Şifreli görüntüyü kaydet
    save_image(encrypted_image, output_path)


# Şifreleme işlemi için bir örnek görüntü yolu

image_path = 'sifreli_piksel_konum_v2.png'
output_path = 'SONUC.png'
encrypt_image(image_path, output_path)

