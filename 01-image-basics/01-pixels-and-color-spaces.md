# Piksel dan Ruang Warna

> **Sebelum kita bisa "melihat" seperti komputer, kita perlu memahami bagaimana gambar digital sebenarnya tersusun.**

Dalam bab ini, kita akan mempelajari konsep dasar yang menjadi fondasi seluruh Computer Vision: **piksel** dan **ruang warna**.

---

## 🔲 Apa itu Piksel?

**Piksel** (singkatan dari *Picture Element*) adalah **unit terkecil** dari sebuah gambar digital.

Bayangkan gambar sebagai **mozaik**—ribuan kotak kecil yang masing-masing memiliki satu warna. Setiap kotak itu adalah satu piksel!

![Ilustrasi Piksel: Zoom dari gambar ke grid piksel](./images/03-pixel-grid.png)

### Karakteristik Piksel

| Properti | Penjelasan |
|---|---|
| **Lokasi** | Setiap piksel memiliki koordinat (x, y) |
| **Nilai Warna** | Angka yang merepresentasikan warna piksel |
| **Resolusi** | Jumlah total piksel (contoh: 1920×1080 = 2 juta piksel) |

### Contoh Resolusi Umum

| Resolusi | Nama | Total Piksel |
|---|---|---|
| 640×480 | VGA | 307.200 |
| 1280×720 | HD (720p) | 921.600 |
| 1920×1080 | Full HD (1080p) | 2.073.600 |
| 3840×2160 | 4K UHD | 8.294.400 |

> 💡 **Fun Fact:** Kamera smartphone modern bisa mencapai 108 megapiksel (108 juta piksel)!

---

## 🖼️ Gambar Sebagai Array

Dalam pemrograman, gambar direpresentasikan sebagai **array multidimensi** (matriks angka).

### 🔑 Konsep Penting: Layer/Channel

Perbedaan utama antara gambar berwarna dan hitam-putih adalah **jumlah layer (channel)**:

![Perbandingan Struktur RGB (3 Layer) vs Grayscale (1 Layer)](./images/05-rgb-vs-gray-layers.png)

| Jenis Gambar | Jumlah Layer | Dimensi Array | Penjelasan |
|---|:---:|---|---|
| **Grayscale** | 1 | (Tinggi × Lebar) | Hanya intensitas cahaya |
| **RGB** | 3 | (Tinggi × Lebar × 3) | Red + Green + Blue |
| **RGBA** | 4 | (Tinggi × Lebar × 4) | RGB + Alpha (transparansi) |

> 💡 **Analogi:** Bayangkan gambar RGB seperti **sandwich 3 lapis**—setiap lapis (R, G, B) memiliki informasi berbeda, dan ketika digabungkan menghasilkan warna penuh!

### Gambar Grayscale (Hitam-Putih)
- **2D Array**: (Tinggi × Lebar)
- Setiap nilai: 0 (hitam) sampai 255 (putih)


```
Contoh gambar 3×3 piksel:
┌─────┬─────┬─────┐
│  0  │ 128 │ 255 │
├─────┼─────┼─────┤
│ 100 │ 200 │  50 │
├─────┼─────┼─────┤
│ 255 │  0  │ 128 │
└─────┴─────┴─────┘
```

### Gambar Berwarna (RGB)
- **3D Array**: (Tinggi × Lebar × 3 Channel)
- Setiap piksel memiliki 3 nilai: Red, Green, Blue

```python
# Contoh di Python dengan NumPy
import numpy as np

# Gambar 100x100 piksel, 3 channel (RGB)
gambar = np.zeros((100, 100, 3), dtype=np.uint8)

# Akses satu piksel di posisi (50, 50)
# Format: [Red, Green, Blue]
gambar[50, 50] = [255, 0, 0]  # Piksel merah
```

---

## 🎨 Ruang Warna (Color Spaces)

**Ruang warna** adalah cara sistematis untuk merepresentasikan warna dengan angka. Ada beberapa ruang warna yang umum digunakan.

---

### 1. RGB (Red, Green, Blue)

Ruang warna **paling umum** untuk tampilan digital (monitor, TV, smartphone).

![Model Warna RGB - Pencampuran Warna Additif](./images/01-rgb-model.png)

#### Cara Kerja RGB

Warna dihasilkan dari **pencampuran cahaya** (additive color mixing):

| Warna | R | G | B | Hasil |
|---|:---:|:---:|:---:|---|
| Merah | 255 | 0 | 0 | 🔴 |
| Hijau | 0 | 255 | 0 | 🟢 |
| Biru | 0 | 0 | 255 | 🔵 |
| Kuning | 255 | 255 | 0 | 🟡 (R + G) |
| Cyan | 0 | 255 | 255 | 🔵 (G + B) |
| Magenta | 255 | 0 | 255 | 🟣 (R + B) |
| Putih | 255 | 255 | 255 | ⚪ (Semua) |
| Hitam | 0 | 0 | 0 | ⚫ (Tidak ada) |

#### Kelebihan RGB
- ✅ Native untuk layar digital
- ✅ Mudah dipahami
- ✅ Dukungan luas di semua library

#### Kekurangan RGB
- ❌ Tidak intuitif untuk manipulasi warna
- ❌ Sulit memisahkan *warna* dari *kecerahan*

---

### 2. Grayscale (Skala Abu-abu)

Gambar dengan **satu channel** saja, merepresentasikan intensitas cahaya.

![Konversi dari RGB ke Grayscale](./images/04-grayscale.png)

#### Rentang Nilai

| Nilai | Representasi |
|:---:|---|
| 0 | Hitam (gelap total) |
| 128 | Abu-abu sedang |
| 255 | Putih (terang penuh) |

#### Rumus Konversi RGB → Grayscale

```
Gray = 0.299×R + 0.587×G + 0.114×B
```

> 💡 **Mengapa bobotnya berbeda?** Mata manusia lebih sensitif terhadap cahaya hijau!

#### Kapan Menggunakan Grayscale?

- **Deteksi tepi** (Canny, Sobel)
- **Preprocessing** untuk mempercepat komputasi
- **Pengenalan pola** di mana warna tidak penting

```python
import cv2

# Baca gambar berwarna
gambar_rgb = cv2.imread('foto.jpg')

# Konversi ke grayscale
gambar_gray = cv2.cvtColor(gambar_rgb, cv2.COLOR_BGR2GRAY)

print(gambar_rgb.shape)   # (height, width, 3)
print(gambar_gray.shape)  # (height, width)
```

---

### 3. HSV (Hue, Saturation, Value)

Ruang warna yang lebih **intuitif** untuk manusia dan sangat berguna untuk **segmentasi warna**.

![Perbandingan Model RGB vs HSV](./images/02-rgb-vs-hsv.png)

#### Komponen HSV

| Komponen | Rentang | Arti |
|---|:---:|---|
| **H** (Hue) | 0-179° | Jenis warna (merah, biru, hijau, dll) |
| **S** (Saturation) | 0-255 | Kepekatan warna (0 = abu-abu, 255 = warna penuh) |
| **V** (Value) | 0-255 | Kecerahan (0 = gelap, 255 = terang) |

> ⚠️ **Catatan:** Di OpenCV, Hue hanya 0-179 (bukan 0-359) untuk muat dalam 8-bit!

#### Mengapa HSV Penting untuk CV?

HSV memisahkan **informasi warna (Hue)** dari **pencahayaan (Value)**, sehingga lebih robust terhadap perubahan cahaya!

**Contoh:** Mendeteksi objek berwarna merah

```python
import cv2
import numpy as np

# Baca dan konversi ke HSV
gambar = cv2.imread('apel.jpg')
hsv = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)

# Definisikan rentang warna merah
batas_bawah = np.array([0, 100, 100])
batas_atas = np.array([10, 255, 255])

# Buat mask untuk warna merah
mask = cv2.inRange(hsv, batas_bawah, batas_atas)

# Terapkan mask
hasil = cv2.bitwise_and(gambar, gambar, mask=mask)
```

---

### 4. Ruang Warna Lainnya

| Ruang Warna | Kegunaan |
|---|---|
| **LAB** | Perceptually uniform, bagus untuk perbandingan warna |
| **YCrCb** | Digunakan dalam kompresi video (JPEG, MPEG) |
| **CMYK** | Untuk pencetakan (Cyan, Magenta, Yellow, Key/Black) |

---

## 🔄 Konversi Antar Ruang Warna

OpenCV menyediakan fungsi `cvtColor()` untuk konversi:

```python
import cv2

gambar = cv2.imread('foto.jpg')

# Konversi ke berbagai ruang warna
gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(gambar, cv2.COLOR_BGR2LAB)
rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)  # Matplotlib pakai RGB!
```

> ⚠️ **Penting:** OpenCV membaca gambar dalam format **BGR** (bukan RGB)!

---

## 📊 Perbandingan Ruang Warna

| Ruang Warna | Channel | Kegunaan Utama |
|---|:---:|---|
| **RGB** | 3 | Tampilan, penyimpanan gambar |
| **Grayscale** | 1 | Preprocessing, deteksi tepi |
| **HSV** | 3 | Segmentasi warna, tracking objek |
| **LAB** | 3 | Color correction, histogram equalization |

---

## 💻 Latihan Praktis

Coba jalankan kode berikut untuk melihat perbedaan ruang warna:

```python
import cv2
import matplotlib.pyplot as plt

# Baca gambar
img = cv2.imread('foto.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Konversi ke berbagai ruang warna
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Tampilkan
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original (RGB)')

axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('Grayscale')

axes[1, 0].imshow(hsv[:, :, 0], cmap='hsv')
axes[1, 0].set_title('Hue Channel')

axes[1, 1].imshow(hsv[:, :, 2], cmap='gray')
axes[1, 1].set_title('Value Channel')

plt.tight_layout()
plt.show()
```

---

## 📚 Rangkuman

| Konsep | Penjelasan |
|---|---|
| **Piksel** | Unit terkecil gambar, memiliki koordinat dan nilai warna |
| **Resolusi** | Jumlah piksel total (lebar × tinggi) |
| **RGB** | Ruang warna additif (Merah, Hijau, Biru) |
| **Grayscale** | Satu channel, nilai 0-255 (hitam ke putih) |
| **HSV** | Hue (warna), Saturation (kepekatan), Value (kecerahan) |
| **cvtColor()** | Fungsi OpenCV untuk konversi ruang warna |

---

## ➡️ Langkah Selanjutnya

Sekarang Anda memahami bagaimana gambar tersusun dari piksel dan bagaimana warna direpresentasikan. Selanjutnya, kita akan mempelajari **Operasi Dasar pada Gambar** seperti resize, crop, dan rotate!

[Lanjut ke: Operasi Citra Dasar →](./02-image-operations.md)
