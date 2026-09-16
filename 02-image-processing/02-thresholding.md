# Thresholding

> **Thresholding adalah teknik untuk mengubah gambar menjadi biner (hitam-putih), memisahkan objek dari latar belakang.**

Thresholding adalah salah satu teknik segmentasi paling sederhana namun sangat powerful. Dengan thresholding, kita dapat dengan mudah memisahkan foreground (objek) dari background.

---

## 🎯 Konsep Dasar Thresholding

Thresholding bekerja dengan **membandingkan setiap piksel dengan nilai ambang batas (threshold)**:

```
JIKA piksel > threshold:
    piksel_baru = PUTIH (255)
JIKA TIDAK:
    piksel_baru = HITAM (0)
```

### Visualisasi Konsep

![Konsep Dasar Thresholding](./images/04-threshold-concept.png)

**Penjelasan Piksel per Piksel:**

![Cara Kerja Thresholding](./images/09-pixel-by-pixel.png)

**Rumus Threshold (T=127):**

| Kondisi | Hasil |
|---------|-------|
| Jika piksel > 127 | → 255 (PUTIH) ✓ |
| Jika piksel ≤ 127 | → 0 (HITAM) ✗ |

**Contoh Perhitungan:**

| Nilai Asli | Perbandingan | Hasil |
|:----------:|:------------:|:-----:|
| 50 | 50 ≤ 127 | 0 (Hitam) ✗ |
| 100 | 100 ≤ 127 | 0 (Hitam) ✗ |
| 150 | 150 > 127 | 255 (Putih) ✓ |
| 200 | 200 > 127 | 255 (Putih) ✓ |
| 250 | 250 > 127 | 255 (Putih) ✓ |

---

## 📊 Jenis-jenis Thresholding

OpenCV menyediakan beberapa metode thresholding:

| Tipe | Flag OpenCV | Penjelasan |
|---|---|---|
| **Binary** | `cv2.THRESH_BINARY` | Di atas threshold → putih, sisanya hitam |
| **Binary Inv** | `cv2.THRESH_BINARY_INV` | Kebalikan Binary |
| **Trunc** | `cv2.THRESH_TRUNC` | Di atas threshold → dipotong ke threshold |
| **To Zero** | `cv2.THRESH_TOZERO` | Di bawah threshold → jadi 0 |
| **To Zero Inv** | `cv2.THRESH_TOZERO_INV` | Di atas threshold → jadi 0 |

---

## 1️⃣ Simple Thresholding

Metode paling dasar dengan **nilai threshold tetap**.

### Kode Python

```python
import cv2

# Baca gambar dalam grayscale
img = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)

# Simple threshold
# Return: (threshold_value, thresholded_image)
thresh_value = 127
ret, thresh_binary = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)

# Tipe lainnya
ret, thresh_binary_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh_trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh_tozero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh_tozero_inv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
```

### Visualisasi Hasil Tiap Tipe

![5 Jenis Thresholding](./images/05-threshold-types-detail.png)

### Kapan Menggunakan Simple Thresholding?

✅ **Cocok untuk:**
- Gambar dengan pencahayaan merata
- Latar belakang dan objek kontras tinggi
- Preprocessing cepat

❌ **Tidak cocok untuk:**
- Gambar dengan pencahayaan tidak merata
- Shadow atau bayangan
- Kontras rendah

---

## 2️⃣ Adaptive Thresholding

Threshold yang **berbeda untuk setiap region** gambar—sangat efektif untuk pencahayaan tidak merata!

### Cara Kerja

Alih-alih satu nilai threshold untuk seluruh gambar, adaptive thresholding menghitung threshold **lokal** untuk setiap area kecil.

```
┌─────────────────────────────┐
│  Terang   │   Gelap         │
│  (T=200)  │   (T=80)        │  ← Threshold berbeda per region
│           │                 │
└─────────────────────────────┘
```

### Kode Python

```python
import cv2

img = cv2.imread('dokumen.jpg', cv2.IMREAD_GRAYSCALE)

# Adaptive threshold - Mean
# Parameter: maxValue, adaptiveMethod, thresholdType, blockSize, C
adaptive_mean = cv2.adaptiveThreshold(
    img,
    255,                           # maxValue
    cv2.ADAPTIVE_THRESH_MEAN_C,    # metode: rata-rata
    cv2.THRESH_BINARY,             # tipe output
    11,                            # blockSize (harus ganjil)
    2                              # C (konstanta dikurangi dari mean)
)

# Adaptive threshold - Gaussian
adaptive_gaussian = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # metode: Gaussian-weighted
    cv2.THRESH_BINARY,
    11,
    2
)
```

### Parameter Penting

| Parameter | Penjelasan |
|---|---|
| **maxValue** | Nilai maksimum yang diberikan (biasanya 255) |
| **adaptiveMethod** | `MEAN_C` atau `GAUSSIAN_C` |
| **thresholdType** | `BINARY` atau `BINARY_INV` |
| **blockSize** | Ukuran neighborhood (harus ganjil: 3, 5, 7, ...) |
| **C** | Konstanta yang dikurangi dari rata-rata |

### MEAN_C vs GAUSSIAN_C

| Metode | Penjelasan |
|---|---|
| **ADAPTIVE_THRESH_MEAN_C** | Threshold = rata-rata piksel di neighborhood - C |
| **ADAPTIVE_THRESH_GAUSSIAN_C** | Threshold = rata-rata tertimbang Gaussian - C |

> 💡 **Tips:** Gaussian biasanya memberikan hasil lebih halus dan natural.

### Perbandingan Simple vs Adaptive

![Simple vs Adaptive Thresholding](./images/06-adaptive-vs-simple.png)

### Pengaruh Block Size

![Pengaruh Block Size](./images/10-blocksize-effect.png)

---

## 3️⃣ Otsu's Binarization

**Otsu's method** secara otomatis menemukan nilai threshold optimal berdasarkan histogram gambar!

### Kapan Menggunakan Otsu?

Gunakan Otsu ketika:
- Gambar memiliki **dua puncak** di histogram (bimodal)
- Anda tidak tahu nilai threshold yang tepat
- Ingin threshold yang **otomatis optimal**

### Kode Python

```python
import cv2

img = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)

# Otsu's thresholding
# Set threshold ke 0 dan tambahkan flag THRESH_OTSU
ret, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Threshold optimal yang ditemukan: {ret}")
```

### Otsu dengan Gaussian Blur (Recommended!)

```python
import cv2

img = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)

# Blur dulu untuk mengurangi noise
blur = cv2.GaussianBlur(img, (5, 5), 0)

# Kemudian Otsu
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Threshold Otsu: {ret}")
```

### Cara Kerja Otsu (Simplified)

1. Hitung histogram gambar
2. Cari threshold yang **memaksimalkan varian antar-kelas**
3. Threshold tersebut memisahkan foreground dan background dengan optimal

```
Histogram Bimodal:
      │    ▓▓▓
      │   ▓▓▓▓▓               ▓▓▓
      │  ▓▓▓▓▓▓▓             ▓▓▓▓▓
Freq  │ ▓▓▓▓▓▓▓▓▓           ▓▓▓▓▓▓▓
      │▓▓▓▓▓▓▓▓▓▓▓         ▓▓▓▓▓▓▓▓▓
      └────────────────┬────────────────
                       ↑
                  Threshold Otsu
            (background) │ (foreground)
```

### Visualisasi Otsu's Thresholding

![Otsu Thresholding](./images/07-otsu-threshold.png)

---

## 💻 Praktik: Perbandingan Semua Metode

```python
import cv2
import matplotlib.pyplot as plt

# Baca gambar
img = cv2.imread('dokumen.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Simple Threshold
_, simple = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 2. Adaptive Mean
adaptive_mean = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)

# 3. Adaptive Gaussian
adaptive_gaussian = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# 4. Otsu
blur = cv2.GaussianBlur(img, (5, 5), 0)
_, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Tampilkan
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

axes[0, 1].imshow(simple, cmap='gray')
axes[0, 1].set_title('Simple (T=127)')

axes[0, 2].imshow(adaptive_mean, cmap='gray')
axes[0, 2].set_title('Adaptive Mean')

axes[1, 0].imshow(adaptive_gaussian, cmap='gray')
axes[1, 0].set_title('Adaptive Gaussian')

axes[1, 1].imshow(otsu, cmap='gray')
axes[1, 1].set_title("Otsu's")

# Histogram
axes[1, 2].hist(img.ravel(), 256, [0, 256])
axes[1, 2].set_title('Histogram')

plt.tight_layout()
plt.show()
```

---

## 🎨 Use Cases Thresholding

| Use Case | Metode Rekomendasi |
|---|---|
| **Scan Dokumen** | Adaptive Gaussian |
| **Plat Nomor Kendaraan** | Otsu + preprocessing |
| **Deteksi Objek Sederhana** | Simple Threshold |
| **Medical Imaging** | Adaptive atau Otsu |
| **Barcode/QR Reader** | Adaptive Threshold |

---

## ⚙️ Tips Memilih Threshold

### 1. Analisis Histogram Dulu

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)

plt.hist(img.ravel(), 256, [0, 256])
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
```

### 2. Panduan Memilih Metode

```
Apakah pencahayaan MERATA?
    │
    ├── YA → Apakah tahu nilai threshold?
    │         │
    │         ├── YA → ✅ Simple Threshold
    │         │
    │         └── TIDAK → ✅ Otsu's Threshold
    │
    └── TIDAK → ✅ Adaptive Threshold
                    │
                    ├── Dokumen/Teks → Gaussian
                    │
                    └── Noise tinggi → Mean
```

### 3. Contoh Use Case dengan Gambar Nyata

#### 🏷️ Simple Threshold - Logo Kontras Tinggi

![Contoh Logo](./images/usecase-logo.png)

**Kapan:** Gambar dengan kontras tinggi, latar belakang seragam, pencahayaan merata.

**Hasil Thresholding:**

![Demo Simple Threshold](./images/demo-simple-logo.png)

---

#### 🚗 Otsu's Threshold - Plat Nomor Kendaraan

![Contoh Plat Nomor](./images/usecase-license-plate.png)

**Kapan:** Histogram bimodal (dua puncak), tidak tahu nilai threshold yang tepat, butuh threshold otomatis.

**Hasil Thresholding:**

![Demo Otsu Threshold](./images/demo-otsu-plate.png)

---

#### 📄 Adaptive Gaussian - Scan Dokumen

![Contoh Dokumen](./images/usecase-document.png)

**Kapan:** Pencahayaan tidak merata, bayangan, dokumen dengan teks.

**Hasil Thresholding:**

![Demo Adaptive Document](./images/demo-adaptive-document.png)

---

#### 📱 Adaptive Threshold - Barcode/QR Code

![Contoh QR Code](./images/usecase-qrcode.png)

**Kapan:** Gambar dengan pencahayaan tidak konsisten, perlu segmentasi lokal.

**Hasil Thresholding:**

![Demo Adaptive QR Code](./images/demo-adaptive-qrcode.png)

---

## 📚 Rangkuman

| Metode | Fungsi | Kapan Digunakan |
|---|---|---|
| **Simple** | `cv2.threshold()` | Pencahayaan merata, threshold diketahui |
| **Adaptive Mean** | `cv2.adaptiveThreshold()` | Pencahayaan tidak merata |
| **Adaptive Gaussian** | `cv2.adaptiveThreshold()` | Dokumen, teks, hasil halus |
| **Otsu** | `cv2.THRESH_OTSU` | Histogram bimodal, otomatis |

### Cheat Sheet

```python
# Simple
_, out = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Adaptive Mean
out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                            cv2.THRESH_BINARY, 11, 2)

# Adaptive Gaussian
out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY, 11, 2)

# Otsu
_, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

---

## ➡️ Langkah Selanjutnya

Setelah menguasai thresholding, selanjutnya kita akan belajar **Transformasi Morfologi**—operasi untuk membersihkan dan memperbaiki hasil thresholding!

[Lanjut ke: Transformasi Morfologi →](./03-morphological-transformations.md)
