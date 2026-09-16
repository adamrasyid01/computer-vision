# Gradien dan Deteksi Tepi

> **Deteksi tepi adalah kemampuan fundamental untuk "melihat" batas objek dalam gambar—langkah penting sebelum analisis lebih lanjut.**

Tepi (edge) adalah area di mana intensitas piksel berubah drastis. Dengan mendeteksi tepi, kita dapat menemukan kontur objek, bentuk, dan struktur dalam gambar.

---

## 🎯 Apa itu Edge/Tepi?

**Tepi** adalah perubahan tajam dalam intensitas piksel—menandakan batas antara objek dan background, atau antara dua objek berbeda.

### Visualisasi Konsep Edge

![Konsep Edge](./images/edge-concept.png)

**Penjelasan:**
- **Profil intensitas** menunjukkan nilai piksel di sepanjang garis
- **Edge** terjadi di mana ada **lompatan tajam** (step) pada intensitas
- **Gradient** (turunan) mencapai **puncak** di lokasi edge

### Jenis Tepi

| Jenis | Visualisasi | Contoh |
|---|:---:|---|
| **Step Edge** | █████░░░░ | Batas objek dengan background |
| **Ramp Edge** | █████████░░░░░░░░ | Bayangan gradual |
| **Roof Edge** | ░░░███░░░ | Garis tipis |

---

## 📐 Gradien Gambar

**Gradien** adalah ukuran seberapa cepat intensitas berubah di setiap piksel. Gradien tinggi = kemungkinan tepi!

### Gradien sebagai Vektor

![Gradient Vector](./images/demo-gradient-vector.png)

### Rumus Gradien

| Komponen | Rumus | Arti |
|---|---|---|
| **Gx** | ∂I/∂x | Perubahan di arah horizontal |
| **Gy** | ∂I/∂y | Perubahan di arah vertikal |
| **Magnitude** | √(Gx² + Gy²) | Kekuatan tepi |
| **Direction** | arctan(Gy/Gx) | Arah tepi |

---

## 🔧 Operator Gradien

### 1. Sobel Operator

Operator paling populer untuk deteksi gradien. Menggunakan kernel 3×3 yang sensitif terhadap tepi horizontal atau vertikal.

#### Kernel Sobel

```
Sobel X (deteksi tepi vertikal):    Sobel Y (deteksi tepi horizontal):
┌────┬────┬────┐                    ┌────┬────┬────┐
│ -1 │  0 │ +1 │                    │ -1 │ -2 │ -1 │
├────┼────┼────┤                    ├────┼────┼────┤
│ -2 │  0 │ +2 │                    │  0 │  0 │  0 │
├────┼────┼────┤                    ├────┼────┼────┤
│ -1 │  0 │ +1 │                    │ +1 │ +2 │ +1 │
└────┴────┴────┘                    └────┴────┴────┘
```

#### Kode Python

```python
import cv2
import numpy as np

img = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel X (deteksi tepi vertikal)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# Sobel Y (deteksi tepi horizontal)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Gabungkan (magnitude)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Atau dengan rumus manual
sobel_abs = np.sqrt(sobel_x**2 + sobel_y**2)

# Konversi ke uint8 untuk tampilan
sobel_display = cv2.convertScaleAbs(sobel_combined)
```

#### Parameter Sobel

| Parameter | Penjelasan |
|---|---|
| **ddepth** | Kedalaman output (`cv2.CV_64F` untuk presisi) |
| **dx** | Order turunan di X (1 = aktif, 0 = tidak) |
| **dy** | Order turunan di Y |
| **ksize** | Ukuran kernel (1, 3, 5, atau 7) |

**Demo Sobel X vs Y:**

![Demo Sobel X Y](./images/demo-sobel-xy.png)

> 💡 **Catatan:** Sobel X mendeteksi tepi **vertikal** (|), Sobel Y mendeteksi tepi **horizontal** (—)

---

### 2. Scharr Operator

Versi **lebih akurat** dari Sobel untuk kernel 3×3.

```python
# Scharr X
scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)

# Scharr Y
scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)

# Atau dengan Sobel ksize=-1
scharr_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
```

---

### 3. Laplacian Operator

Mendeteksi tepi di **semua arah** sekaligus menggunakan turunan kedua.

#### Kernel Laplacian

```
┌────┬────┬────┐
│  0 │  1 │  0 │
├────┼────┼────┤
│  1 │ -4 │  1 │
├────┼────┼────┤
│  0 │  1 │  0 │
└────┴────┴────┘
```

#### Kode Python

```python
import cv2

img = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)

# Laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Konversi untuk tampilan
laplacian_display = cv2.convertScaleAbs(laplacian)
```

---

## ⭐ Canny Edge Detection

**Canny** adalah algoritma deteksi tepi paling populer dan robust! Dikembangkan oleh John Canny pada 1986.

### Langkah-langkah Canny

```
1. Gaussian Blur      → Kurangi noise
        ↓
2. Hitung Gradien     → Sobel X dan Y
        ↓
3. Non-Maximum        → Tipiskan tepi (1 piksel)
   Suppression
        ↓
4. Double Threshold   → Tentukan tepi kuat/lemah
        ↓
5. Hysteresis         → Hubungkan tepi lemah ke kuat
        ↓
   OUTPUT: Tepi bersih!
```

**Demo Langkah Canny:**

![Langkah Canny](./images/demo-canny-steps.png)

### Kode Python

```python
import cv2

img = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)

# Canny edge detection
# Parameter: threshold1 (low), threshold2 (high)
edges = cv2.Canny(img, 100, 200)

# Dengan blur preprocessing (recommended!)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
edges_clean = cv2.Canny(blurred, 100, 200)
```

### Parameter Canny

| Parameter | Penjelasan |
|---|---|
| **threshold1** | Threshold bawah (tepi lemah) |
| **threshold2** | Threshold atas (tepi kuat) |
| **apertureSize** | Ukuran kernel Sobel (default: 3) |
| **L2gradient** | True = gunakan L2 norm (lebih akurat) |

### Memilih Threshold

```
Aturan Praktis:
- Ratio threshold2:threshold1 = 2:1 atau 3:1
- Contoh: (50, 150), (100, 200), (50, 100)

Threshold rendah → Lebih banyak tepi (termasuk noise)
Threshold tinggi → Lebih sedikit tepi (hanya yang kuat)
```

**Demo Efek Threshold:**

![Canny Thresholds](./images/demo-canny-thresholds.png)

---

## 📊 Perbandingan Metode

![Perbandingan Edge Detection](./images/demo-edge-comparison.png)

| Metode | Kecepatan | Akurasi | Noise Sensitivity |
|---|:---:|:---:|:---:|
| **Sobel** | ⚡⚡⚡ | ✅ | Sedang |
| **Scharr** | ⚡⚡⚡ | ✅✅ | Sedang |
| **Laplacian** | ⚡⚡⚡ | ✅ | Tinggi |
| **Canny** | ⚡⚡ | ✅✅✅ | Rendah |

---

## 💻 Praktik: Perbandingan Semua Metode

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)

# Blur untuk kurangi noise
blurred = cv2.GaussianBlur(img, (3, 3), 0)

# Sobel
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = cv2.convertScaleAbs(sobel)

# Scharr
scharr_x = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
scharr = cv2.magnitude(scharr_x, scharr_y)
scharr = cv2.convertScaleAbs(scharr)

# Laplacian
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Canny
canny = cv2.Canny(blurred, 100, 200)

# Tampilkan
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

axes[0, 1].imshow(sobel, cmap='gray')
axes[0, 1].set_title('Sobel')

axes[0, 2].imshow(scharr, cmap='gray')
axes[0, 2].set_title('Scharr')

axes[1, 0].imshow(laplacian, cmap='gray')
axes[1, 0].set_title('Laplacian')

axes[1, 1].imshow(canny, cmap='gray')
axes[1, 1].set_title('Canny ✓')

axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
```

---

## 🎯 Use Cases

**Demo: Lane Detection pada Jalan Raya**

![Lane Detection](./images/demo-lane-detection.png)

| Use Case | Metode Rekomendasi |
|---|---|
| **Deteksi objek umum** | Canny |
| **Preprocessing untuk kontur** | Canny |
| **Analisis tekstur** | Sobel/Scharr |
| **Deteksi garis** | Canny + Hough Transform |
| **Real-time processing** | Sobel |

---

## 📚 Rangkuman

| Metode | Fungsi | Karakteristik |
|---|---|---|
| **Sobel** | `cv2.Sobel()` | Gradien X atau Y, cepat |
| **Scharr** | `cv2.Scharr()` | Lebih akurat dari Sobel 3×3 |
| **Laplacian** | `cv2.Laplacian()` | Semua arah, sensitif noise |
| **Canny** | `cv2.Canny()` | Multi-stage, hasil terbaik |

### Cheat Sheet

```python
import cv2

img = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5, 5), 0)

# Sobel
sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

# Laplacian
laplacian = cv2.Laplacian(blur, cv2.CV_64F)

# Canny (RECOMMENDED)
edges = cv2.Canny(blur, 100, 200)
```

---

## ➡️ Langkah Selanjutnya

Setelah menguasai deteksi tepi, selanjutnya kita akan belajar **Contours**—cara menemukan dan menganalisis bentuk objek dari tepi yang terdeteksi!

[Lanjut ke: Kontur →](./02-contours.md)
