# Operasi Dasar pada Gambar

> **Setelah memahami piksel dan ruang warna, saatnya belajar memanipulasi gambar!**

Operasi dasar pada gambar adalah fondasi penting dalam Computer Vision. Hampir semua proyek CV dimulai dengan preprocessing—mengubah ukuran, memotong, atau memutar gambar sebelum dianalisis.

---

## 📐 Resize (Mengubah Ukuran)

**Resize** adalah operasi mengubah dimensi gambar (lebar dan tinggi).

![Operasi Resize dan Crop](./images/06-resize-crop.png)

### Mengapa Perlu Resize?

| Alasan | Penjelasan |
|---|---|
| **Standarisasi Input** | Model Deep Learning butuh ukuran input yang sama |
| **Efisiensi Memori** | Gambar lebih kecil = proses lebih cepat |
| **Consistency** | Dataset harus seragam untuk training |

### Jenis Resize

| Jenis | Penjelasan |
|---|---|
| **Downscale** | Mengecilkan gambar (misal: 1920×1080 → 640×480) |
| **Upscale** | Memperbesar gambar (misal: 640×480 → 1920×1080) |

### Kode Python

```python
import cv2

# Baca gambar
img = cv2.imread('foto.jpg')
print(f"Ukuran asli: {img.shape}")  # (height, width, channels)

# Resize ke ukuran spesifik
img_resized = cv2.resize(img, (640, 480))  # (width, height)
print(f"Ukuran baru: {img_resized.shape}")

# Resize dengan skala (50% dari asli)
img_half = cv2.resize(img, None, fx=0.5, fy=0.5)

# Resize dengan skala (200% dari asli)
img_double = cv2.resize(img, None, fx=2.0, fy=2.0)
```

### Metode Interpolasi

Saat resize, OpenCV perlu menghitung nilai piksel baru. Ada beberapa metode:

| Metode | Flag OpenCV | Kapan Digunakan |
|---|---|---|
| **Nearest Neighbor** | `cv2.INTER_NEAREST` | Cepat, hasil kasar |
| **Bilinear** | `cv2.INTER_LINEAR` | Default, keseimbangan |
| **Bicubic** | `cv2.INTER_CUBIC` | Lebih halus, lebih lambat |
| **Area** | `cv2.INTER_AREA` | Terbaik untuk downscale |
| **Lanczos** | `cv2.INTER_LANCZOS4` | Kualitas tinggi untuk upscale |

```python
# Resize dengan interpolasi berbeda
img_area = cv2.resize(img, (300, 200), interpolation=cv2.INTER_AREA)
img_cubic = cv2.resize(img, (300, 200), interpolation=cv2.INTER_CUBIC)
```

> 💡 **Tips:** Gunakan `INTER_AREA` untuk mengecilkan dan `INTER_CUBIC` atau `INTER_LANCZOS4` untuk memperbesar!

---

## ✂️ Crop (Memotong)

**Crop** adalah mengambil sebagian area dari gambar (Region of Interest / ROI).

### Cara Kerja Crop

Karena gambar adalah array, crop dilakukan dengan **slicing**:

```python
# Format: gambar[y1:y2, x1:x2]
# y = baris (vertikal), x = kolom (horizontal)

import cv2

img = cv2.imread('foto.jpg')

# Crop area dari (100,50) sampai (400,300)
# Koordinat: (x1=100, y1=50) ke (x2=400, y2=300)
cropped = img[50:300, 100:400]

cv2.imshow('Cropped', cropped)
cv2.waitKey(0)
```

### Visualisasi Koordinat

```
       x=0        x=100       x=400        x=width
        ↓           ↓           ↓            ↓
y=0  →  ┌───────────┬───────────┬────────────┐
        │           │           │            │
y=50 →  │           ├───────────┤            │
        │           │   CROP    │            │
        │           │   AREA    │            │
y=300→  │           ├───────────┤            │
        │           │           │            │
y=height│           │           │            │
        └───────────┴───────────┴────────────┘
```

### Contoh Penggunaan Crop

```python
import cv2

img = cv2.imread('wajah.jpg')
height, width = img.shape[:2]

# Crop bagian tengah gambar (50% dari ukuran asli)
center_x, center_y = width // 2, height // 2
crop_w, crop_h = width // 4, height // 4

cropped = img[center_y - crop_h : center_y + crop_h,
              center_x - crop_w : center_x + crop_w]
```

> ⚠️ **Perhatian:** Urutan koordinat adalah `[y:y, x:x]`, BUKAN `[x:x, y:y]`!

---

## 🔄 Rotate (Memutar)

**Rotate** adalah memutar gambar dengan sudut tertentu.

![Operasi Rotasi dan Flip](./images/07-rotate-flip.png)

### Rotasi Sederhana (90°, 180°, 270°)

```python
import cv2

img = cv2.imread('foto.jpg')

# Rotasi 90° searah jarum jam
rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Rotasi 180°
rotated_180 = cv2.rotate(img, cv2.ROTATE_180)

# Rotasi 90° berlawanan jarum jam (= 270° searah)
rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
```

### Rotasi dengan Sudut Bebas

Untuk rotasi sudut bebas (misal: 45°), gunakan **transformation matrix**:

```python
import cv2

img = cv2.imread('foto.jpg')
height, width = img.shape[:2]

# Titik pusat rotasi
center = (width // 2, height // 2)

# Buat matriks rotasi: pusat, sudut, skala
# Sudut positif = berlawanan jarum jam
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)

# Terapkan rotasi
rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
```

### Parameter Rotasi

| Parameter | Penjelasan |
|---|---|
| **center** | Titik pusat rotasi (x, y) |
| **angle** | Sudut rotasi (derajat), positif = counter-clockwise |
| **scale** | Faktor skala (1.0 = ukuran tetap) |

---

## 🪞 Flip (Membalik/Mencerminkan)

**Flip** adalah mencerminkan gambar secara horizontal atau vertikal.

### Jenis Flip

| Kode | Jenis | Penjelasan |
|:---:|---|---|
| `1` | Horizontal | Cermin kiri-kanan |
| `0` | Vertikal | Cermin atas-bawah |
| `-1` | Keduanya | Horizontal + Vertikal |

```python
import cv2

img = cv2.imread('foto.jpg')

# Flip horizontal (mirror)
flipped_h = cv2.flip(img, 1)

# Flip vertikal
flipped_v = cv2.flip(img, 0)

# Flip keduanya
flipped_both = cv2.flip(img, -1)
```

### Kapan Menggunakan Flip?

- **Data Augmentation**: Menambah variasi dataset untuk training
- **Mirror Effect**: Efek artistik
- **Koreksi Orientasi**: Memperbaiki gambar yang terbalik

---

## 🔲 Transformasi Affine

**Transformasi affine** adalah operasi geometris yang mempertahankan garis lurus dan paralelisme.

![Jenis-jenis Transformasi Affine](./images/08-affine-transform.png)

### Jenis Transformasi Affine

| Transformasi | Penjelasan |
|---|---|
| **Translation** | Menggeser posisi gambar |
| **Scaling** | Mengubah ukuran (resize) |
| **Rotation** | Memutar gambar |
| **Shear** | Memiringkan gambar |

### Translation (Menggeser)

```python
import cv2
import numpy as np

img = cv2.imread('foto.jpg')
height, width = img.shape[:2]

# Geser 100 piksel ke kanan, 50 piksel ke bawah
tx, ty = 100, 50

# Matriks translasi
M = np.float32([[1, 0, tx],
                [0, 1, ty]])

# Terapkan translasi
translated = cv2.warpAffine(img, M, (width, height))
```

### Shear (Memiringkan)

```python
import cv2
import numpy as np

img = cv2.imread('foto.jpg')
height, width = img.shape[:2]

# Matriks shear
# shear_x = memiringkan horizontal
# shear_y = memiringkan vertikal
shear_x = 0.2

M = np.float32([[1, shear_x, 0],
                [0, 1, 0]])

sheared = cv2.warpAffine(img, M, (int(width * 1.5), height))
```

---

## 🎨 Praktik: Menggabungkan Operasi

Biasanya dalam preprocessing, kita menggabungkan beberapa operasi:

```python
import cv2

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Pipeline preprocessing standar untuk Deep Learning
    """
    # 1. Baca gambar
    img = cv2.imread(image_path)
    
    # 2. Konversi BGR ke RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Resize ke ukuran target
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # 4. Normalisasi (0-255 → 0-1)
    img = img / 255.0
    
    return img

# Contoh penggunaan
processed = preprocess_image('foto.jpg', (224, 224))
print(f"Shape: {processed.shape}")  # (224, 224, 3)
print(f"Range: {processed.min():.2f} - {processed.max():.2f}")  # 0.00 - 1.00
```

---

## 💻 Latihan Praktis

Coba buat script yang melakukan operasi berikut:

```python
import cv2
import matplotlib.pyplot as plt

# Baca gambar
img = cv2.imread('foto.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Operasi
resized = cv2.resize(img_rgb, (200, 200))
cropped = img_rgb[50:200, 100:300]
rotated = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
flipped = cv2.flip(img_rgb, 1)

# Tampilkan
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original')

axes[0, 1].imshow(resized)
axes[0, 1].set_title('Resized (200x200)')

axes[0, 2].imshow(cropped)
axes[0, 2].set_title('Cropped')

axes[1, 0].imshow(rotated)
axes[1, 0].set_title('Rotated 90°')

axes[1, 1].imshow(flipped)
axes[1, 1].set_title('Flipped Horizontal')

axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
```

---

## 📚 Rangkuman

| Operasi | Fungsi OpenCV | Penjelasan |
|---|---|---|
| **Resize** | `cv2.resize()` | Mengubah dimensi gambar |
| **Crop** | Slicing `img[y1:y2, x1:x2]` | Memotong area tertentu |
| **Rotate** | `cv2.rotate()` / `cv2.warpAffine()` | Memutar gambar |
| **Flip** | `cv2.flip()` | Mencerminkan gambar |
| **Translate** | `cv2.warpAffine()` | Menggeser posisi gambar |
| **Shear** | `cv2.warpAffine()` | Memiringkan gambar |

---

## ➡️ Langkah Selanjutnya

Setelah menguasai operasi dasar, kita akan masuk ke **Image Processing** yang lebih advanced—dimulai dengan **Filtering dan Blurring**!

[Lanjut ke: Filtering dan Blurring →](../02-Image-Processing/01-filtering-and-blurring.md)
