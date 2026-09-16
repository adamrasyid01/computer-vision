# Transformasi Morfologi

> **Morfologi adalah seni "membentuk" gambar biner—menghilangkan noise, mengisi lubang, dan memperjelas struktur objek.**

Setelah melakukan thresholding, hasil gambar biner seringkali tidak sempurna. Ada noise kecil, lubang-lubang, atau tepi yang tidak rata. **Transformasi morfologi** adalah solusinya!

---

## 🧱 Konsep Dasar

Transformasi morfologi bekerja pada **gambar biner** (hitam-putih) menggunakan **structuring element** (kernel) yang meluncur di atas gambar.

### Structuring Element

Kernel berbentuk kotak, elips, atau salib yang menentukan "neighborhood" untuk operasi:

```
Kotak 3x3:        Elips 5x5:                 Salib 3x3:
┌───┬───┬───┐     ┌───┬───┬───┬───┬───┐     ┌───┬───┬───┐
│ 1 │ 1 │ 1 │     │ 0 │ 1 │ 1 │ 1 │ 0 │     │ 0 │ 1 │ 0 │
├───┼───┼───┤     ├───┼───┼───┼───┼───┤     ├───┼───┼───┤
│ 1 │ 1 │ 1 │     │ 1 │ 1 │ 1 │ 1 │ 1 │     │ 1 │ 1 │ 1 │
├───┼───┼───┤     ├───┼───┼───┼───┼───┤     ├───┼───┼───┤
│ 1 │ 1 │ 1 │     │ 1 │ 1 │ 1 │ 1 │ 1 │     │ 0 │ 1 │ 0 │
└───┴───┴───┘     ├───┼───┼───┼───┼───┤     └───┴───┴───┘
                  │ 1 │ 1 │ 1 │ 1 │ 1 │
                  ├───┼───┼───┼───┼───┤
                  │ 0 │ 1 │ 1 │ 1 │ 0 │
                  └───┴───┴───┴───┴───┘
```

### Membuat Structuring Element di OpenCV

```python
import cv2
import numpy as np

# Kotak 5x5
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Elips 5x5
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Salib 5x5
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# Atau manual dengan numpy
kernel_manual = np.ones((3, 3), np.uint8)
```

**Demo Efek Bentuk Kernel:**

![Kernel Shapes](./images/demo-kernel-shapes.png)

---

## ⬇️ Erosi (Erosion)

**Erosi** mengikis/menciutkan area putih (foreground). Piksel putih menjadi hitam jika **ada tetangga hitam**.

### Cara Kerja

```
Aturan: Piksel tetap PUTIH hanya jika SEMUA tetangga dalam kernel = PUTIH
        Jika ada yang HITAM → piksel jadi HITAM

Sebelum Erosi:       Setelah Erosi:
┌───┬───┬───┬───┐    ┌───┬───┬───┬───┐
│ ■ │ ■ │ ■ │ ■ │    │   │ ■ │ ■ │   │
├───┼───┼───┼───┤    ├───┼───┼───┼───┤
│ ■ │ ■ │ ■ │ ■ │ →  │ ■ │ ■ │ ■ │ ■ │  (objek menyusut)
├───┼───┼───┼───┤    ├───┼───┼───┼───┤
│ ■ │ ■ │ ■ │ ■ │    │   │ ■ │ ■ │   │
└───┴───┴───┴───┘    └───┴───┴───┴───┘
```

### Kode Python

```python
import cv2
import numpy as np

img = cv2.imread('binary.jpg', cv2.IMREAD_GRAYSCALE)

# Buat kernel
kernel = np.ones((3, 3), np.uint8)

# Erosi
eroded = cv2.erode(img, kernel, iterations=1)

# Erosi lebih kuat (2 kali iterasi)
eroded_strong = cv2.erode(img, kernel, iterations=2)
```

**Efek Iterasi pada Erosi:**

![Iterations](./images/demo-iterations.png)

### Kegunaan Erosi

| Kegunaan | Penjelasan |
|---|---|
| ✅ Menghilangkan noise kecil | Titik-titik putih kecil hilang |
| ✅ Memisahkan objek yang menyatu | Objek yang terhubung tipis terputus |
| ❌ Efek samping | Objek menjadi lebih kecil |

**Demo Erosi vs Dilasi:**

![Erosion Dilation](./images/demo-erosion-dilation.png)

---

## ⬆️ Dilasi (Dilation)

**Dilasi** memperluas area putih (foreground). Piksel hitam menjadi putih jika **ada tetangga putih**.

### Cara Kerja

```
Aturan: Piksel jadi PUTIH jika ADA tetangga dalam kernel = PUTIH
        Piksel hitam "tertulari" putih

Sebelum Dilasi:      Setelah Dilasi:
┌───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┐
│   │ ■ │ ■ │   │    │ ■ │ ■ │ ■ │ ■ │ ■ │
├───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
│   │ ■ │ ■ │   │ →  │ ■ │ ■ │ ■ │ ■ │ ■ │  (objek membesar)
├───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
│   │ ■ │ ■ │   │    │ ■ │ ■ │ ■ │ ■ │ ■ │
└───┴───┴───┴───┘    └───┴───┴───┴───┴───┘
```

### Kode Python

```python
import cv2
import numpy as np

img = cv2.imread('binary.jpg', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((3, 3), np.uint8)

# Dilasi
dilated = cv2.dilate(img, kernel, iterations=1)

# Dilasi lebih kuat
dilated_strong = cv2.dilate(img, kernel, iterations=2)
```

### Kegunaan Dilasi

| Kegunaan | Penjelasan |
|---|---|
| ✅ Menutup lubang kecil | Lubang hitam di dalam objek tertutup |
| ✅ Menyambung objek terputus | Bagian yang hampir menyatu jadi tersambung |
| ❌ Efek samping | Objek menjadi lebih besar |

---

## 🔄 Opening (Erosi → Dilasi)

**Opening** = Erosi diikuti Dilasi. Efektif untuk **menghilangkan noise kecil** tanpa terlalu mengubah ukuran objek!

### Visualisasi

```
Original → Erosi → Dilasi = Opening
   ●●        ●       ●●
  ●●●●  →   ●●   →  ●●●●   (noise hilang, objek tetap)
   ●●        ●       ●●
   
  ·          ·               (noise kecil hilang!)
```

### Kode Python

```python
import cv2
import numpy as np

img = cv2.imread('noisy_binary.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)

# Opening (erosi lalu dilasi)
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
```

### Kapan Menggunakan Opening?

✅ Ada **titik-titik noise putih** pada background hitam

**Demo Opening & Closing:**

![Opening Closing](./images/demo-opening-closing.png)

---

## 🔄 Closing (Dilasi → Erosi)

**Closing** = Dilasi diikuti Erosi. Efektif untuk **menutup lubang kecil** di dalam objek!

### Visualisasi

```
Original → Dilasi → Erosi = Closing
  ●●●       ●●●●     ●●●
  ● ●   →  ●●●●  →  ●●●    (lubang tertutup!)
  ●●●       ●●●●     ●●●
```

### Kode Python

```python
import cv2
import numpy as np

img = cv2.imread('holey_binary.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)

# Closing (dilasi lalu erosi)
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```

### Kapan Menggunakan Closing?

✅ Ada **lubang-lubang kecil** di dalam objek putih

---

## 🎭 Gradient Morfologi

**Gradient** = Dilasi - Erosi. Menghasilkan **outline/tepi** objek!

```python
import cv2
import numpy as np

img = cv2.imread('binary.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)

# Gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
```

### Visualisasi

```
Original:    Gradient:
  ●●●          □□□
  ●●●    →    □ □    (hanya outline)
  ●●●          □□□
```

**Demo Gradient, Top Hat & Black Hat:**

![Gradient Tophat](./images/demo-gradient-tophat.png)

---

## 🎩 Top Hat & Black Hat

### Top Hat

**Top Hat** = Original - Opening

Mengidentifikasi **objek kecil/terang** yang lebih kecil dari kernel.

```python
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
```

### Black Hat

**Black Hat** = Closing - Original

Mengidentifikasi **lubang kecil/gelap** yang lebih kecil dari kernel.

```python
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
```

---

## 📊 Ringkasan Operasi Morfologi

| Operasi | Rumus | Kegunaan |
|---|---|---|
| **Erosi** | - | Mengikis, hapus noise kecil |
| **Dilasi** | - | Memperluas, tutup lubang |
| **Opening** | Erosi → Dilasi | Hapus noise putih |
| **Closing** | Dilasi → Erosi | Tutup lubang hitam |
| **Gradient** | Dilasi - Erosi | Ekstrak outline |
| **Top Hat** | Original - Opening | Deteksi objek kecil terang |
| **Black Hat** | Closing - Original | Deteksi lubang kecil gelap |

---

## 💻 Praktik: Membersihkan Gambar Biner

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img = cv2.imread('noisy.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Kernel
kernel = np.ones((3, 3), np.uint8)

# Operasi morfologi
eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(binary, kernel, iterations=1)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

# Tampilkan
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(binary, cmap='gray')
axes[0, 0].set_title('Original Binary')

axes[0, 1].imshow(eroded, cmap='gray')
axes[0, 1].set_title('Eroded')

axes[0, 2].imshow(dilated, cmap='gray')
axes[0, 2].set_title('Dilated')

axes[1, 0].imshow(opened, cmap='gray')
axes[1, 0].set_title('Opening')

axes[1, 1].imshow(closed, cmap='gray')
axes[1, 1].set_title('Closing')

axes[1, 2].imshow(gradient, cmap='gray')
axes[1, 2].set_title('Gradient')

plt.tight_layout()
plt.show()
```

---

## 🎯 Pipeline Umum: Thresholding + Morfologi

```python
import cv2
import numpy as np

def clean_binary_image(image_path):
    """
    Pipeline: Grayscale → Threshold → Morfologi
    """
    # 1. Baca grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Blur untuk kurangi noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 3. Otsu threshold
    _, binary = cv2.threshold(blurred, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Morfologi untuk bersihkan
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

result = clean_binary_image('foto.jpg')
cv2.imwrite('hasil_bersih.jpg', result)
```

**Demo Use Case: Membersihkan Text:**

![Morph Usecase](./images/demo-morph-usecase.png)

---

## 📚 Rangkuman

| Operasi | Fungsi OpenCV | Kapan Digunakan |
|---|---|---|
| **Erosi** | `cv2.erode()` | Mengikis objek, hapus noise |
| **Dilasi** | `cv2.dilate()` | Memperluas objek, tutup lubang |
| **Opening** | `cv2.MORPH_OPEN` | Hapus noise putih kecil |
| **Closing** | `cv2.MORPH_CLOSE` | Tutup lubang hitam kecil |
| **Gradient** | `cv2.MORPH_GRADIENT` | Ekstrak outline objek |

### Cheat Sheet

```python
kernel = np.ones((3, 3), np.uint8)

# Dasar
eroded = cv2.erode(img, kernel, iterations=1)
dilated = cv2.dilate(img, kernel, iterations=1)

# Kombinasi
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
```

---

## ➡️ Langkah Selanjutnya

Setelah menguasai image processing dasar, selanjutnya kita akan masuk ke **Feature Detection**—dimulai dengan **Gradien dan Deteksi Tepi**!

[Lanjut ke: Gradien dan Deteksi Tepi →](../03-Feature-Detection/01-gradients-and-edges.md)
