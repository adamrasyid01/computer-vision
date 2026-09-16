 # Filtering dan Blurring

> **Menghaluskan gambar adalah langkah penting untuk mengurangi noise sebelum analisis lebih lanjut.**

Dalam bab ini, kita akan mempelajari konsep **filtering** (penyaringan) dan berbagai teknik **blurring** (pengaburan) yang fundamental dalam image processing.

---

## 🧮 Apa itu Konvolusi (Convolution)?

**Konvolusi** adalah operasi matematika fundamental dalam image processing. Hampir semua filter (blur, sharpen, edge detection) menggunakan konvolusi.

> 💡 **Analogi Sederhana:** Bayangkan Anda memindai gambar dengan "jendela kecil" (kernel). Di setiap posisi, Anda menghitung nilai baru berdasarkan piksel di bawah jendela tersebut.

![2D_Convolution_Animation](./images/2D_Convolution_Animation.gif)

### Rumus Konvolusi 2D

![Rumus Konvolusi](./images/conv-formula.png)

**Dalam bahasa sederhana:**
```
Output[x,y] = Jumlah dari (setiap piksel tetangga × bobot kernel)
```

### Cara Kerja Konvolusi: Step-by-Step

![Proses Konvolusi Step by Step](./images/conv-step-by-step.png)

**Penjelasan Langkah:**

| Langkah | Aksi | Hasil |
|:---:|---|---|
| 1 | Letakkan kernel (3×3) di atas gambar | Pilih region piksel |
| 2 | Ambil nilai piksel di bawah kernel | Region 3×3: [25, 35, 45, 30, 40, 50, 35, 45, 55] |
| 3 | Kalikan setiap piksel dengan bobot kernel | 25×(1/9), 35×(1/9), ... |
| 4 | Jumlahkan semua hasil perkalian | (25+35+45+30+40+50+35+45+55)/9 = 40 |
| 5 | Nilai 40 menjadi output di posisi tersebut | Output[1,1] = 40 |
| 6 | Geser kernel, ulangi untuk semua posisi | Gambar output selesai |

### Kernel Meluncur (Sliding)

Kernel bergerak dari **kiri→kanan**, kemudian **turun** ke baris berikutnya:

![Kernel Sliding Animation](./images/conv-kernel-sliding.png)

### Berbagai Jenis Kernel

Kernel yang berbeda menghasilkan efek yang berbeda:

![Kernel Umum](./images/conv-kernel-matrix.png)

| Kernel | Jumlah Bobot | Efek |
|---|:---:|---|
| **Identity** | 1 | Tidak berubah (untuk testing) |
| **Average Blur** | 1 | Rata-rata merata |
| **Gaussian Blur** | 1 | Blur natural, tengah lebih berat |
| **Sharpen** | 1 | Menajamkan tepi |
| **Edge Detection** | 0 | Deteksi tepi (output bisa negatif) |
| **Emboss** | 1 | Efek timbul/3D |

### Efek Kernel pada Gambar Nyata

![Efek Berbagai Kernel](./images/conv-different-kernels.png)

### Implementasi dengan OpenCV

```python
import cv2
import numpy as np

# Baca gambar
img = cv2.imread('foto.jpg')

# Definisi kernel custom
kernel_blur = np.ones((3, 3), np.float32) / 9    # Average blur
kernel_sharp = np.array([[0, -1, 0],              # Sharpen
                         [-1, 5, -1],
                         [0, -1, 0]])

# Terapkan konvolusi
blurred = cv2.filter2D(img, -1, kernel_blur)
sharpened = cv2.filter2D(img, -1, kernel_sharp)
```

> ⚠️ **Poin Penting:**
> - Ukuran kernel harus **ganjil** (3×3, 5×5, 7×7) agar ada piksel tengah
> - Kernel lebih besar = efek lebih kuat, tapi **lebih lambat**
> - Jumlah bobot kernel = 1 → brightness tetap, = 0 → deteksi tepi

---

## 🔇 Jenis-jenis Noise

Sebelum belajar blur, penting memahami jenis noise yang ingin dihilangkan:

| Jenis Noise | Karakteristik | Contoh Penyebab |
|---|---|---|
| **Gaussian Noise** | Bintik halus merata | Sensor kamera low-light |
| **Salt-and-Pepper** | Titik hitam-putih acak | Transmisi data rusak |
| **Speckle Noise** | Bercak granular | Gambar ultrasound/radar |

---

## 🌫️ Teknik Blurring


**Demo dengan Gambar Nyata:**

![Demo Perbandingan Blur](./images/demo-blur-types.png)

![Efek Blur pada Tekstur](./images/demo-blur-landscape.png)

---

### 1. Average Blur (Box Filter)

Blur paling sederhana—mengambil **rata-rata** piksel di area kernel.

```python
import cv2

img = cv2.imread('foto.jpg')

# Average blur dengan kernel 5x5
blurred = cv2.blur(img, (5, 5))

# Atau menggunakan filter2D
import numpy as np
kernel = np.ones((5, 5), np.float32) / 25
blurred = cv2.filter2D(img, -1, kernel)
```

#### Karakteristik

| Kelebihan | Kekurangan |
|---|---|
| ✅ Cepat dan sederhana | ❌ Mengaburkan tepi (edge) |
| ✅ Mengurangi noise umum | ❌ Hasil kurang natural |

---

### 2. Gaussian Blur

Blur yang lebih **natural** menggunakan distribusi Gaussian (bobot lebih besar di tengah).

```python
import cv2

img = cv2.imread('foto.jpg')

# Gaussian blur
# Parameter: (kernel_size, sigmaX)
# kernel_size harus ganjil: 3, 5, 7, dst.
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Dengan sigma manual
blurred_custom = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)
```

#### Kernel Gaussian 3×3 (Aproximasi)

```
        ┌─────┬─────┬─────┐
        │  1  │  2  │  1  │
   1    ├─────┼─────┼─────┤
  ── ×  │  2  │  4  │  2  │
  16    ├─────┼─────┼─────┤
        │  1  │  2  │  1  │
        └─────┴─────┴─────┘
```

#### Parameter Penting

| Parameter | Penjelasan |
|---|---|
| **ksize** | Ukuran kernel (harus ganjil) |
| **sigmaX** | Standar deviasi di sumbu X (0 = dihitung otomatis) |
| **sigmaY** | Standar deviasi di sumbu Y (default = sigmaX) |

> 💡 **Tips:** Sigma lebih besar = blur lebih kuat. Kernel lebih besar = blur lebih lebar.

---

### 3. Median Blur

Mengganti piksel tengah dengan **median** (nilai tengah) dari area kernel. **Sangat efektif untuk salt-and-pepper noise!**

```python
import cv2

img = cv2.imread('foto_noisy.jpg')

# Median blur
# Parameter: kernel_size (harus ganjil)
blurred = cv2.medianBlur(img, 5)
```

#### Cara Kerja Median

```
Piksel di kernel: [52, 55, 58, 60, 255, 61, 59, 54, 57]
                              ↑ noise (salt)

Setelah diurutkan: [52, 54, 55, 57, 58, 59, 60, 61, 255]
                                    ↑ median

Nilai baru = 58 (noise 255 dihilangkan!)
```

#### Karakteristik

| Kelebihan | Kekurangan |
|---|---|
| ✅ Sangat bagus untuk salt-and-pepper | ❌ Lebih lambat dari Gaussian |
| ✅ Mempertahankan tepi lebih baik | ❌ Kurang efektif untuk Gaussian noise |

**Demo: Median Blur untuk Salt-Pepper Noise:**

![Demo Salt-Pepper Noise Removal](./images/demo-salt-pepper.png)

---

### 4. Bilateral Filter

Filter yang **mengaburkan sambil mempertahankan tepi**. Ini filter paling canggih!

```python
import cv2

img = cv2.imread('foto.jpg')

# Bilateral filter
# Parameter: d, sigmaColor, sigmaSpace
blurred = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
```

#### Parameter Bilateral

| Parameter | Penjelasan |
|---|---|
| **d** | Diameter area piksel (gunakan 9 untuk offline, 5 untuk real-time) |
| **sigmaColor** | Pengaruh warna—nilai besar = warna berbeda ikut dicampur |
| **sigmaSpace** | Pengaruh jarak—nilai besar = piksel jauh ikut dicampur |

#### Cara Kerja

Bilateral filter mempertimbangkan **dua faktor**:
1. **Jarak spasial**: Piksel dekat lebih berpengaruh
2. **Perbedaan warna**: Piksel dengan warna mirip lebih berpengaruh

> 💡 **Hasilnya:** Area halus jadi blur, tapi tepi tetap tajam!

**Demo: Bilateral Filter untuk Smoothing Wajah:**

![Demo Bilateral Portrait](./images/demo-bilateral-portrait.png)

**Demo: Noise Reduction pada Foto Low-Light:**

![Demo Noise Low-Light](./images/demo-noise-lowlight.png)

---

## 📊 Perbandingan Teknik Blur

| Teknik | Kecepatan | Noise Umum | Salt-Pepper | Preserve Edge |
|---|:---:|:---:|:---:|:---:|
| **Average** | ⚡⚡⚡ | ✅ | ❌ | ❌ |
| **Gaussian** | ⚡⚡⚡ | ✅✅ | ❌ | ❌ |
| **Median** | ⚡⚡ | ✅ | ✅✅✅ | ✅ |
| **Bilateral** | ⚡ | ✅✅ | ✅ | ✅✅✅ |

---

## 🛠️ Custom Kernel

Anda bisa membuat kernel sendiri untuk efek tertentu:

### Sharpening (Penajaman)

```python
import cv2
import numpy as np

img = cv2.imread('foto.jpg')

# Kernel sharpening
kernel_sharpen = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])

sharpened = cv2.filter2D(img, -1, kernel_sharpen)
```

### Edge Enhancement

```python
# Kernel edge enhancement
kernel_edge = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])

enhanced = cv2.filter2D(img, -1, kernel_edge)
```

### Emboss Effect

```python
# Kernel emboss
kernel_emboss = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])

embossed = cv2.filter2D(img, -1, kernel_emboss)
```

**Demo: Efek Berbagai Kernel:**

![Demo Kernel Effects](./images/demo-kernel-effects.png)

---

## 💻 Praktik: Membandingkan Blur

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img = cv2.imread('foto.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Tambahkan noise salt-and-pepper untuk testing
def add_salt_pepper(image, amount=0.02):
    noisy = image.copy()
    # Salt
    num_salt = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i-1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    # Pepper
    num_pepper = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

noisy = add_salt_pepper(img_rgb)

# Apply berbagai blur
average = cv2.blur(noisy, (5, 5))
gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
median = cv2.medianBlur(noisy, 5)
bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)

# Tampilkan
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original')

axes[0, 1].imshow(noisy)
axes[0, 1].set_title('Noisy (Salt-Pepper)')

axes[0, 2].imshow(average)
axes[0, 2].set_title('Average Blur')

axes[1, 0].imshow(gaussian)
axes[1, 0].set_title('Gaussian Blur')

axes[1, 1].imshow(median)
axes[1, 1].set_title('Median Blur ✓')

axes[1, 2].imshow(bilateral)
axes[1, 2].set_title('Bilateral Filter')

plt.tight_layout()
plt.show()
```

---

## 📚 Rangkuman

| Konsep | Penjelasan |
|---|---|
| **Konvolusi** | Operasi kernel yang meluncur di atas gambar |
| **Kernel** | Matriks kecil (3×3, 5×5) untuk transformasi |
| **Average Blur** | Rata-rata sederhana, cepat tapi blur merata |
| **Gaussian Blur** | Natural, menggunakan distribusi Gaussian |
| **Median Blur** | Terbaik untuk salt-and-pepper noise |
| **Bilateral** | Blur halus, tapi mempertahankan tepi |

---

## ➡️ Langkah Selanjutnya

Setelah memahami filtering, selanjutnya kita akan belajar **Thresholding**—teknik untuk mengubah gambar menjadi biner (hitam-putih)!

[Lanjut ke: Thresholding →](./02-thresholding.md)
