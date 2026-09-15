# Kontur (Contours)

> **Kontur adalah kurva yang menghubungkan semua titik kontinu dengan warna atau intensitas yang sama—menggambarkan bentuk objek.**

Setelah deteksi tepi dengan Canny, langkah selanjutnya adalah menemukan **kontur**—outline objek yang dapat dianalisis lebih lanjut untuk mengenali bentuk, menghitung luas, atau tracking objek.

---

## 🎯 Apa itu Kontur?

**Kontur** adalah garis yang mengelilingi batas objek dalam gambar biner. Berbeda dengan edge detection yang hanya menandai piksel tepi, kontur menyimpan informasi **hierarki** dan **geometri** objek.

### Perbedaan Edge vs Kontur

```
Edge Detection:              Contour:
  ░░█░░█░░                    ┌───────┐
  █░░░░░░█                    │       │
  █░░░░░░█       →           │       │  (outline tersambung)
  █░░░░░░█                    │       │
  ░░█░░█░░                    └───────┘
(piksel terpisah)            (kurva kontinu)
```

**Demo Deteksi Kontur:**

![Demo Contour Detection](./images/demo-contour-detection.png)

---

## 🔍 Menemukan Kontur

### Kode Dasar

```python
import cv2
import numpy as np

# 1. Baca gambar
img = cv2.imread('objek.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Threshold atau Canny
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# Atau: edges = cv2.Canny(gray, 100, 200)

# 3. Temukan kontur
contours, hierarchy = cv2.findContours(binary, 
                                        cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE)

print(f"Jumlah kontur ditemukan: {len(contours)}")

# 4. Gambar kontur
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contours', img)
cv2.waitKey(0)
```

---

## 📊 Mode Retrieval (RETR)

Parameter `mode` menentukan bagaimana kontur diorganisasi:

| Mode | Penjelasan |
|---|---|
| `RETR_EXTERNAL` | Hanya kontur terluar (paling sering digunakan) |
| `RETR_LIST` | Semua kontur tanpa hierarki |
| `RETR_CCOMP` | 2 level: luar dan dalam |
| `RETR_TREE` | Hierarki lengkap (parent-child) |

### Visualisasi Hierarki

```
RETR_EXTERNAL:        RETR_TREE:
┌─────────────┐       ┌─────────────┐
│             │       │  ┌───────┐  │
│             │       │  │       │  │
│             │       │  │ ┌───┐ │  │
│             │       │  │ │   │ │  │
│             │       │  │ └───┘ │  │
│             │       │  └───────┘  │
└─────────────┘       └─────────────┘
   1 kontur              3 kontur
                        (dengan hierarchy)
```

**Demo RETR_EXTERNAL vs RETR_TREE:**

![Demo Hierarchy](./images/demo-hierarchy.png)

---

## 📐 Mode Approximation

Parameter `method` menentukan bagaimana titik kontur disimpan:

| Method | Penjelasan |
|---|---|
| `CHAIN_APPROX_NONE` | Simpan semua titik boundary |
| `CHAIN_APPROX_SIMPLE` | Kompres garis lurus (hemat memori) |

```
CHAIN_APPROX_NONE:           CHAIN_APPROX_SIMPLE:
. . . . . . . .              .             .
.             .              
.             .              
.             .              
.             .              
. . . . . . . .              .             .
  (setiap piksel)            (hanya sudut = 4 titik!)
```

---

## 🎨 Menggambar Kontur

```python
import cv2

# Gambar SEMUA kontur
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Gambar kontur ke-0 saja
cv2.drawContours(img, contours, 0, (255, 0, 0), 3)

# Gambar kontur terisi (filled)
cv2.drawContours(img, contours, -1, (0, 0, 255), -1)
```

### Parameter drawContours

| Parameter | Penjelasan |
|---|---|
| **image** | Gambar target untuk menggambar |
| **contours** | List kontur |
| **contourIdx** | Index kontur (-1 = semua) |
| **color** | Warna (B, G, R) |
| **thickness** | Ketebalan garis (-1 = filled) |

---

## 📏 Properti Kontur

### 1. Luas (Area)

```python
area = cv2.contourArea(contour)
print(f"Luas: {area} piksel²")
```

### 2. Keliling (Perimeter)

```python
perimeter = cv2.arcLength(contour, closed=True)
print(f"Keliling: {perimeter} piksel")
```

### 3. Bounding Rectangle

```python
# Bounding box tegak lurus
x, y, w, h = cv2.boundingRect(contour)
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Rotated bounding box (minimum area)
rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect)
box = np.int32(box)
cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
```

### 4. Minimum Enclosing Circle

```python
(x, y), radius = cv2.minEnclosingCircle(contour)
cv2.circle(img, (int(x), int(y)), int(radius), (255, 0, 0), 2)
```

### 5. Centroid (Pusat Massa)

```python
M = cv2.moments(contour)
if M["m00"] != 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cv2.circle(img, (cx, cy), 5, (255, 255, 0), -1)
```

**Demo Properti Kontur:**

![Demo Contour Properties](./images/demo-contour-properties.png)

---

## 🔷 Approximasi Bentuk

### Polygon Approximation (Douglas-Peucker)

Menyederhanakan kontur menjadi polygon dengan lebih sedikit titik:

```python
# Epsilon = akurasi (0.01-0.05 dari keliling)
epsilon = 0.02 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

print(f"Titik asli: {len(contour)}")
print(f"Titik setelah approx: {len(approx)}")

# Gambar polygon
cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
```

### Convex Hull

Membungkus kontur dengan polygon convex (tanpa lekukan):

```python
hull = cv2.convexHull(contour)
cv2.drawContours(img, [hull], 0, (255, 0, 0), 2)
```

---

## 🎯 Mendeteksi Bentuk

```python
import cv2
import numpy as np

def detect_shape(contour):
    """Deteksi bentuk berdasarkan jumlah sudut"""
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    
    if vertices == 3:
        return "Segitiga"
    elif vertices == 4:
        # Cek apakah persegi atau persegi panjang
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Persegi"
        else:
            return "Persegi Panjang"
    elif vertices == 5:
        return "Pentagon"
    elif vertices == 6:
        return "Hexagon"
    else:
        return "Lingkaran"

# Contoh penggunaan
for contour in contours:
    shape = detect_shape(contour)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(img, shape, (cx-50, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
```

**Demo Deteksi Bentuk:**

![Demo Shape Detection](./images/demo-shape-detection.png)

---

## 💻 Praktik Lengkap

```python
import cv2
import numpy as np

# Baca gambar
img = cv2.imread('shapes.jpg')
original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur dan threshold
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

# Temukan kontur
contours, hierarchy = cv2.findContours(binary, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)

# Analisis setiap kontur
for i, contour in enumerate(contours):
    # Filter kontur kecil (noise)
    area = cv2.contourArea(contour)
    if area < 100:
        continue
    
    # Gambar kontur
    cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
    
    # Centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
    
    # Label
    cv2.putText(img, f"A={int(area)}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

cv2.imshow('Contour Analysis', img)
cv2.waitKey(0)
```

---

## 📚 Rangkuman

| Fungsi | Kegunaan |
|---|---|
| `cv2.findContours()` | Menemukan semua kontur |
| `cv2.drawContours()` | Menggambar kontur |
| `cv2.contourArea()` | Menghitung luas |
| `cv2.arcLength()` | Menghitung keliling |
| `cv2.boundingRect()` | Bounding box |
| `cv2.moments()` | Properti statistik (centroid, dll) |
| `cv2.approxPolyDP()` | Simplify ke polygon |
| `cv2.convexHull()` | Bungkus dengan convex polygon |

---

## ➡️ Langkah Selanjutnya

Setelah menguasai kontur, selanjutnya kita akan belajar **Deteksi Bentuk dan Sudut** secara lebih mendalam!

[Lanjut ke: Deteksi Bentuk dan Sudut →](./03-shapes-and-corners.md)
