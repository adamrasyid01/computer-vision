# Deteksi Bentuk dan Sudut

> **Dari kontur ke bentuk geometris—pelajari cara mendeteksi lingkaran, garis, dan corner points dalam gambar.**

Bab ini membahas teknik lanjutan untuk mendeteksi pola geometris spesifik: **garis**, **lingkaran**, dan **sudut (corner points)**.

---

## 📏 Hough Line Transform

**Hough Transform** adalah teknik untuk mendeteksi **garis lurus** dalam gambar, bahkan jika garis tersebut terputus-putus.

### Konsep Hough Space

```
Ruang Gambar:              Ruang Hough:
    y                          θ
    ↑    /                     ↑
    │   /                      │    ╲
    │  /   ← garis             │     ╲ ← titik
    │ /                        │      ╲
    └──────→ x                 └────────→ ρ

Setiap titik di gambar → kurva di Hough space
Titik-titik segaris → kurva berpotongan di satu titik
```

### Kode Python - Standard Hough

```python
import cv2
import numpy as np

img = cv2.imread('road.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deteksi tepi
edges = cv2.Canny(gray, 50, 150)

# Standard Hough Transform
lines = cv2.HoughLines(edges, 
                       rho=1,           # Resolusi jarak (piksel)
                       theta=np.pi/180, # Resolusi sudut (radian)
                       threshold=100)   # Minimum votes

# Gambar garis
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
```

### Kode Python - Probabilistic Hough (Lebih Praktis!)

```python
import cv2
import numpy as np

img = cv2.imread('road.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# Probabilistic Hough Transform
lines = cv2.HoughLinesP(edges,
                        rho=1,
                        theta=np.pi/180,
                        threshold=50,
                        minLineLength=50,   # Panjang minimum
                        maxLineGap=10)      # Gap maksimum

# Gambar garis (lebih mudah!)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

### Parameter HoughLinesP

| Parameter | Penjelasan |
|---|---|
| **rho** | Resolusi jarak dalam piksel |
| **theta** | Resolusi sudut dalam radian |
| **threshold** | Minimum votes untuk garis |
| **minLineLength** | Panjang minimum garis |
| **maxLineGap** | Gap maksimum antara segmen |

**Demo Hough Line Transform:**

![Demo Line Detection](./images/demo-line-detection.png)

---

## ⭕ Hough Circle Transform

Mendeteksi **lingkaran** dalam gambar.

### Kode Python

```python
import cv2
import numpy as np

img = cv2.imread('coins.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur untuk kurangi noise
blurred = cv2.medianBlur(gray, 5)

# Deteksi lingkaran
circles = cv2.HoughCircles(blurred,
                           cv2.HOUGH_GRADIENT,
                           dp=1,           # Inverse ratio resolusi
                           minDist=50,     # Jarak minimum antar pusat
                           param1=100,     # Threshold Canny atas
                           param2=30,      # Threshold akumulator
                           minRadius=10,   # Radius minimum
                           maxRadius=100)  # Radius maksimum

# Gambar lingkaran
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        x, y, radius = circle
        # Gambar lingkaran
        cv2.circle(img, (x, y), radius, (0, 255, 0), 2)
        # Gambar pusat
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
```

### Parameter HoughCircles

| Parameter | Penjelasan |
|---|---|
| **dp** | Inverse ratio resolusi (1 = sama dengan gambar) |
| **minDist** | Jarak minimum antar pusat lingkaran |
| **param1** | Threshold Canny atas |
| **param2** | Threshold akumulator (lebih kecil = lebih banyak) |
| **minRadius** | Radius minimum |
| **maxRadius** | Radius maksimum |

**Demo Hough Circle Transform:**

![Demo Circle Detection](./images/demo-circle-detection.png)

---

## 📐 Corner Detection

**Corner** adalah titik di mana dua edge bertemu—sangat berguna untuk tracking dan matching.

### 1. Harris Corner Detector

Algoritma klasik untuk mendeteksi sudut.

```python
import cv2
import numpy as np

img = cv2.imread('chessboard.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Harris corner detection
# Parameter: blockSize, ksize, k
harris = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilasi untuk memperjelas
harris = cv2.dilate(harris, None)

# Tandai corner (threshold 1% dari max)
img[harris > 0.01 * harris.max()] = [0, 0, 255]

cv2.imshow('Harris Corners', img)
cv2.waitKey(0)
```

### Parameter Harris

| Parameter | Penjelasan |
|---|---|
| **blockSize** | Ukuran neighborhood |
| **ksize** | Ukuran kernel Sobel |
| **k** | Harris detector free parameter (0.04-0.06) |

---

### 2. Shi-Tomasi (Good Features to Track)

Versi **lebih baik** dari Harris, lebih stabil untuk tracking.

```python
import cv2
import numpy as np

img = cv2.imread('building.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi corner detection
corners = cv2.goodFeaturesToTrack(gray,
                                   maxCorners=100,     # Max jumlah corner
                                   qualityLevel=0.01,  # Threshold kualitas
                                   minDistance=10)     # Jarak minimum

# Gambar corner
if corners is not None:
    corners = np.int32(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

cv2.imshow('Shi-Tomasi Corners', img)
cv2.waitKey(0)
```

### Parameter goodFeaturesToTrack

| Parameter | Penjelasan |
|---|---|
| **maxCorners** | Jumlah maksimum corner |
| **qualityLevel** | Threshold relatif (0-1) |
| **minDistance** | Jarak minimum antar corner |

---

## ⚡ FAST Corner Detector

Algoritma **sangat cepat** untuk real-time applications.

```python
import cv2

img = cv2.imread('foto.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# FAST detector
fast = cv2.FastFeatureDetector_create(threshold=25)
keypoints = fast.detect(gray, None)

# Gambar keypoints
img_keypoints = cv2.drawKeypoints(img, keypoints, None, 
                                   color=(0, 255, 0))

print(f"Jumlah corner: {len(keypoints)}")
cv2.imshow('FAST Corners', img_keypoints)
cv2.waitKey(0)
```

---

## 📊 Perbandingan Metode Corner

| Metode | Kecepatan | Akurasi | Use Case |
|---|:---:|:---:|---|
| **Harris** | ⚡⚡ | ✅✅ | General purpose |
| **Shi-Tomasi** | ⚡⚡ | ✅✅✅ | Tracking (recommended) |
| **FAST** | ⚡⚡⚡ | ✅ | Real-time |

**Demo Harris vs Shi-Tomasi:**

![Demo Corner Detection](./images/demo-corner-detection.png)

---

## 💻 Praktik: Deteksi Garis & Lingkaran

```python
import cv2
import numpy as np

img = cv2.imread('shapes.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --- Deteksi Garis ---
edges = cv2.Canny(blurred, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                        minLineLength=30, maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
# --- Deteksi Lingkaran ---
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,
                           param1=100, param2=30,
                           minRadius=10, maxRadius=100)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0, :]:
        cv2.circle(img, (x, y), r, (255, 0, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

# --- Deteksi Corner ---
corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
if corners is not None:
    for corner in np.int32(corners):
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 4, (0, 255, 255), -1)

cv2.imshow('Detection', img)
cv2.waitKey(0)
```

---

## 📚 Rangkuman

| Fungsi | Kegunaan |
|---|---|
| `cv2.HoughLines()` | Deteksi garis (standard) |
| `cv2.HoughLinesP()` | Deteksi garis (probabilistic) |
| `cv2.HoughCircles()` | Deteksi lingkaran |
| `cv2.cornerHarris()` | Deteksi sudut Harris |
| `cv2.goodFeaturesToTrack()` | Deteksi sudut Shi-Tomasi |
| `cv2.FastFeatureDetector` | Deteksi sudut FAST |

---

## ➡️ Langkah Selanjutnya

Selamat! Anda telah menyelesaikan bagian **Feature Detection**. Selanjutnya, kita akan masuk ke dunia **Deep Learning untuk Computer Vision**!

[Lanjut ke: Neural Networks dalam CV →](../04-Deep-Learning-Intro/01-neural-networks-in-cv.md)
