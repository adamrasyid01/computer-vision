# =============================================================================
# 🔍 GRADIEN DAN DETEKSI TEPI - Workshop Visi Komputer
# =============================================================================
# Notebook ini dapat dijalankan langsung di Google Colab.
# Semua dependensi (OpenCV, NumPy, Matplotlib) sudah tersedia secara default.
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import urllib.request

# =============================================================================
# 🔧 HELPER: Download gambar & konversi warna
# =============================================================================

def download_image(url: str, filename: str) -> np.ndarray:
    """Download gambar, simpan ke disk, kembalikan sebagai grayscale."""
    urllib.request.urlretrieve(url, filename)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return img

def show_images(images: dict, title: str, cmap: str = "gray", figsize=(16, 5)):
    """Helper untuk menampilkan beberapa gambar dalam satu baris."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=15, fontweight="bold")
    for ax, (name, im) in zip(axes, images.items()):
        ax.imshow(im, cmap=cmap)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# ── Download gambar sample ────────────────────────────────────────────────────
LENA_URL = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lena.png"

print("⬇️  Mengunduh gambar sample ...")
try:
    img = download_image(LENA_URL, "lena.png")
    print(f"✅ Berhasil! Ukuran: {img.shape}")
except Exception as e:
    print(f"⚠️  Gagal download ({e}) – membuat gambar sintetis.")
    # Buat gambar sintetis dengan beberapa tepi jelas
    img = np.zeros((256, 256), dtype=np.uint8)
    img[60:120, 60:120]   = 180   # kotak abu-abu
    img[130:200, 130:200] = 230   # kotak terang
    img[40:220, 40:42]    = 255   # garis vertikal
    img[40:42,  40:220]   = 255   # garis horizontal
    rr, cc = np.ogrid[:256, :256]
    circle_mask = (rr - 128)**2 + (cc - 128)**2 < 60**2
    img[circle_mask] = np.maximum(img[circle_mask], 120)
    print("   Gambar sintetis dibuat.")

# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 1: KONSEP TEPI (EDGE) & GRADIEN
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("📐 BAGIAN 1: KONSEP TEPI & GRADIEN")
print("="*60)

# ── 1a. Visualisasi profil intensitas (1 baris piksel) ───────────────────────
row_idx = img.shape[0] // 2          # ambil baris tengah gambar
profile = img[row_idx, :].astype(float)
gradient_1d = np.gradient(profile)   # turunan numerik sederhana

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("📐 Konsep Tepi & Gradien (profil 1 baris piksel)", fontsize=14, fontweight="bold")

axes[0].imshow(img, cmap="gray")
axes[0].axhline(row_idx, color="red", linewidth=1.5, linestyle="--")
axes[0].set_title("Gambar (baris merah = profil)", fontsize=11)
axes[0].axis("off")

axes[1].plot(profile, color="royalblue", linewidth=1.2)
axes[1].set_title("Profil Intensitas (I)", fontsize=11)
axes[1].set_xlabel("Posisi piksel (x)")
axes[1].set_ylabel("Nilai intensitas")
axes[1].grid(True, alpha=0.3)

axes[2].plot(gradient_1d, color="crimson", linewidth=1.2)
axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")
axes[2].set_title("Gradien dI/dx  (puncak = tepi!)", fontsize=11)
axes[2].set_xlabel("Posisi piksel (x)")
axes[2].set_ylabel("Nilai gradien")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ── 1b. Penjelasan rumus gradien ─────────────────────────────────────────────
print("""
📋 Rumus Gradien 2D:
  ┌────────────┬──────────────────┬─────────────────────────────────┐
  │ Komponen   │ Rumus            │ Arti                            │
  ├────────────┼──────────────────┼─────────────────────────────────┤
  │ Gx         │ ∂I/∂x            │ Perubahan di arah horizontal    │
  │ Gy         │ ∂I/∂y            │ Perubahan di arah vertikal      │
  │ Magnitude  │ √(Gx² + Gy²)    │ Kekuatan tepi                   │
  │ Direction  │ arctan(Gy / Gx)  │ Arah tepi (dalam radian)        │
  └────────────┴──────────────────┴─────────────────────────────────┘
""")

# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 2: OPERATOR SOBEL
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("🔧 BAGIAN 2: OPERATOR SOBEL")
print("="*60)

# ── Kernel Sobel manual (untuk edukasi) ──────────────────────────────────────
K_SOBEL_X = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)

K_SOBEL_Y = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)

print("  Kernel Sobel X (deteksi tepi vertikal |):")
print(K_SOBEL_X)
print("\n  Kernel Sobel Y (deteksi tepi horizontal ─):")
print(K_SOBEL_Y)

# ── Terapkan Sobel dengan OpenCV ─────────────────────────────────────────────
# Gunakan CV_64F agar piksel negatif tidak terpotong
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude: √(Gx² + Gy²)
sobel_mag = cv2.magnitude(sobel_x, sobel_y)

# Arah tepi (direction) dalam derajat
sobel_dir = np.degrees(np.arctan2(sobel_y, sobel_x))

# Konversi ke uint8 untuk ditampilkan
sobel_x_disp   = cv2.convertScaleAbs(sobel_x)
sobel_y_disp   = cv2.convertScaleAbs(sobel_y)
sobel_mag_disp = cv2.convertScaleAbs(sobel_mag)

show_images(
    {"Original":   img,
     "Sobel X\n(tepi vertikal)":   sobel_x_disp,
     "Sobel Y\n(tepi horizontal)": sobel_y_disp,
     "Magnitude\n√(Gx²+Gy²)":     sobel_mag_disp},
    "🔧 Operator Sobel",
    figsize=(18, 4)
)

# ── Visualisasi direction (arah tepi) ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("🔧 Sobel: Magnitude & Direction", fontsize=14, fontweight="bold")

im0 = axes[0].imshow(sobel_mag_disp, cmap="hot")
axes[0].set_title("Magnitude (kekuatan tepi)", fontsize=11)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(sobel_dir, cmap="hsv")
axes[1].set_title("Direction (arah tepi, dalam derajat)", fontsize=11)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

plt.tight_layout()
plt.show()

# ── Pengaruh parameter ksize ──────────────────────────────────────────────────
print("\n  Pengaruh ksize pada Sobel:")
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("🔧 Sobel – Pengaruh ksize", fontsize=14, fontweight="bold")

axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")

for ax, k in zip(axes[1:], [1, 3, 5]):
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)
    mag = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
    ax.imshow(mag, cmap="gray")
    ax.set_title(f"ksize={k}", fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.show()


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 3: OPERATOR SCHARR
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("🔧 BAGIAN 3: OPERATOR SCHARR")
print("="*60)

# ── Scharr ───────────────────────────────────────────────────────────────────
scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharr_mag = cv2.magnitude(scharr_x, scharr_y)

scharr_x_disp   = cv2.convertScaleAbs(scharr_x)
scharr_y_disp   = cv2.convertScaleAbs(scharr_y)
scharr_mag_disp = cv2.convertScaleAbs(scharr_mag)

print("  Kernel Scharr X (bawaan OpenCV, lebih akurat dari Sobel 3×3):")
K_SCHARR_X = np.array([[-3,  0,  3],
                        [-10, 0, 10],
                        [-3,  0,  3]], dtype=np.float32)
print(K_SCHARR_X)

show_images(
    {"Original":       img,
     "Scharr X":       scharr_x_disp,
     "Scharr Y":       scharr_y_disp,
     "Magnitude":      scharr_mag_disp},
    "🔧 Operator Scharr (lebih akurat dari Sobel 3×3)",
    figsize=(18, 4)
)

# ── Sobel vs Scharr side-by-side ─────────────────────────────────────────────
show_images(
    {"Original":         img,
     "Sobel Magnitude":  sobel_mag_disp,
     "Scharr Magnitude": scharr_mag_disp},
    "⚖️  Sobel vs Scharr",
    figsize=(14, 4)
)


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 4: OPERATOR LAPLACIAN
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("🔧 BAGIAN 4: OPERATOR LAPLACIAN")
print("="*60)

K_LAPLACIAN = np.array([[0,  1, 0],
                         [1, -4, 1],
                         [0,  1, 0]], dtype=np.float32)

print("  Kernel Laplacian (turunan ke-2, semua arah):")
print(K_LAPLACIAN)

# ── Laplacian tanpa blur ──────────────────────────────────────────────────────
laplacian_raw = cv2.Laplacian(img, cv2.CV_64F)
laplacian_disp = cv2.convertScaleAbs(laplacian_raw)

# ── Laplacian dengan blur dulu (lebih bersih) ─────────────────────────────────
img_blur = cv2.GaussianBlur(img, (3, 3), 0)
laplacian_blur = cv2.convertScaleAbs(cv2.Laplacian(img_blur, cv2.CV_64F))

show_images(
    {"Original":               img,
     "Laplacian (tanpa blur)": laplacian_disp,
     "Laplacian (+ Gaussian)": laplacian_blur},
    "🔧 Operator Laplacian – Turunan Kedua (semua arah)",
    figsize=(14, 4)
)

print("""
  ℹ️  Catatan Laplacian:
     • Mendeteksi tepi di SEMUA arah sekaligus
     • Menggunakan turunan ke-2 (lebih sensitif terhadap noise)
     • Sangat disarankan: blur dulu sebelum Laplacian
     • Tidak memberikan informasi arah tepi
""")


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 5: CANNY EDGE DETECTION
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("⭐ BAGIAN 5: CANNY EDGE DETECTION")
print("="*60)

print("""
  Pipeline Canny (5 Langkah):
  ┌─────────────────────────────────────────────────────┐
  │  1. Gaussian Blur   → Kurangi noise                 │
  │         ↓                                           │
  │  2. Hitung Gradien  → Sobel X & Y                  │
  │         ↓                                           │
  │  3. Non-Maximum     → Tipiskan tepi (1 piksel)      │
  │     Suppression                                     │
  │         ↓                                           │
  │  4. Double Threshold→ Tepi kuat / lemah / bukan    │
  │         ↓                                           │
  │  5. Hysteresis      → Sambungkan tepi lemah→kuat   │
  │         ↓                                           │
  │  OUTPUT: Tepi bersih & tipis 1 piksel               │
  └─────────────────────────────────────────────────────┘
""")

# ── 5a. Canny dasar ───────────────────────────────────────────────────────────
img_blur5 = cv2.GaussianBlur(img, (5, 5), 0)
canny_basic = cv2.Canny(img, 100, 200)                    # tanpa blur
canny_clean = cv2.Canny(img_blur5, 100, 200)              # dengan blur

show_images(
    {"Original":           img,
     "Canny (tanpa blur)": canny_basic,
     "Canny (+ Gaussian)": canny_clean},
    "⭐ Canny – Perbandingan Dengan/Tanpa Pre-Blur",
    figsize=(14, 4)
)

# ── 5b. Pengaruh threshold ────────────────────────────────────────────────────
print("  Pengaruh nilai threshold pada Canny:")

thresholds = [(20, 60), (50, 150), (100, 200), (150, 300)]
fig, axes = plt.subplots(1, len(thresholds) + 1, figsize=(20, 4))
fig.suptitle("⭐ Canny – Pengaruh Threshold", fontsize=14, fontweight="bold")

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original", fontsize=11)
axes[0].axis("off")

for ax, (t1, t2) in zip(axes[1:], thresholds):
    result = cv2.Canny(img_blur5, t1, t2)
    ax.imshow(result, cmap="gray")
    ax.set_title(f"t1={t1}, t2={t2}\nratio={t2//t1}:1", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()

# ── 5c. Demo langkah-langkah Canny secara manual ────────────────────────────
print("\n  Demo langkah Canny step-by-step:")

# Step 1: Gaussian Blur
step1_blur = cv2.GaussianBlur(img, (5, 5), 1.4)

# Step 2: Gradien (Sobel)
Gx = cv2.Sobel(step1_blur, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(step1_blur, cv2.CV_64F, 0, 1, ksize=3)
step2_mag = cv2.convertScaleAbs(cv2.magnitude(Gx, Gy))

# Step 3-5: Gunakan Canny penuh sebagai hasil akhir
step5_canny = cv2.Canny(step1_blur, 100, 200)

show_images(
    {"Original":              img,
     "Step 1: Gaussian Blur": step1_blur,
     "Step 2: Gradien Mag.":  step2_mag,
     "Step 3-5: Canny Output":step5_canny},
    "⭐ Langkah-langkah Canny",
    figsize=(18, 4)
)


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 6: PERBANDINGAN LENGKAP SEMUA METODE (dari materi praktik)
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("📊 BAGIAN 6: PERBANDINGAN SEMUA METODE")
print("="*60)

# Semua metode menggunakan gambar yang sama (dengan pre-blur)
blurred = cv2.GaussianBlur(img, (3, 3), 0)

# Sobel
sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_final = cv2.convertScaleAbs(cv2.magnitude(sx, sy))

# Scharr
scx = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
scy = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
scharr_final = cv2.convertScaleAbs(cv2.magnitude(scx, scy))

# Laplacian
lap_final = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_64F))

# Canny
canny_final = cv2.Canny(blurred, 100, 200)

fig = plt.figure(figsize=(15, 10))
fig.suptitle("📊 Perbandingan Lengkap Metode Deteksi Tepi", fontsize=17, fontweight="bold", y=1.01)
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.1)

datasets = [
    ("Original",        img,           "gray"),
    ("Sobel ⚡⚡⚡",     sobel_final,   "gray"),
    ("Scharr ⚡⚡⚡",    scharr_final,  "gray"),
    ("Laplacian ⚡⚡⚡", lap_final,     "gray"),
    ("Canny ✓ (Best)",  canny_final,   "gray"),
]

for idx, (title, im, cmap) in enumerate(datasets):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    ax.imshow(im, cmap=cmap)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

# Cell terakhir: tabel teks
ax_table = fig.add_subplot(gs[1, 2])
ax_table.axis("off")
table_data = [
    ["Metode",     "Kecepatan", "Akurasi", "Noise"],
    ["Sobel",      "⚡⚡⚡",    "✅",      "Sedang"],
    ["Scharr",     "⚡⚡⚡",    "✅✅",    "Sedang"],
    ["Laplacian",  "⚡⚡⚡",    "✅",      "Tinggi"],
    ["Canny",      "⚡⚡",      "✅✅✅",  "Rendah"],
]
table = ax_table.table(cellText=table_data[1:],
                       colLabels=table_data[0],
                       cellLoc="center",
                       loc="center",
                       bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#ecf0f1")

plt.show()

# ── Tabel di konsol ───────────────────────────────────────────────────────────
print(f"\n{'Metode':<14} {'Kecepatan':<12} {'Akurasi':<10} {'Noise Sensitivity'}")
print("-" * 55)
rows = [
    ("Sobel",      "⚡⚡⚡", "✅",    "Sedang"),
    ("Scharr",     "⚡⚡⚡", "✅✅",  "Sedang"),
    ("Laplacian",  "⚡⚡⚡", "✅",    "Tinggi"),
    ("Canny",      "⚡⚡",   "✅✅✅","Rendah"),
]
for name, speed, acc, noise in rows:
    print(f"  {name:<12} {speed:<12} {acc:<10} {noise}")


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 7: EKSPLORASI INTERAKTIF CANNY (Widget Colab)
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("🎮 BAGIAN 7: EKSPLORASI INTERAKTIF CANNY")
print("="*60)

try:
    from ipywidgets import interact, IntSlider, Dropdown

    def explore_edges(method="Canny", t1=100, t2=200, ksize=3, blur_ksize=5):
        blur_k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        blurred_i = cv2.GaussianBlur(img, (blur_k, blur_k), 0)

        if method == "Sobel":
            sx_i = cv2.Sobel(blurred_i, cv2.CV_64F, 1, 0, ksize=ksize)
            sy_i = cv2.Sobel(blurred_i, cv2.CV_64F, 0, 1, ksize=ksize)
            result = cv2.convertScaleAbs(cv2.magnitude(sx_i, sy_i))
        elif method == "Scharr":
            sx_i = cv2.Scharr(blurred_i, cv2.CV_64F, 1, 0)
            sy_i = cv2.Scharr(blurred_i, cv2.CV_64F, 0, 1)
            result = cv2.convertScaleAbs(cv2.magnitude(sx_i, sy_i))
        elif method == "Laplacian":
            result = cv2.convertScaleAbs(cv2.Laplacian(blurred_i, cv2.CV_64F))
        else:  # Canny
            result = cv2.Canny(blurred_i, t1, t2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img, cmap="gray");    axes[0].set_title("Original");             axes[0].axis("off")
        axes[1].imshow(result, cmap="gray"); axes[1].set_title(f"{method} Output");     axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    interact(
        explore_edges,
        method=Dropdown(options=["Canny", "Sobel", "Scharr", "Laplacian"], value="Canny"),
        t1=IntSlider(min=10,  max=300, step=10, value=100, description="threshold1"),
        t2=IntSlider(min=30,  max=600, step=10, value=200, description="threshold2"),
        ksize=Dropdown(options=[1, 3, 5, 7], value=3, description="ksize (Sobel)"),
        blur_ksize=IntSlider(min=1, max=15, step=2, value=5, description="blur ksize"),
    )
    print("✅ Widget interaktif aktif!")

except ImportError:
    print("   (Mode statis – jalankan di Colab/Jupyter untuk widget interaktif)")

    # Fallback: grid eksplorasi Canny dengan berbagai threshold
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle("🎮 Eksplorasi Canny – Berbagai Kombinasi Threshold & Blur", fontsize=14, fontweight="bold")

    blur_sizes   = [3, 5]
    thresh_pairs = [(20, 40), (50, 100), (100, 200), (150, 300)]

    for row, bk in enumerate(blur_sizes):
        blr = cv2.GaussianBlur(img, (bk, bk), 0)
        for col, (t1, t2) in enumerate(thresh_pairs):
            out = cv2.Canny(blr, t1, t2)
            axes[row][col].imshow(out, cmap="gray")
            axes[row][col].set_title(f"blur={bk}, t=({t1},{t2})", fontsize=9)
            axes[row][col].axis("off")

    # Baris terakhir: gambar original + tiga varian terbaik
    axes[2][0].imshow(img, cmap="gray"); axes[2][0].set_title("Original"); axes[2][0].axis("off")
    for i, (t1, t2) in enumerate([(50,150),(100,200),(150,300)]):
        out = cv2.Canny(cv2.GaussianBlur(img,(5,5),0), t1, t2)
        axes[2][i+1].imshow(out, cmap="gray")
        axes[2][i+1].set_title(f"Rekomendasi t=({t1},{t2})", fontsize=9)
        axes[2][i+1].axis("off")

    plt.tight_layout()
    plt.show()


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# RANGKUMAN
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("📚 RANGKUMAN")
print("="*60)
print("""
  ┌──────────────┬──────────────────────┬───────────────────────────────────┐
  │ Metode       │ Fungsi OpenCV        │ Karakteristik                     │
  ├──────────────┼──────────────────────┼───────────────────────────────────┤
  │ Sobel        │ cv2.Sobel()          │ Gradien X atau Y, cepat           │
  │ Scharr       │ cv2.Scharr()         │ Lebih akurat dari Sobel 3×3       │
  │ Laplacian    │ cv2.Laplacian()      │ Semua arah, sensitif noise        │
  │ Canny        │ cv2.Canny()          │ Multi-stage, hasil terbaik ✓      │
  └──────────────┴──────────────────────┴───────────────────────────────────┘

  💡 Cheat Sheet Canny:
     blurred = cv2.GaussianBlur(img, (5,5), 0)
     edges   = cv2.Canny(blurred, 100, 200)
     # Aturan praktis: ratio t2:t1 = 2:1 atau 3:1

  ✅ Kode selesai dijalankan!
  ➡️  Langkah selanjutnya: Contours (02-contours)
""")
