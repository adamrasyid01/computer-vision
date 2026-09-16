# =============================================================================
# 🎨 FILTERING DAN BLURRING - Workshop Visi Komputer
# =============================================================================
# Notebook ini dapat dijalankan langsung di Google Colab.
# Semua dependensi (OpenCV, NumPy, Matplotlib) sudah tersedia secara default.
# =============================================================================

# ── Import library ─────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import urllib.request

# =============================================================================
# 🔧 HELPER: Download gambar sample dari internet (untuk Google Colab)
# =============================================================================

def download_sample_image(url: str, filename: str) -> np.ndarray:
    """Download gambar dari URL dan kembalikan sebagai array numpy (RGB)."""
    urllib.request.urlretrieve(url, filename)
    img_bgr = cv2.imread(filename)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

# Gunakan gambar Lena (standard benchmark image processing)
LENA_URL = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lena.png"

print("⬇️  Mengunduh gambar sample ...")
try:
    img = download_sample_image(LENA_URL, "lena.png")
    print("✅ Gambar berhasil diunduh!")
except Exception as e:
    print(f"⚠️  Tidak dapat mengunduh gambar ({e}) – menggunakan gambar sintetis.")
    rng = np.random.default_rng(42)
    img = rng.integers(80, 200, (256, 256, 3), dtype=np.uint8)
    img[100:160, 100:160] = [220, 80, 80]
    img[60:90,   60:90]   = [80, 80, 220]

print(f"   Ukuran gambar: {img.shape}")


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 1: KONVOLUSI (CONVOLUTION)
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("📐 BAGIAN 1: KONVOLUSI")
print("="*60)

# ── 1a. Definisi berbagai kernel ──────────────────────────────────────────────

kernel_identity = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]], dtype=np.float32)

kernel_average  = np.ones((3, 3), dtype=np.float32) / 9

kernel_sharpen  = np.array([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]], dtype=np.float32)

kernel_edge     = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]], dtype=np.float32)

kernel_emboss   = np.array([[-2, -1, 0],
                              [-1,  1, 1],
                              [ 0,  1, 2]], dtype=np.float32)

# Gaussian 3x3 aproksimasi
kernel_gaussian = np.array([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]], dtype=np.float32) / 16

kernels = {
    "Identity":     kernel_identity,
    "Average Blur": kernel_average,
    "Gaussian":     kernel_gaussian,
    "Sharpen":      kernel_sharpen,
    "Edge Detect":  kernel_edge,
    "Emboss":       kernel_emboss,
}

# ── 1b. Terapkan semua kernel ke gambar ───────────────────────────────────────
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("🔬 Efek Berbagai Kernel (filter2D)", fontsize=16, fontweight="bold")

for ax, (name, kernel) in zip(axes.flat, kernels.items()):
    result     = cv2.filter2D(img_bgr, -1, kernel)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    ax.imshow(result_rgb)
    ax.set_title(name, fontsize=12)
    ax.axis("off")

plt.tight_layout()
plt.show()

# ── 1c. Cetak nilai kernel ────────────────────────────────────────────────────
print("\n📋 Nilai Kernel yang Digunakan:")
for name, k in kernels.items():
    print(f"\n  [{name}]")
    print(np.round(k, 4))

# ── 1d. Demo konvolusi manual step-by-step ────────────────────────────────────
print("\n" + "-"*50)
print("🔢 Demo Konvolusi Manual (3x3 patch)")

patch     = np.array([[25, 35, 45],
                       [30, 40, 50],
                       [35, 45, 55]], dtype=np.float32)
avg_k     = np.ones((3, 3), dtype=np.float32) / 9
result_val = np.sum(patch * avg_k)

print(f"  Patch piksel :\n{patch}")
print(f"\n  Kernel Average (1/9):\n{np.round(avg_k, 4)}")
print(f"\n  Output[1,1]  = {result_val:.2f}  "
      f"← rata-rata dari {patch.flatten().astype(int).tolist()}")


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 2: SIMULASI JENIS-JENIS NOISE
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("🔇 BAGIAN 2: SIMULASI NOISE")
print("="*60)

def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """Tambahkan Gaussian noise ke gambar."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper(image: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """Tambahkan salt-and-pepper noise ke gambar."""
    noisy     = image.copy()
    num_salt  = int(amount * image.size * 0.5)
    num_pepper= int(amount * image.size * 0.5)
    # Salt (putih)
    coords    = [np.random.randint(0, dim, num_salt) for dim in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    # Pepper (hitam)
    coords    = [np.random.randint(0, dim, num_pepper) for dim in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

def add_speckle_noise(image: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    """Tambahkan speckle noise ke gambar."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) * (1 + noise), 0, 255).astype(np.uint8)
    return noisy

img_gaussian_noise = add_gaussian_noise(img)
img_sp_noise       = add_salt_pepper(img, amount=0.02)
img_speckle_noise  = add_speckle_noise(img)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("🔇 Jenis-jenis Noise", fontsize=15, fontweight="bold")

titles_n = ["Original", "Gaussian Noise\n(σ=25)", "Salt-and-Pepper\n(2%)", "Speckle Noise\n(σ=0.2)"]
images_n = [img, img_gaussian_noise, img_sp_noise, img_speckle_noise]

for ax, title, im in zip(axes, titles_n, images_n):
    ax.imshow(im)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.show()


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 3: TEKNIK BLURRING
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("🌫️  BAGIAN 3: TEKNIK BLURRING")
print("="*60)

# ── Helper ────────────────────────────────────────────────────────────────────
def to_bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ── 3a. Average Blur ──────────────────────────────────────────────────────────
print("\n1️⃣  Average Blur (Box Filter)")
sp_bgr = to_bgr(img_sp_noise)

average_3x3 = to_rgb(cv2.blur(sp_bgr, (3, 3)))
average_5x5 = to_rgb(cv2.blur(sp_bgr, (5, 5)))
average_9x9 = to_rgb(cv2.blur(sp_bgr, (9, 9)))

# Alternatif dengan filter2D
kernel_avg5   = np.ones((5, 5), np.float32) / 25
average_f2d   = to_rgb(cv2.filter2D(sp_bgr, -1, kernel_avg5))

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("1️⃣  Average Blur – Pengaruh Ukuran Kernel", fontsize=14, fontweight="bold")

for ax, title, im in zip(axes,
                          ["Noisy (Input)", "Average 3×3", "Average 5×5", "Average 9×9"],
                          [img_sp_noise, average_3x3, average_5x5, average_9x9]):
    ax.imshow(im); ax.set_title(title, fontsize=11); ax.axis("off")

plt.tight_layout()
plt.show()

# ── 3b. Gaussian Blur ─────────────────────────────────────────────────────────
print("2️⃣  Gaussian Blur")
gn_bgr = to_bgr(img_gaussian_noise)

gauss_s1 = to_rgb(cv2.GaussianBlur(gn_bgr, (5, 5), sigmaX=1))
gauss_s2 = to_rgb(cv2.GaussianBlur(gn_bgr, (5, 5), sigmaX=2))
gauss_s4 = to_rgb(cv2.GaussianBlur(gn_bgr, (9, 9), sigmaX=4))

print("  Kernel Gaussian 3×3 (aproksimasi × 16):")
print((kernel_gaussian * 16).astype(int))

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("2️⃣  Gaussian Blur – Pengaruh Sigma", fontsize=14, fontweight="bold")

for ax, title, im in zip(axes,
                          ["Noisy (Input)", "k=5×5, σ=1", "k=5×5, σ=2", "k=9×9, σ=4"],
                          [img_gaussian_noise, gauss_s1, gauss_s2, gauss_s4]):
    ax.imshow(im); ax.set_title(title, fontsize=11); ax.axis("off")

plt.tight_layout()
plt.show()

# ── 3c. Median Blur ───────────────────────────────────────────────────────────
print("3️⃣  Median Blur (terbaik untuk Salt-and-Pepper)")
sp_bgr2 = to_bgr(img_sp_noise)

median_3 = to_rgb(cv2.medianBlur(sp_bgr2, 3))
median_5 = to_rgb(cv2.medianBlur(sp_bgr2, 5))
median_7 = to_rgb(cv2.medianBlur(sp_bgr2, 7))

sample_patch = np.array([52, 55, 58, 60, 255, 61, 59, 54, 57])
sorted_patch = np.sort(sample_patch)
print(f"\n  Patch asli   : {sample_patch.tolist()}")
print(f"  Setelah sort : {sorted_patch.tolist()}")
print(f"  Nilai median : {int(np.median(sample_patch))}  ← noise (255) dihilangkan!")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("3️⃣  Median Blur – Efektif untuk Salt-and-Pepper", fontsize=14, fontweight="bold")

for ax, title, im in zip(axes,
                          ["Noisy (S&P 2%)", "Median k=3", "Median k=5", "Median k=7"],
                          [img_sp_noise, median_3, median_5, median_7]):
    ax.imshow(im); ax.set_title(title, fontsize=11); ax.axis("off")

plt.tight_layout()
plt.show()

# ── 3d. Bilateral Filter ──────────────────────────────────────────────────────
print("4️⃣  Bilateral Filter (blur + preserve edge)")
orig_bgr = to_bgr(img)

bilateral_mild   = to_rgb(cv2.bilateralFilter(orig_bgr, d=9, sigmaColor=25,  sigmaSpace=25))
bilateral_medium = to_rgb(cv2.bilateralFilter(orig_bgr, d=9, sigmaColor=75,  sigmaSpace=75))
bilateral_strong = to_rgb(cv2.bilateralFilter(orig_bgr, d=9, sigmaColor=150, sigmaSpace=150))

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("4️⃣  Bilateral Filter – Blur Sambil Jaga Tepi", fontsize=14, fontweight="bold")

for ax, title, im in zip(axes,
                          ["Original", "σ=25 (mild)", "σ=75 (medium)", "σ=150 (strong)"],
                          [img, bilateral_mild, bilateral_medium, bilateral_strong]):
    ax.imshow(im); ax.set_title(title, fontsize=11); ax.axis("off")

plt.tight_layout()
plt.show()


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 4: CUSTOM KERNEL
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("🛠️  BAGIAN 4: CUSTOM KERNEL")
print("="*60)

kernel_sharpen_c = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]], dtype=np.float32)

kernel_edge_enh  = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]], dtype=np.float32)

kernel_emboss_c  = np.array([[-2, -1, 0],
                               [-1,  1, 1],
                               [ 0,  1, 2]], dtype=np.float32)

kernel_sobel_x   = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)

kernel_sobel_y   = np.array([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]], dtype=np.float32)

def apply_kernel(img_rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Terapkan kernel ke gambar RGB dan kembalikan hasilnya sebagai RGB."""
    return to_rgb(cv2.filter2D(to_bgr(img_rgb), -1, kernel))

custom_results = {
    "Original":      img,
    "Sharpen":       apply_kernel(img, kernel_sharpen_c),
    "Edge Enhance":  apply_kernel(img, kernel_edge_enh),
    "Emboss":        apply_kernel(img, kernel_emboss_c),
    "Sobel-X":       apply_kernel(img, kernel_sobel_x),
    "Sobel-Y":       apply_kernel(img, kernel_sobel_y),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("🛠️  Custom Kernel Effects", fontsize=16, fontweight="bold")

for ax, (title, im) in zip(axes.flat, custom_results.items()):
    ax.imshow(im); ax.set_title(title, fontsize=12); ax.axis("off")

plt.tight_layout()
plt.show()


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 5: PERBANDINGAN LENGKAP (praktik dari materi)
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("📊 BAGIAN 5: PERBANDINGAN LENGKAP TEKNIK BLUR")
print("="*60)

noisy_bgr = to_bgr(img_sp_noise)
orig_bgr  = to_bgr(img)

results = {
    "Original":                   img,
    "Noisy\n(Salt-Pepper 2%)":    img_sp_noise,
    "Average Blur\n(5×5)":        to_rgb(cv2.blur(noisy_bgr, (5, 5))),
    "Gaussian Blur\n(5×5, σ=0)":  to_rgb(cv2.GaussianBlur(noisy_bgr, (5, 5), 0)),
    "Median Blur\n(k=5) ✓":       to_rgb(cv2.medianBlur(noisy_bgr, 5)),
    "Bilateral Filter\n(d=9)":    to_rgb(cv2.bilateralFilter(orig_bgr, 9, 75, 75)),
}

fig = plt.figure(figsize=(15, 10))
fig.suptitle("📊 Perbandingan Lengkap Teknik Blur", fontsize=17, fontweight="bold", y=1.01)
gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.1)

for idx, (title, im) in enumerate(results.items()):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    ax.imshow(im); ax.set_title(title, fontsize=12); ax.axis("off")

plt.show()

# ── Tabel ringkasan ───────────────────────────────────────────────────────────
print("\n📋 Tabel Perbandingan Teknik Blur:")
print(f"{'Teknik':<20} {'Kecepatan':<12} {'Noise Umum':<14} {'Salt-Pepper':<14} {'Preserve Edge'}")
print("-" * 75)
rows = [
    ("Average",   "⚡⚡⚡",  "✅",    "❌",     "❌"),
    ("Gaussian",  "⚡⚡⚡",  "✅✅",  "❌",     "❌"),
    ("Median",    "⚡⚡",    "✅",    "✅✅✅", "✅"),
    ("Bilateral", "⚡",      "✅✅",  "✅",     "✅✅✅"),
]
for name, speed, general, sp, edge in rows:
    print(f"  {name:<18} {speed:<12} {general:<14} {sp:<14} {edge}")


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# BAGIAN 6: EKSPLORASI PARAMETER (Widget Interaktif di Colab)
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*60)
print("🎮 BAGIAN 6: EKSPLORASI INTERAKTIF PARAMETER")
print("="*60)

try:
    from ipywidgets import interact, IntSlider, Dropdown

    def explore_blur(method="Gaussian", kernel_size=5, sigma=1,
                     bilateral_d=9, sigma_color=75, sigma_space=75):
        k   = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        bgr = to_bgr(img)

        if method == "Average":
            out = cv2.blur(bgr, (k, k))
        elif method == "Gaussian":
            out = cv2.GaussianBlur(bgr, (k, k), sigmaX=sigma)
        elif method == "Median":
            out = cv2.medianBlur(bgr, k)
        else:  # Bilateral
            out = cv2.bilateralFilter(bgr, bilateral_d, sigma_color, sigma_space)

        out_rgb = to_rgb(out)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img);     axes[0].set_title("Original");           axes[0].axis("off")
        axes[1].imshow(out_rgb); axes[1].set_title(f"{method} (k={k})"); axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    interact(
        explore_blur,
        method=Dropdown(options=["Average", "Gaussian", "Median", "Bilateral"], value="Gaussian"),
        kernel_size=IntSlider(min=3, max=21, step=2, value=5),
        sigma=IntSlider(min=1, max=10, value=1),
        bilateral_d=IntSlider(min=3, max=15, step=2, value=9),
        sigma_color=IntSlider(min=10, max=200, step=10, value=75),
        sigma_space=IntSlider(min=10, max=200, step=10, value=75),
    )

except ImportError:
    # Fallback: tampilkan grid statis jika bukan Jupyter/Colab
    print("   (Mode statis – jalankan di Colab/Jupyter untuk widget interaktif)")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("🎮 Eksplorasi Gaussian Blur – Berbagai σ & Kernel", fontsize=14, fontweight="bold")

    bgr    = to_bgr(img)
    combos = [(3,1),(3,3),(5,1),(5,3),(7,1),(7,3),(11,1),(11,3)]
    for ax, (k, s) in zip(axes.flat, combos):
        out = to_rgb(cv2.GaussianBlur(bgr, (k, k), sigmaX=s))
        ax.imshow(out); ax.set_title(f"k={k}×{k}, σ={s}", fontsize=10); ax.axis("off")

    plt.tight_layout()
    plt.show()


# =============================================================================
# RANGKUMAN
# =============================================================================
print("\n" + "="*60)
print("📚 RANGKUMAN")
print("="*60)
print("""
  ┌─────────────────┬──────────────────────────────────────────────────────┐
  │ Konsep          │ Penjelasan                                           │
  ├─────────────────┼──────────────────────────────────────────────────────┤
  │ Konvolusi       │ Operasi kernel yang meluncur di atas gambar          │
  │ Kernel          │ Matriks kecil (3×3, 5×5) untuk transformasi piksel  │
  │ Average Blur    │ Rata-rata sederhana, cepat tapi blur merata          │
  │ Gaussian Blur   │ Natural, bobot tengah lebih besar (dist. Gaussian)  │
  │ Median Blur     │ Terbaik untuk salt-and-pepper noise                  │
  │ Bilateral       │ Blur halus sambil mempertahankan tepi                │
  └─────────────────┴──────────────────────────────────────────────────────┘

  ✅ Kode selesai dijalankan!
  ➡️  Langkah selanjutnya: Thresholding (02-thresholding)
""")
