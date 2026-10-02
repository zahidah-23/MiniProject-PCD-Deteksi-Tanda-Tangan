import cv2
import os
import numpy as np

# =========================
# PENGATURAN
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_IMAGE = os.path.join(
    BASE_DIR, "dataset", "01_HighQuality_Enhanced.jpg"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_uji")
ROI_FILE = os.path.join(BASE_DIR, "roi_tanda_tangan.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# BACA GAMBAR
# =========================
image = cv2.imread(INPUT_IMAGE)

if image is None:
    print("Gambar tidak ditemukan:")
    print(INPUT_IMAGE)
    raise SystemExit


# =========================
# CEK ROI
# =========================
if os.path.exists(ROI_FILE):

    print("ROI tanda tangan sudah tersimpan.")
    print("Tidak perlu memilih/crop ulang.")

    with open(ROI_FILE, "r") as f:
        data = f.read().strip().split(",")

    x, y, w, h = map(int, data)

else:

    print("ROI belum tersedia.")
    print("Pilih area tanda tangan Dekan.")

    height, width = image.shape[:2]

    scale = min(1000 / width, 800 / height, 1)

    preview = cv2.resize(
        image,
        (int(width * scale), int(height * scale))
    )

    x, y, w, h = cv2.selectROI(
        "Pilih Area Tanda Tangan Dekan",
        preview,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        print("Area tanda tangan belum dipilih.")
        raise SystemExit

    # Kembalikan koordinat ke ukuran gambar asli
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    # Simpan ROI
    with open(ROI_FILE, "w") as f:
        f.write(f"{x},{y},{w},{h}")

    print("ROI berhasil disimpan.")
    print("File:", ROI_FILE)


# =========================
# AMBIL AREA TTD
# =========================
roi = image[y:y+h, x:x+w].copy()

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


# =========================
# CARI WARNA BACKGROUND
# =========================
border = np.concatenate([
    gray[0, :],
    gray[-1, :],
    gray[:, 0],
    gray[:, -1]
])

background_value = int(np.median(border))


# =========================
# 1. TTD ASLI
# =========================
original = image.copy()


# =========================
# 2. TTD DIHAPUS
# =========================
blank = image.copy()

blank[y:y+h, x:x+w] = background_value


# =========================
# 3. SISAKAN SEDIKIT BERCAK
# =========================
specks = blank.copy()

rng = np.random.default_rng(42)

dark_pixels = gray < (background_value - 25)

# Hanya mempertahankan sekitar 2% piksel gelap
keep = rng.random(gray.shape) < 0.02

speck_mask = dark_pixels & keep

target = specks[y:y+h, x:x+w]

target[speck_mask] = roi[speck_mask]


# =========================
# SIMPAN HASIL
# =========================
cv2.imwrite(
    os.path.join(OUTPUT_DIR, "10_Tanpa_TTD.jpg"),
    blank
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "11_Sisa_Bercak.jpg"),
    specks
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "12_TTD_Asli.jpg"),
    original
)


# =========================
# SELESAI
# =========================
print()
print("====================================")
print("CITRA UJI BERHASIL DIBUAT")
print("====================================")
print("10_Tanpa_TTD.jpg")
print("11_Sisa_Bercak.jpg")
print("12_TTD_Asli.jpg")
print()
print("ROI tersimpan di:")
print(ROI_FILE)
print()
print("Program selesai.")
