import cv2
import os
import numpy as np

# =========================================================
# 1. LOKASI FOLDER
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER = os.path.join(BASE_DIR, "dataset")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "dataset_uji")

ROI_FILE = os.path.join(BASE_DIR, "roi_tanda_tangan.txt")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================================================
# 2. DAFTAR 9 DATASET
# =========================================================

IMAGE_FILES = [
    "01_HighQuality_Enhanced.jpg",
    "02_LowContrast.jpg",
    "03_Blurred.jpg",
    "04_HighNoise.jpg",
    "05_LowResolution_Upsampled.jpg",
    "06_Faded_Underexposed.jpg",
    "07_ColorShift_WarmTint.jpg",
    "08_JPEGCompression_Artifacts.jpg",
    "09_CombinedDegradation.jpg"
]


# =========================================================
# 3. PERSENTASE TINTA YANG DIPERTAHANKAN
# =========================================================

SISA_TINTA = 0.05

# 0.05 = sekitar 5% tinta tanda tangan dipertahankan.


# =========================================================
# 4. BACA ROI
# =========================================================

if not os.path.exists(ROI_FILE):
    print("ERROR: File ROI tidak ditemukan!")
    print()
    print("Pastikan file ini ada:")
    print(ROI_FILE)
    raise SystemExit

with open(ROI_FILE, "r") as f:
    data = f.read().strip().split(",")

x, y, w, h = map(int, data)

print("=" * 60)
print("ROI TANDA TANGAN")
print("=" * 60)
print(f"x = {x}")
print(f"y = {y}")
print(f"w = {w}")
print(f"h = {h}")

print()
print(f"Sisa tinta: {SISA_TINTA * 100:.0f}%")
print()


# =========================================================
# 5. RANDOM GENERATOR
# =========================================================

rng = np.random.default_rng(42)


# =========================================================
# 6. PROSES 9 GAMBAR
# =========================================================

berhasil = 0

for i, filename in enumerate(IMAGE_FILES, start=1):

    input_path = os.path.join(INPUT_FOLDER, filename)

    print("-" * 60)
    print(f"[{i}/9] Memproses: {filename}")

    # -----------------------------------------------------
    # CEK FILE
    # -----------------------------------------------------

    if not os.path.exists(input_path):
        print("  ERROR: File tidak ditemukan!")
        continue

    # -----------------------------------------------------
    # BACA GAMBAR
    # -----------------------------------------------------

    image = cv2.imread(input_path)

    if image is None:
        print("  ERROR: Gambar gagal dibaca!")
        continue

    # -----------------------------------------------------
    # AMBIL AREA TANDA TANGAN
    # -----------------------------------------------------

    roi = image[y:y+h, x:x+w].copy()

    # -----------------------------------------------------
    # GRAYSCALE
    # -----------------------------------------------------

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------
    # TENTUKAN BACKGROUND
    # -----------------------------------------------------

    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    background_value = int(np.median(border))

    # -----------------------------------------------------
    # DETEKSI PIXEL TINTA
    # -----------------------------------------------------
    # Pixel yang lebih gelap dari background dianggap
    # sebagai bagian dari tanda tangan.

    selisih = 25

    ink_mask = gray < (background_value - selisih)

    jumlah_tinta = np.count_nonzero(ink_mask)

    if jumlah_tinta == 0:
        print("  WARNING: Tidak ditemukan tinta tanda tangan.")
        continue

    # -----------------------------------------------------
    # PILIH SEBAGIAN TINTA UNTUK DIPERTAHANKAN
    # -----------------------------------------------------

    posisi_tinta = np.argwhere(ink_mask)

    jumlah_dipertahankan = max(
        1,
        int(len(posisi_tinta) * SISA_TINTA)
    )

    # Acak posisi tinta yang dipertahankan
    indeks = rng.choice(
        len(posisi_tinta),
        size=jumlah_dipertahankan,
        replace=False
    )

    posisi_terpilih = posisi_tinta[indeks]

    # -----------------------------------------------------
    # BUAT MASK TINTA YANG TERSISA
    # -----------------------------------------------------

    sisa_mask = np.zeros_like(gray, dtype=np.uint8)

    for py, px in posisi_terpilih:
        sisa_mask[py, px] = 255

    # -----------------------------------------------------
    # BUAT ROI BARU
    # -----------------------------------------------------

    roi_baru = np.full_like(
        roi,
        background_value
    )

    # Masukkan kembali sedikit tinta asli
    roi_baru[sisa_mask == 255] = roi[sisa_mask == 255]

    # -----------------------------------------------------
    # MASUKKAN KEMBALI KE GAMBAR ASLI
    # -----------------------------------------------------

    hasil = image.copy()

    hasil[y:y+h, x:x+w] = roi_baru

    # -----------------------------------------------------
    # NAMA FILE OUTPUT
    # -----------------------------------------------------

    nomor = f"{i:02d}"

    output_name = f"{nomor}_Sisa_TTD.jpg"

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_name
    )

    # -----------------------------------------------------
    # SIMPAN
    # -----------------------------------------------------

    cv2.imwrite(
        output_path,
        hasil,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    print(f"  Tinta asli       : {jumlah_tinta:,} pixel")
    print(f"  Tinta tersisa    : {jumlah_dipertahankan:,} pixel")
    print(f"  Hasil            : {output_name}")

    berhasil += 1


# =========================================================
# 7. SELESAI
# =========================================================

print()
print("=" * 60)
print("PEMBUATAN DATASET UJI SELESAI")
print("=" * 60)

print(f"Berhasil dibuat : {berhasil}/9 gambar")

print()
print("Lokasi hasil:")
print(OUTPUT_FOLDER)

print()
print("Isi folder dataset_uji:")

for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    if filename.endswith(".jpg"):
        print(" -", filename)

print()
print("Semua gambar masih memiliki sedikit sisa tinta")
print("tanda tangan untuk digunakan sebagai data uji.")
