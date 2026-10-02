import cv2
import os
import numpy as np
import pandas as pd

# =========================================================
# 1. LOKASI FOLDER
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ASLI = os.path.join(BASE_DIR, "dataset")
DATASET_UJI = os.path.join(BASE_DIR, "dataset_uji")

OUTPUT_FOLDER = os.path.join(BASE_DIR, "hasil_uji_perbandingan")

ROI_FILE = os.path.join(BASE_DIR, "roi_tanda_tangan.txt")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================================================
# 2. NAMA 9 DATASET AWAL
# =========================================================

DATASET_FILES = [
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
# 3. NAMA 9 DATASET UJI
# =========================================================

UJI_FILES = [
    "01_Sisa_TTD.jpg",
    "02_Sisa_TTD.jpg",
    "03_Sisa_TTD.jpg",
    "04_Sisa_TTD.jpg",
    "05_Sisa_TTD.jpg",
    "06_Sisa_TTD.jpg",
    "07_Sisa_TTD.jpg",
    "08_Sisa_TTD.jpg",
    "09_Sisa_TTD.jpg"
]


# =========================================================
# 4. BATAS KLASIFIKASI
# =========================================================

BATAS_SIGNATURE = 3.0


# =========================================================
# 5. BACA ROI
# =========================================================

if not os.path.exists(ROI_FILE):

    print("ERROR: roi_tanda_tangan.txt tidak ditemukan!")
    print()
    print("Pastikan file ini ada:")
    print(ROI_FILE)

    raise SystemExit


with open(ROI_FILE, "r") as f:
    data = f.read().strip().split(",")

x, y, w, h = map(int, data)


print("=" * 70)
print("UJI PERBANDINGAN DATASET")
print("=" * 70)

print(f"ROI : x={x}, y={y}, w={w}, h={h}")
print(f"Batas klasifikasi : {BATAS_SIGNATURE}%")
print()


# =========================================================
# 6. FUNGSI SEGMENTASI
# =========================================================

def proses_gambar(image, nama_file, kelompok, nomor):

    # -----------------------------------------------------
    # CROP
    # -----------------------------------------------------

    crop = image[y:y+h, x:x+w]

    # -----------------------------------------------------
    # GRAYSCALE
    # -----------------------------------------------------

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------
    # GLOBAL THRESHOLD
    # -----------------------------------------------------

    _, global_threshold = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    # -----------------------------------------------------
    # OTSU
    # -----------------------------------------------------

    otsu_value, otsu_threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # -----------------------------------------------------
    # MORPHOLOGICAL OPENING
    # -----------------------------------------------------

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        otsu_threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    # -----------------------------------------------------
    # MORPHOLOGICAL CLOSING
    # -----------------------------------------------------

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    # -----------------------------------------------------
    # HITUNG PIXEL
    # -----------------------------------------------------

    total_pixel = closing.shape[0] * closing.shape[1]

    global_pixel = cv2.countNonZero(global_threshold)
    otsu_pixel = cv2.countNonZero(otsu_threshold)
    opening_pixel = cv2.countNonZero(opening)
    closing_pixel = cv2.countNonZero(closing)

    # -----------------------------------------------------
    # PERSENTASE
    # -----------------------------------------------------

    global_percent = global_pixel / total_pixel * 100
    otsu_percent = otsu_pixel / total_pixel * 100
    opening_percent = opening_pixel / total_pixel * 100
    closing_percent = closing_pixel / total_pixel * 100

    # -----------------------------------------------------
    # KLASIFIKASI
    # -----------------------------------------------------

    if closing_percent >= BATAS_SIGNATURE:
        hasil = "SIGNATURE PRESENT"
    else:
        hasil = "SIGNATURE ABSENT"

    # -----------------------------------------------------
    # FOLDER OUTPUT
    # -----------------------------------------------------

    nama_tanpa_ext = os.path.splitext(nama_file)[0]

    folder = os.path.join(
        OUTPUT_FOLDER,
        kelompok,
        nama_tanpa_ext
    )

    os.makedirs(folder, exist_ok=True)

    # -----------------------------------------------------
    # SIMPAN HASIL SEGMENTASI
    # -----------------------------------------------------

    cv2.imwrite(
        os.path.join(folder, "01_crop.jpg"),
        crop
    )

    cv2.imwrite(
        os.path.join(folder, "02_grayscale.jpg"),
        gray
    )

    cv2.imwrite(
        os.path.join(folder, "03_global_threshold.jpg"),
        global_threshold
    )

    cv2.imwrite(
        os.path.join(folder, "04_otsu_threshold.jpg"),
        otsu_threshold
    )

    cv2.imwrite(
        os.path.join(folder, "05_opening.jpg"),
        opening
    )

    cv2.imwrite(
        os.path.join(folder, "06_closing.jpg"),
        closing
    )

    # -----------------------------------------------------
    # DATA HASIL
    # -----------------------------------------------------

    return {
        "No": nomor,
        "Kelompok": kelompok,
        "Nama Gambar": nama_file,
        "Global (%)": round(global_percent, 2),
        "Otsu (%)": round(otsu_percent, 2),
        "Opening (%)": round(opening_percent, 2),
        "Closing (%)": round(closing_percent, 2),
        "Hasil Sistem": hasil
    }


# =========================================================
# 7. PROSES DATASET AWAL
# =========================================================

hasil_semua = []

print("MEMPROSES DATASET AWAL")
print("-" * 70)

for i, filename in enumerate(DATASET_FILES, start=1):

    path = os.path.join(DATASET_ASLI, filename)

    if not os.path.exists(path):

        print(f"[{i}/9] File tidak ditemukan: {filename}")
        continue

    image = cv2.imread(path)

    if image is None:

        print(f"[{i}/9] Gagal membaca: {filename}")
        continue

    hasil = proses_gambar(
        image,
        filename,
        "dataset_awal",
        i
    )

    hasil_semua.append(hasil)

    print(
        f"[{i}/9] {filename} -> "
        f"{hasil['Hasil Sistem']} "
        f"(Closing {hasil['Closing (%)']}%)"
    )


# =========================================================
# 8. PROSES DATASET UJI
# =========================================================

print()
print("MEMPROSES DATASET UJI")
print("-" * 70)

for i, filename in enumerate(UJI_FILES, start=1):

    path = os.path.join(DATASET_UJI, filename)

    if not os.path.exists(path):

        print(f"[{i}/9] File tidak ditemukan: {filename}")
        continue

    image = cv2.imread(path)

    if image is None:

        print(f"[{i}/9] Gagal membaca: {filename}")
        continue

    hasil = proses_gambar(
        image,
        filename,
        "dataset_uji",
        i
    )

    hasil_semua.append(hasil)

    print(
        f"[{i}/9] {filename} -> "
        f"{hasil['Hasil Sistem']} "
        f"(Closing {hasil['Closing (%)']}%)"
    )


# =========================================================
# 9. SIMPAN CSV
# =========================================================

if len(hasil_semua) == 0:

    print()
    print("Tidak ada gambar yang berhasil diproses.")
    raise SystemExit


df = pd.DataFrame(hasil_semua)

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "hasil_perbandingan.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# =========================================================
# 10. TAMPILKAN HASIL
# =========================================================

print()
print("=" * 70)
print("HASIL PERBANDINGAN")
print("=" * 70)

print(
    df.to_string(index=False)
)


# =========================================================
# 11. RINGKASAN
# =========================================================

print()
print("=" * 70)
print("RINGKASAN")
print("=" * 70)

awal = df[df["Kelompok"] == "dataset_awal"]
uji = df[df["Kelompok"] == "dataset_uji"]

print()
print("DATASET AWAL")
print(f"Jumlah gambar : {len(awal)}")
print(
    f"Rata-rata Closing : "
    f"{awal['Closing (%)'].mean():.2f}%"
)

print(
    "PRESENT :",
    (awal["Hasil Sistem"] == "SIGNATURE PRESENT").sum()
)

print(
    "ABSENT  :",
    (awal["Hasil Sistem"] == "SIGNATURE ABSENT").sum()
)


print()
print("DATASET UJI")
print(f"Jumlah gambar : {len(uji)}")
print(
    f"Rata-rata Closing : "
    f"{uji['Closing (%)'].mean():.2f}%"
)

print(
    "PRESENT :",
    (uji["Hasil Sistem"] == "SIGNATURE PRESENT").sum()
)

print(
    "ABSENT  :",
    (uji["Hasil Sistem"] == "SIGNATURE ABSENT").sum()
)


print()
print("=" * 70)
print("SELESAI")
print("=" * 70)

print()
print("Hasil segmentasi:")
print(OUTPUT_FOLDER)

print()
print("Hasil tabel:")
print(csv_path)