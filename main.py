import cv2
import numpy as np
import os
import csv

# ============================================================
# PENGATURAN FOLDER
# ============================================================

# Folder tempat 9 gambar berada
INPUT_FOLDER = r"C:\Users\HP\Downloads\T6_PCD\Praktikum citra-20261001T013856Z-1-001\Praktikum citra"

# Folder untuk menyimpan semua hasil
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "hasil_segmentasi")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# DAFTAR 9 GAMBAR
# ============================================================

image_files = [
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


# ============================================================
# CEK FILE
# ============================================================

print("==========================================")
print("CEK DATASET")
print("==========================================")

valid_files = []

for filename in image_files:

    path = os.path.join(INPUT_FOLDER, filename)

    if os.path.exists(path):
        valid_files.append(filename)
        print("[OK]  ", filename)
    else:
        print("[TIDAK DITEMUKAN]", filename)

if len(valid_files) == 0:
    print("\nTidak ada gambar yang ditemukan!")
    print("Periksa kembali lokasi folder.")
    exit()

print("\nJumlah gambar ditemukan:", len(valid_files))


# ============================================================
# 1. BUKA GAMBAR PERTAMA UNTUK MENENTUKAN CROP
# ============================================================

first_file = valid_files[0]
first_path = os.path.join(INPUT_FOLDER, first_file)

first_image = cv2.imread(first_path)

if first_image is None:
    print("Gambar pertama gagal dibaca.")
    exit()

print("\n==========================================")
print("CROP AREA TANDA TANGAN DEKAN")
print("==========================================")

print("Pilih area tanda tangan DEKAN.")
print("Klik kiri + tahan mouse.")
print("Tarik kotak mengelilingi tanda tangan.")
print("Tekan ENTER atau SPACE setelah selesai.")
print("Tekan C jika ingin membatalkan.")

# Ukuran maksimal tampilan
max_width = 1200
max_height = 800

height, width = first_image.shape[:2]

scale = min(
    max_width / width,
    max_height / height,
    1
)

display_width = int(width * scale)
display_height = int(height * scale)

display_image = cv2.resize(
    first_image,
    (display_width, display_height)
)

roi = cv2.selectROI(
    "PILIH AREA TANDA TANGAN DEKAN",
    display_image,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

x, y, w, h = roi

if w == 0 or h == 0:
    print("\nCrop dibatalkan.")
    exit()


# ============================================================
# 2. KEMBALIKAN KOORDINAT KE UKURAN ASLI
# ============================================================

x = int(x / scale)
y = int(y / scale)
w = int(w / scale)
h = int(h / scale)

print("\nKoordinat crop:")
print("X =", x)
print("Y =", y)
print("Width =", w)
print("Height =", h)


# ============================================================
# 3. MORPHOLOGICAL KERNEL
# ============================================================

kernel = np.ones(
    (3, 3),
    np.uint8
)


# ============================================================
# 4. HASIL CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "hasil_segmentasi.csv"
)

results = []


# ============================================================
# 5. PROSES SEMUA GAMBAR
# ============================================================

print("\n")
print("==========================================")
print("MEMPROSES SEMUA GAMBAR")
print("==========================================")


for number, filename in enumerate(valid_files, start=1):

    print(f"\n[{number}/{len(valid_files)}] {filename}")

    image_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:
        print("Gagal membaca gambar.")
        continue


    # --------------------------------------------------------
    # CROP
    # --------------------------------------------------------

    cropped = image[
        y:y+h,
        x:x+w
    ]

    # Nama folder berdasarkan nama gambar
    image_name = os.path.splitext(filename)[0]

    result_folder = os.path.join(
        OUTPUT_FOLDER,
        image_name
    )

    os.makedirs(
        result_folder,
        exist_ok=True
    )

    cv2.imwrite(
        os.path.join(
            result_folder,
            "01_crop_tanda_tangan.jpg"
        ),
        cropped
    )


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        cropped,
        cv2.COLOR_BGR2GRAY
    )

    cv2.imwrite(
        os.path.join(
            result_folder,
            "02_grayscale.jpg"
        ),
        gray
    )


    # --------------------------------------------------------
    # GLOBAL THRESHOLD
    # --------------------------------------------------------

    _, global_threshold = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    cv2.imwrite(
        os.path.join(
            result_folder,
            "03_global_threshold.jpg"
        ),
        global_threshold
    )


    # --------------------------------------------------------
    # OTSU THRESHOLD
    # --------------------------------------------------------

    otsu_value, otsu_threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    cv2.imwrite(
        os.path.join(
            result_folder,
            "04_otsu_threshold.jpg"
        ),
        otsu_threshold
    )


    # --------------------------------------------------------
    # MORPHOLOGICAL OPENING
    # --------------------------------------------------------

    opening = cv2.morphologyEx(
        otsu_threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    cv2.imwrite(
        os.path.join(
            result_folder,
            "05_opening.jpg"
        ),
        opening
    )


    # --------------------------------------------------------
    # MORPHOLOGICAL CLOSING
    # --------------------------------------------------------

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    cv2.imwrite(
        os.path.join(
            result_folder,
            "06_closing.jpg"
        ),
        closing
    )


    # --------------------------------------------------------
    # HITUNG FOREGROUND PIXEL
    # --------------------------------------------------------

    total_pixel = cropped.shape[0] * cropped.shape[1]

    foreground_global = cv2.countNonZero(
        global_threshold
    )

    foreground_otsu = cv2.countNonZero(
        otsu_threshold
    )

    foreground_opening = cv2.countNonZero(
        opening
    )

    foreground_closing = cv2.countNonZero(
        closing
    )


    # --------------------------------------------------------
    # HITUNG PERSENTASE FOREGROUND
    # --------------------------------------------------------

    percentage_global = (
        foreground_global / total_pixel
    ) * 100

    percentage_otsu = (
        foreground_otsu / total_pixel
    ) * 100

    percentage_opening = (
        foreground_opening / total_pixel
    ) * 100

    percentage_closing = (
        foreground_closing / total_pixel
    ) * 100


    # --------------------------------------------------------
    # SIMPAN HASIL
    # --------------------------------------------------------

    results.append([
        filename,
        total_pixel,
        foreground_global,
        round(percentage_global, 2),
        round(otsu_value, 2),
        foreground_otsu,
        round(percentage_otsu, 2),
        foreground_opening,
        round(percentage_opening, 2),
        foreground_closing,
        round(percentage_closing, 2)
    ])


    print("  Crop              :", cropped.shape[1], "x", cropped.shape[0])
    print("  Global foreground :", foreground_global)
    print("  Otsu threshold    :", round(otsu_value, 2))
    print("  Otsu foreground   :", foreground_otsu)
    print("  Opening           :", foreground_opening)
    print("  Closing           :", foreground_closing)

    print("  Berhasil.")


# ============================================================
# 6. SIMPAN DATA KE CSV
# ============================================================

with open(
    csv_path,
    mode="w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Nama Gambar",
        "Total Pixel",
        "Global Foreground Pixel",
        "Global Foreground (%)",
        "Nilai Otsu",
        "Otsu Foreground Pixel",
        "Otsu Foreground (%)",
        "Opening Foreground Pixel",
        "Opening (%)",
        "Closing Foreground Pixel",
        "Closing (%)"
    ])

    writer.writerows(results)


# ============================================================
# 7. TAMPILKAN HASIL AKHIR
# ============================================================

print("\n")
print("==========================================")
print("SELESAI")
print("==========================================")

print("Jumlah gambar diproses :", len(results))

print("\nFolder hasil:")
print(OUTPUT_FOLDER)

print("\nFile tabel:")
print(csv_path)

print("\nSemua 9 gambar sudah diproses.")


# ============================================================
# 8. TAMPILKAN RINGKASAN
# ============================================================

print("\n")
print("==========================================")
print("RINGKASAN HASIL")
print("==========================================")

for row in results:

    print("\n", row[0])

    print(
        "  Global  :",
        row[2],
        "pixel (",
        row[3],
        "%)"
    )

    print(
        "  Otsu    :",
        row[5],
        "pixel (",
        row[6],
        "%)"
    )

    print(
        "  Opening :",
        row[7],
        "pixel (",
        row[8],
        "%)"
    )

    print(
        "  Closing :",
        row[9],
        "pixel (",
        row[10],
        "%)"
    )


print("\n==========================================")
print("PROGRAM SELESAI")
print("==========================================")