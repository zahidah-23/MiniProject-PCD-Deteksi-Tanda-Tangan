import cv2
import os
import csv

# ============================================================
# 1. PENGATURAN FOLDER
# ============================================================

INPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset"
)

OUTPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hasil_segmentasi"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. DAFTAR GAMBAR
# ============================================================

nama_file = [
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
# 3. CEK FILE
# ============================================================

print("=" * 60)
print("PROGRAM DETEKSI TANDA TANGAN")
print("=" * 60)

print("\nMemeriksa file gambar...\n")

for nama in nama_file:
    path = os.path.join(INPUT_FOLDER, nama)

    if os.path.exists(path):
        print("[OK]", nama)
    else:
        print("[TIDAK DITEMUKAN]", nama)


# ============================================================
# 4. MEMBUKA GAMBAR PERTAMA
#    UNTUK MEMILIH AREA TANDA TANGAN
# ============================================================

gambar_pertama = os.path.join(INPUT_FOLDER, nama_file[0])

img = cv2.imread(gambar_pertama)

if img is None:
    print("\nGambar tidak dapat dibuka.")
    exit()


tinggi, lebar = img.shape[:2]

print("\nUkuran gambar pertama:")
print("Lebar  :", lebar)
print("Tinggi :", tinggi)


# ============================================================
# 5. MEMPERKECIL GAMBAR UNTUK TAMPILAN
# ============================================================

MAX_WIDTH = 1200
MAX_HEIGHT = 800

scale = min(MAX_WIDTH / lebar, MAX_HEIGHT / tinggi, 1)

display_width = int(lebar * scale)
display_height = int(tinggi * scale)

display_img = cv2.resize(
    img,
    (display_width, display_height)
)


# ============================================================
# 6. MEMILIH AREA TANDA TANGAN
# ============================================================

print("\n==============================================")
print("PILIH AREA TANDA TANGAN DEKAN")
print("==============================================")
print("1. Klik dan tahan mouse.")
print("2. Tarik kotak pada area tanda tangan.")
print("3. Lepaskan mouse.")
print("4. Tekan ENTER.")
print("==============================================\n")


roi = cv2.selectROI(
    "Pilih Area Tanda Tangan",
    display_img,
    False,
    False
)

cv2.destroyAllWindows()


x, y, w, h = roi

if w == 0 or h == 0:
    print("Area tanda tangan belum dipilih.")
    exit()


# ============================================================
# 7. MENGEMBALIKAN KOORDINAT KE UKURAN ASLI
# ============================================================

x = int(x / scale)
y = int(y / scale)
w = int(w / scale)
h = int(h / scale)

print("\nKoordinat area tanda tangan:")
print("X =", x)
print("Y =", y)
print("W =", w)
print("H =", h)


# ============================================================
# 8. PERSIAPAN CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "hasil_segmentasi.csv"
)

hasil_csv = []


# ============================================================
# 9. BATAS KLASIFIKASI
# ============================================================

BATAS_SIGNATURE = 3.0


# ============================================================
# 10. PROSES SEMUA GAMBAR
# ============================================================

for nama in nama_file:

    print("\n")
    print("=" * 60)
    print("Memproses:", nama)
    print("=" * 60)

    path_gambar = os.path.join(INPUT_FOLDER, nama)

    image = cv2.imread(path_gambar)

    if image is None:
        print("Gambar tidak dapat dibaca.")
        continue


    # --------------------------------------------------------
    # CROP
    # --------------------------------------------------------

    crop = image[
        y:y+h,
        x:x+w
    ]


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # GLOBAL THRESHOLD
    # --------------------------------------------------------

    global_threshold_value = 127

    _, global_thresh = cv2.threshold(
        gray,
        global_threshold_value,
        255,
        cv2.THRESH_BINARY_INV
    )


    # --------------------------------------------------------
    # OTSU THRESHOLD
    # --------------------------------------------------------

    otsu_value, otsu_thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )


    # --------------------------------------------------------
    # MORPHOLOGICAL OPENING
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    opening = cv2.morphologyEx(
        otsu_thresh,
        cv2.MORPH_OPEN,
        kernel
    )


    # --------------------------------------------------------
    # MORPHOLOGICAL CLOSING
    # --------------------------------------------------------

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )


    # ========================================================
    # 11. MENGHITUNG FOREGROUND PIXEL
    # ========================================================

    total_pixel = closing.shape[0] * closing.shape[1]


    global_foreground = cv2.countNonZero(
        global_thresh
    )

    otsu_foreground = cv2.countNonZero(
        otsu_thresh
    )

    opening_foreground = cv2.countNonZero(
        opening
    )

    closing_foreground = cv2.countNonZero(
        closing
    )


    # ========================================================
    # 12. MENGHITUNG PERSENTASE
    # ========================================================

    global_percent = (
        global_foreground / total_pixel
    ) * 100

    otsu_percent = (
        otsu_foreground / total_pixel
    ) * 100

    opening_percent = (
        opening_foreground / total_pixel
    ) * 100

    closing_percent = (
        closing_foreground / total_pixel
    ) * 100


    # ========================================================
    # 13. ATURAN KLASIFIKASI
    # ========================================================

    if closing_percent >= BATAS_SIGNATURE:

        hasil = "SIGNATURE PRESENT"

    else:

        hasil = "SIGNATURE ABSENT"


    # ========================================================
    # 14. MENAMPILKAN HASIL
    # ========================================================

    print("\nHasil:")
    print("Global Foreground :", global_foreground)
    print("Global (%)        :", round(global_percent, 2), "%")

    print("Otsu Value        :", round(otsu_value, 2))
    print("Otsu Foreground   :", otsu_foreground)
    print("Otsu (%)          :", round(otsu_percent, 2), "%")

    print("Opening Foreground:", opening_foreground)
    print("Opening (%)       :", round(opening_percent, 2), "%")

    print("Closing Foreground:", closing_foreground)
    print("Closing (%)       :", round(closing_percent, 2), "%")

    print("\nBatas klasifikasi :", BATAS_SIGNATURE, "%")
    print("HASIL             :", hasil)


    # ========================================================
    # 15. MEMBUAT FOLDER HASIL
    # ========================================================

    nama_dasar = os.path.splitext(nama)[0]

    folder_hasil = os.path.join(
        OUTPUT_FOLDER,
        nama_dasar
    )

    os.makedirs(
        folder_hasil,
        exist_ok=True
    )


    # ========================================================
    # 16. MENYIMPAN HASIL GAMBAR
    # ========================================================

    cv2.imwrite(
        os.path.join(folder_hasil, "01_crop.jpg"),
        crop
    )

    cv2.imwrite(
        os.path.join(folder_hasil, "02_grayscale.jpg"),
        gray
    )

    cv2.imwrite(
        os.path.join(folder_hasil, "03_global_threshold.jpg"),
        global_thresh
    )

    cv2.imwrite(
        os.path.join(folder_hasil, "04_otsu_threshold.jpg"),
        otsu_thresh
    )

    cv2.imwrite(
        os.path.join(folder_hasil, "05_opening.jpg"),
        opening
    )

    cv2.imwrite(
        os.path.join(folder_hasil, "06_closing.jpg"),
        closing
    )


    # ========================================================
    # 17. MEMASUKKAN HASIL KE CSV
    # ========================================================

    hasil_csv.append([
        nama,
        total_pixel,

        global_foreground,
        round(global_percent, 2),

        round(otsu_value, 2),
        otsu_foreground,
        round(otsu_percent, 2),

        opening_foreground,
        round(opening_percent, 2),

        closing_foreground,
        round(closing_percent, 2),

        hasil
    ])


# ============================================================
# 18. MENYIMPAN CSV
# ============================================================

with open(
    csv_path,
    mode="w",
    newline="",
    encoding="utf-8"
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
        "Closing (%)",

        "Hasil Klasifikasi"
    ])

    writer.writerows(hasil_csv)


# ============================================================
# 19. SELESAI
# ============================================================

print("\n")
print("=" * 60)
print("SEMUA GAMBAR SELESAI DIPROSES")
print("=" * 60)

print("\nHasil disimpan di:")
print(OUTPUT_FOLDER)

print("\nFile CSV:")
print(csv_path)

print("\nAturan klasifikasi:")
print("Closing Foreground (%) >= 3%")
print("     -> SIGNATURE PRESENT")

print("Closing Foreground (%) < 3%")
print("     -> SIGNATURE ABSENT")

print("\nProgram selesai.")