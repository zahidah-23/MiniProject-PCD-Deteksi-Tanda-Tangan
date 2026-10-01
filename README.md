# Mini Project PCD - Deteksi Tanda Tangan

## Deskripsi

Mini project ini merupakan implementasi pengolahan citra digital untuk mendeteksi keberadaan tanda tangan pada dokumen.

Tahapan pengolahan citra yang digunakan meliputi:

1. Crop area tanda tangan
2. Konversi citra ke grayscale
3. Global Thresholding
4. Otsu Thresholding
5. Operasi morfologi Opening
6. Operasi morfologi Closing
7. Perhitungan jumlah foreground pixel
8. Penentuan keberadaan tanda tangan

Pada project ini, area tanda tangan yang dianalisis adalah tanda tangan Dekan pada dokumen ijazah universitas.

## Teknologi yang Digunakan

- Python
- OpenCV
- NumPy
- Pandas

## How to Run

### 1. Clone Repository

Buka Command Prompt atau PowerShell, kemudian jalankan:

```bash
git clone LINK_REPOSITORY_GITHUB
```

Kemudian masuk ke folder project:

```bash
cd MiniProject-PCD-Deteksi-Tanda-Tangan
```

### 2. Install Library

Install library yang diperlukan dengan perintah:

```bash
pip install -r requirements.txt
```

### 3. Jalankan Program

Jalankan program dengan:

```bash
python main.py
```

### 4. Memilih Area Tanda Tangan

Setelah gambar ditampilkan:

1. Klik dan tahan mouse pada awal area tanda tangan.
2. Geser mouse sampai seluruh area tanda tangan terseleksi.
3. Lepaskan mouse.
4. Tekan tombol `ENTER`.

Area yang dipilih akan digunakan sebagai area crop.

### 5. Hasil Pengolahan

Program melakukan beberapa tahap pengolahan:

- Grayscale
- Global Thresholding
- Otsu Thresholding
- Opening
- Closing

Hasil pengolahan disimpan pada folder:

```text
hasil_segmentasi/
```

Program juga menghasilkan file:

```text
hasil_segmentasi.csv
```

File CSV berisi hasil perhitungan jumlah foreground pixel dari setiap metode.

## Metode yang Digunakan

### Global Thresholding

Global threshold digunakan untuk memisahkan foreground dan background menggunakan nilai ambang tertentu.

Pada program ini digunakan nilai threshold 127.

### Otsu Thresholding

Metode Otsu menentukan nilai threshold secara otomatis berdasarkan distribusi intensitas citra.

### Opening

Opening digunakan untuk mengurangi noise kecil pada hasil thresholding.

### Closing

Closing digunakan untuk membantu menghubungkan bagian foreground yang terputus dan mengisi celah kecil.

## Output

Output program berupa:

- Citra hasil crop
- Citra grayscale
- Hasil Global Thresholding
- Hasil Otsu Thresholding
- Hasil Opening
- Hasil Closing
- Jumlah foreground pixel
- Persentase foreground pixel
- File `hasil_segmentasi.csv`

## Dataset

Dataset yang digunakan terdiri dari beberapa citra dokumen ijazah dengan kondisi kualitas citra yang berbeda.

Contohnya:

- High Quality
- Low Contrast
- Blurred
- High Noise
- Low Resolution
- Faded / Underexposed
- Color Shift
- JPEG Compression
- Combined Degradation

## Author

Nama: Asiyah Zahidah Farhat

NIM: F1G124026

Mata Kuliah: Pengolahan Citra Digital

Universitas Halu Oleo
