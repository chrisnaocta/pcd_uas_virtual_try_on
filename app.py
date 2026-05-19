import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
# Catatan: pastikan semua library di atas sudah terinstall di environment Python Anda.

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# Mengatur tampilan aplikasi web: judul tab browser, layout lebar penuh,
# dan ikon emoji yang muncul di tab browser
# ==============================================================================
st.set_page_config(
    page_title="Virtual Try-On App — UAS PCD", layout="wide", page_icon="👕"
)

# Judul utama dan deskripsi singkat aplikasi yang tampil di bagian atas halaman
st.title("👕 Aplikasi Virtual Try-On Interaktif")
st.write("Implementasi Sistem Pengolahan Citra Digital untuk Virtual Try-On")
st.markdown("---")

# ==============================================================================
# KONSTANTA DIMENSI TARGET
# Semua gambar (baju maupun orang) akan di-resize ke ukuran standar 768×1024 px.
# Ini penting agar koordinat keypoint dari dataset VITON tetap valid dan
# konsisten dengan dimensi gambar yang diproses.
# ==============================================================================
TARGET_W, TARGET_H = 768, 1024


# ==============================================================================
# ████████████████████████████████████████████████████████████████████████████
# BAGIAN 1 — FUNGSI-FUNGSI INTI PENGOLAHAN CITRA DIGITAL
# Semua logika pemrosesan gambar dikumpulkan di sini agar terpisah dari UI,
# mengikuti prinsip pemisahan logika (separation of concerns).
# ████████████████████████████████████████████████████████████████████████████
# ==============================================================================


# ------------------------------------------------------------------------------
# FUNGSI: resize_to_target
# Tujuan : Menyamakan ukuran semua gambar input ke 768×1024 px.
# Alasan : Dataset VITON menggunakan resolusi ini sebagai standar,
#          dan koordinat keypoint JSON merujuk ke resolusi yang sama.
#          Tanpa resize, posisi baju bisa meleset dari tubuh orang.
# Input  : img_array — numpy array gambar (RGB)
# Output : numpy array gambar yang sudah di-resize ke TARGET_W × TARGET_H
# ------------------------------------------------------------------------------
def resize_to_target(img_array):
    img_pil = Image.fromarray(img_array)
    if img_pil.size != (TARGET_W, TARGET_H):
        # LANCZOS = algoritma resampling berkualitas tinggi,
        # meminimalkan aliasing/artefak saat mengubah ukuran gambar
        img_pil = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
    return np.array(img_pil)


# ------------------------------------------------------------------------------
# FUNGSI: remove_white_background  ←  TAHAP 1: PREPROCESSING BAJU
# Tujuan : Menghapus background putih/abu-abu dari foto baju secara otomatis
#          menggunakan deteksi warna netral, menghasilkan gambar RGBA
#          dengan background transparan (alpha channel = 0).
#
# Pendekatan:
#   Background dideteksi berdasarkan 2 kondisi SEKALIGUS:
#   1. Kecerahan tinggi  → pixel terang (grayscale > bright_thresh)
#   2. Warna netral      → R ≈ G ≈ B (selisih < gray_tolerance)
#   Baju warna terang (pink, krem, kuning) TIDAK terhapus karena
#   komponen warnanya tidak seimbang (mis. R >> G untuk warna merah muda).
#
# Parameter:
#   bright_thresh  : ambang batas kecerahan (default 200 dari skala 0-255)
#   gray_tolerance : toleransi selisih antar channel R,G,B (default 20)
# Input  : img_rgb — numpy array gambar baju (RGB, sudah di-resize)
# Output : cloth_rgba — gambar RGBA (4 channel, background transparan)
#          cloth_mask — mask biner (putih=baju, hitam=background)
# ------------------------------------------------------------------------------
def remove_white_background(img_rgb, bright_thresh=200, gray_tolerance=20):
    h, w = img_rgb.shape[:2]

    # Pisahkan channel R, G, B ke variabel terpisah untuk perbandingan
    r = img_rgb[:, :, 0].astype(int)
    g = img_rgb[:, :, 1].astype(int)
    b = img_rgb[:, :, 2].astype(int)

    # Kondisi 1: Konversi ke grayscale → cek apakah pixel cukup terang
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(int)
    is_bright = gray > bright_thresh

    # Kondisi 2: Cek apakah warna pixel netral/abu-abu
    # (semua selisih antar channel R-G, G-B, R-B di bawah toleransi)
    is_neutral = (
        (np.abs(r - g) < gray_tolerance)
        & (np.abs(g - b) < gray_tolerance)
        & (np.abs(r - b) < gray_tolerance)
    )

    # Background = pixel yang TERANG sekaligus NETRAL → tandai dengan 255
    bg_mask = (is_bright & is_neutral).astype(np.uint8) * 255
    # Inversi: area baju = putih (255), background = hitam (0)
    rough_mask = cv2.bitwise_not(bg_mask)

    # Morphological Closing: menutup lubang kecil dalam mask baju
    # (mis. logo putih di tengah baju tidak ikut dihapus)
    # Kernel elips ukuran 7×7 dipilih agar operasi lebih halus
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rough_mask = cv2.morphologyEx(
        rough_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    # Temukan kontur (outline) dari area baju yang terdeteksi
    contours, _ = cv2.findContours(
        rough_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # Jika tidak ada kontur ditemukan, gunakan rough_mask langsung
        cloth_mask = rough_mask
    else:
        # Ambil kontur TERBESAR = pasti area baju utama (bukan noise kecil)
        largest = max(contours, key=cv2.contourArea)
        # Buat mask bersih hanya dari kontur terbesar (isi penuh = solid)
        cloth_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cloth_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Gabungkan RGB baju + mask sebagai Alpha channel → format RGBA (4 channel)
    # Alpha 255 = opaque (baju), Alpha 0 = transparan (background)
    cloth_rgba = np.dstack([img_rgb, cloth_mask])
    return cloth_rgba, cloth_mask


# ------------------------------------------------------------------------------
# FUNGSI: parse_keypoints_from_json  ←  TAHAP 2: EKSTRAKSI KOORDINAT
# Tujuan : Membaca koordinat titik-titik penting tubuh dari file JSON OpenPose
#          yang disediakan dataset VITON (format BODY_25).
#
# Format JSON OpenPose:
#   pose_keypoints_2d = [x0,y0,conf0, x1,y1,conf1, x2,y2,conf2, ...]
#   Setiap titik terdiri dari 3 nilai: koordinat X, Y, dan confidence score.
#   Confidence < 0.1 berarti titik tidak terdeteksi dengan andal → diabaikan.
#
# Titik yang diambil (indeks BODY_25):
#   1=Neck, 2=RShoulder, 5=LShoulder, 8=MidHip, 9=RHip, 12=LHip
#   (catatan: "Left/Right" = perspektif SUBJEK, bukan penonton)
#
# Fallback:
#   Jika salah satu hip tidak terdeteksi, posisinya diestimasi dari
#   mid_hip dan hip sisi lain (simetri tubuh).
#
# Input  : json_data — dictionary hasil parsing file JSON
# Output : dict koordinat {nama_titik: (x, y)} atau None jika gagal
# ------------------------------------------------------------------------------
def parse_keypoints_from_json(json_data):
    # Mapping nama titik ke indeks array pose_keypoints_2d
    # Hanya titik yang relevan untuk area baju (torso) yang diambil
    KP_IDX = {
        # 'nose':           0,   # tidak dipakai
        'neck':           1,
        'right_shoulder': 2,    # perspektif SUBJEK (= kiri di gambar)
        # 'right_elbow':    3,   # tidak dipakai
        # 'right_wrist':    4,   # tidak dipakai
        'left_shoulder':  5,    # perspektif SUBJEK (= kanan di gambar)
        # 'left_elbow':     6,   # tidak dipakai
        # 'left_wrist':     7,   # tidak dipakai
        'mid_hip':        8,
        'right_hip':      9,
        # 'right_knee':    10,   # tidak dipakai
        # 'right_ankle':   11,   # tidak dipakai
        'left_hip':      12,
        # 'left_knee':     13,   # tidak dipakai
        # 'left_ankle':    14,   # tidak dipakai
    }

    people = json_data.get("people", [])
    if not people:
        return None  # JSON kosong atau tidak ada orang terdeteksi

    # Ambil array keypoint orang pertama (indeks 0)
    kps = people[0].get("pose_keypoints_2d", [])

    def get_pt(idx):
        # Setiap titik = 3 elemen: [x, y, confidence] → idx*3 s.d. idx*3+2
        x, y, conf = kps[idx * 3], kps[idx * 3 + 1], kps[idx * 3 + 2]
        # Hanya kembalikan koordinat jika confidence cukup tinggi (> 0.1)
        return (round(x), round(y)) if conf > 0.1 else None

    # Buat dictionary semua titik: {nama: (x,y) atau None}
    raw = {name: get_pt(idx) for name, idx in KP_IDX.items()}

    # --- Fallback Estimasi Hip ---
    # Jika salah satu atau kedua titik hip tidak terdeteksi,
    # estimasi posisinya menggunakan simetri dari mid_hip
    mh = raw["mid_hip"]
    ls = raw["left_shoulder"]
    rs = raw["right_shoulder"]

    if mh:
        # Perkiraan setengah lebar tubuh dari lebar bahu
        sh_half = abs(ls[0] - rs[0]) // 4 if ls and rs else 40

        if raw["left_hip"] is None and raw["right_hip"] is not None:
            # Estimasi left_hip sebagai cerminan right_hip terhadap mid_hip
            raw["left_hip"] = (
                2 * mh[0] - raw["right_hip"][0],
                raw["right_hip"][1],
            )
        elif raw["right_hip"] is None and raw["left_hip"] is not None:
            # Estimasi right_hip sebagai cerminan left_hip terhadap mid_hip
            raw["right_hip"] = (
                2 * mh[0] - raw["left_hip"][0],
                raw["left_hip"][1],
            )
        elif raw["left_hip"] is None and raw["right_hip"] is None:
            # Kedua hip tidak terdeteksi → estimasi dari mid_hip ± sh_half
            raw["left_hip"] = (mh[0] - sh_half, mh[1])
            raw["right_hip"] = (mh[0] + sh_half, mh[1])

    # Validasi: 4 titik wajib harus ada, jika tidak → kembalikan None
    required = ["neck", "left_shoulder", "right_shoulder", "mid_hip"]
    missing = [k for k in required if raw.get(k) is None]
    if missing:
        return None  # Titik penting tidak terdeteksi, tidak bisa dilanjutkan

    return raw


# ------------------------------------------------------------------------------
# FUNGSI: compute_pseudo_points  ←  KOREKSI KOORDINAT UNTUK WARPING
# Tujuan : Menghitung 4 titik kontrol baju yang lebih akurat dari raw keypoints.
#          Raw keypoints OpenPose tidak bisa langsung dipakai karena:
#
#   MASALAH 1 — Bahu terlalu ke dalam:
#     Titik shoulder OpenPose ada di sendi glenohumeral (dalam),
#     bukan di ujung bahu fisik → baju akan terlihat terlalu sempit.
#     SOLUSI: Extend titik bahu ke luar sebesar (shoulder_width × extend_ratio).
#
#   MASALAH 2 — Hip terlalu ke bawah:
#     Titik hip OpenPose ada di panggul bawah, bukan pinggang baju.
#     Jika dipakai langsung → baju terlalu panjang dan melar ke bawah.
#     SOLUSI: Buat "pseudo-waist" = 58-80% jarak neck→mid_hip (lebih atas).
#
#   MASALAH 3 — Label kiri/kanan terbalik perspektif:
#     "left_shoulder" di OpenPose = kiri subjek = KANAN di gambar.
#     SOLUSI: Sort berdasarkan koordinat X di gambar, bukan label subjek.
#
# Parameter:
#   extend_ratio  : seberapa jauh bahu di-extend ke luar (0.35 = 35% lebar bahu)
#   waist_factor  : posisi pinggang baju = neck + (torso_height × waist_factor)
#   shoulder_lift : seberapa tinggi bahu di-naikkan (0.2 = 20% tinggi torso)
#
# Output: dict 4 titik siap warp:
#   pseudo_left_shoulder, pseudo_right_shoulder,
#   pseudo_left_waist, pseudo_right_waist
# ------------------------------------------------------------------------------
def compute_pseudo_points(raw_pts, extend_ratio, waist_factor, shoulder_lift):
    neck = raw_pts["neck"]
    ls = raw_pts["left_shoulder"]
    rs = raw_pts["right_shoulder"]
    mh = raw_pts["mid_hip"]

    # KOREKSI 1: Sort shoulder berdasarkan X gambar (bukan label subjek)
    # img_left_sh  = titik yang berada di KIRI gambar (X lebih kecil)
    # img_right_sh = titik yang berada di KANAN gambar (X lebih besar)
    if ls[0] < rs[0]:
        img_left_sh, img_right_sh = ls, rs
    else:
        img_left_sh, img_right_sh = rs, ls

    # Hitung lebar bahu dan tinggi torso sebagai referensi ukuran
    shoulder_width = abs(img_right_sh[0] - img_left_sh[0])
    torso_height = mh[1] - neck[1]

    # KOREKSI 2: Extend bahu ke luar + naikkan ke atas
    extend = shoulder_width * extend_ratio          # jarak extend horizontal
    lift = int(torso_height * shoulder_lift)        # jarak naik vertikal

    # Titik bahu kiri lebih ke kiri (X dikurangi) dan lebih ke atas (Y dikurangi)
    pseudo_left_sh = (int(img_left_sh[0] - extend), int(img_left_sh[1] - lift))
    # Titik bahu kanan lebih ke kanan (X ditambah) dan lebih ke atas (Y dikurangi)
    pseudo_right_sh = (
        int(img_right_sh[0] + extend),
        int(img_right_sh[1] - lift),
    )

    # KOREKSI 3: Pseudo-waist — posisi Y pinggang baju (bukan hip asli)
    # waist_y = neck.y + torso_height × waist_factor
    # Contoh: waist_factor=0.8 → 80% dari leher ke panggul = setinggi pinggang baju
    waist_y = int(neck[1] + torso_height * waist_factor)

    # 4 titik final: X pinggang sama dengan X bahu (trapesoid tegak)
    return {
        "pseudo_left_shoulder": pseudo_left_sh,
        "pseudo_right_shoulder": pseudo_right_sh,
        "pseudo_left_waist": (pseudo_left_sh[0], waist_y),    # X sama dengan bahu kiri
        "pseudo_right_waist": (pseudo_right_sh[0], waist_y),  # X sama dengan bahu kanan
    }


# ------------------------------------------------------------------------------
# FUNGSI: get_cloth_bbox
# Tujuan : Menemukan bounding box (kotak pembatas) area baju dari alpha channel.
#          Diperlukan karena gambar baju berukuran 768×1024 tetapi area bajunya
#          hanya ada di tengah — banyak ruang kosong transparan di sekeliling.
#          Jika kita pakai sudut gambar sebagai src_pts, area kosong itu
#          ikut di-warp dan hasilnya meleset.
#
# Cara kerja:
#   Threshold alpha → binary mask → findNonZero → boundingRect
#   boundingRect menghasilkan (x, y, w, h) → dikonversi ke (x_min, y_min, x_max, y_max)
#
# Input  : alpha — channel alpha (grayscale, shape H×W)
# Output : (x_min, y_min, x_max, y_max) — koordinat pojok bounding box baju
# ------------------------------------------------------------------------------
def get_cloth_bbox(alpha):
    # Threshold: pixel alpha > 10 dianggap bagian baju (bukan transparan)
    _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    # Temukan semua koordinat pixel yang tidak nol (= area baju)
    coords = cv2.findNonZero(binary)
    if coords is None:
        # Fallback: jika mask kosong, kembalikan seluruh ukuran gambar
        return 0, 0, TARGET_W, TARGET_H
    # boundingRect menghitung kotak terkecil yang mencakup semua koordinat baju
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x + w, y + h  # konversi ke format (x_min, y_min, x_max, y_max)


# ------------------------------------------------------------------------------
# FUNGSI: clean_cloth_edges  ←  FIX PERMANEN GARIS PUTIH DI TEPI BAJU
# Tujuan : Menghilangkan garis/noise putih yang muncul di tepi baju saat di-warp.
#
# AKAR MASALAH:
#   warpPerspective menggunakan interpolasi bilinear:
#   pixel_baru = α × warna_baju + (1-α) × warna_tetangga
#   Jika area transparan di sekitar baju masih menyimpan RGB=(255,255,255),
#   maka pixel tepi hasil warp = campuran warna_baju + putih = GARIS PUTIH.
#
# SOLUSI (2 langkah):
#   LANGKAH 1 — Flood warna inti ke tepi:
#     Erode mask untuk dapat area INTI baju yang pasti bersih warnanya.
#     Dilate warna inti ke seluruh area (tepi + transparan).
#     Ini mengganti warna putih di tepi dengan warna baju terdekat.
#
#   LANGKAH 2 — Gabungkan:
#     Pixel inti baju → tetap pakai warna asli (tidak diubah)
#     Pixel tepi/transparan → pakai warna hasil flood (bukan putih lagi)
#     Alpha channel tidak diubah sama sekali di fungsi ini.
#
# Parameter:
#   erode_px  : radius erode untuk dapat mask inti (default 6px)
#   flood_iter: jumlah iterasi flood (default 25 = 25 layer pixel ke luar)
# ------------------------------------------------------------------------------
def clean_cloth_edges(cloth_rgba, erode_px=6, flood_iter=25):
    # Pisahkan RGB dan alpha dari gambar baju RGBA
    rgb = cloth_rgba[:, :, :3].astype(np.float32)
    alpha = cloth_rgba[:, :, 3].copy()

    # Buat hard mask binary dari alpha channel
    _, hard = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

    # Erode mask untuk dapat area INTI baju (jauh dari tepi yang kotor)
    # Kernel elips ukuran (erode_px*2+1) × (erode_px*2+1)
    k_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
    )
    interior = cv2.erode(hard, k_erode, iterations=1)

    # Flood: mulai dari warna inti, dilate keluar sebanyak flood_iter kali
    interior_color = rgb.copy()
    k_flood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for _ in range(flood_iter):
        # Dilate warna ke semua arah
        dilated_color = np.stack(
            [
                cv2.dilate(interior_color[:, :, c], k_flood)
                for c in range(3)   # lakukan untuk channel R, G, B
            ],
            axis=-1,
        )
        # Hanya update pixel NON-INTI (tepi + transparan)
        # Pixel inti tidak boleh diubah agar warna asli baju terjaga
        not_interior = (interior == 0)[:, :, np.newaxis]
        interior_color = np.where(not_interior, dilated_color, interior_color)

    # Gabungkan: inti=warna asli, tepi/transparan=warna flood
    is_interior_3ch = np.stack([interior == 255] * 3, axis=-1)
    rgb_clean = np.where(is_interior_3ch, rgb, interior_color)

    # Buat gambar hasil: RGB yang sudah dibersihkan + alpha ASLI (tidak diubah)
    result = cloth_rgba.copy().astype(np.float32)
    result[:, :, :3] = np.clip(rgb_clean, 0, 255)
    result[:, :, 3] = alpha   # alpha tidak dimodifikasi
    return result.astype(np.uint8)


# ------------------------------------------------------------------------------
# FUNGSI: warp_cloth_to_body  ←  TAHAP 3: PERSPECTIVE TRANSFORM
# Tujuan : Mentransformasi gambar baju agar menyesuaikan bentuk dan posisi
#          torso orang menggunakan Perspective Transform (homografi).
#
# KONSEP PERSPEKTIF TRANSFORM:
#   Diberikan 4 titik sumber (src) dan 4 titik tujuan (dst),
#   cv2.getPerspectiveTransform menghitung matrix 3×3 (Homography Matrix M)
#   yang memetakan setiap pixel dari src ke dst.
#   cv2.warpPerspective menerapkan matrix M ke seluruh gambar.
#
# MAPPING:
#   src (sudut bounding box baju)  →  dst (pseudo-titik tubuh)
#   ┌────────────────────────────────────────────────────────┐
#   │ bbox kiri atas   (x_min, y_min) → pseudo_left_shoulder  │
#   │ bbox kanan atas  (x_max, y_min) → pseudo_right_shoulder │
#   │ bbox kanan bawah (x_max, y_max) → pseudo_right_waist    │
#   │ bbox kiri bawah  (x_min, y_max) → pseudo_left_waist     │
#   └────────────────────────────────────────────────────────┘
#
# PIPELINE FUNGSI INI:
#   Step 0: clean_cloth_edges → cegah garis putih
#   Step 1: Hard threshold + erode alpha → potong tepi kotor 4px
#   Step 2: Hitung bounding box dari mask yang sudah bersih
#   Step 3: Warp semua 4 channel (R,G,B,A) dengan matrix yang sama
#   Step 4: Hard threshold alpha setelah warp → buang sisa semi-transparan
# ------------------------------------------------------------------------------
def warp_cloth_to_body(cloth_rgba, pseudo_pts):

    # STEP 0: Bersihkan warna tepi dengan flood fill → cegah garis putih
    cloth_rgba = clean_cloth_edges(cloth_rgba, erode_px=6, flood_iter=25)

    # STEP 1: Buat mask binary yang ketat + erode 2 iterasi (potong ~4px tepi)
    # Pixel di tepi luar yang mungkin masih kotor dibuang sebelum warp
    alpha = cloth_rgba[:, :, 3]
    _, hard = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    k_trim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    trimmed_mask = cv2.erode(hard, k_trim, iterations=2)  # potong ±4px dari tepi

    # Terapkan mask yang sudah di-trim ke gambar baju
    cloth_clean = cloth_rgba.copy()
    cloth_clean[:, :, 3] = trimmed_mask

    # STEP 2: Hitung bounding box dari mask yang sudah bersih
    # (bukan dari sudut gambar 768×1024 yang penuh area kosong)
    x_min, y_min, x_max, y_max = get_cloth_bbox(trimmed_mask)

    # 4 titik sumber = sudut-sudut bounding box baju (di gambar baju)
    src_pts = np.float32(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    )
    # 4 titik tujuan = pseudo-titik tubuh (di gambar orang)
    dst_pts = np.float32(
        [
            list(pseudo_pts["pseudo_left_shoulder"]),
            list(pseudo_pts["pseudo_right_shoulder"]),
            list(pseudo_pts["pseudo_right_waist"]),
            list(pseudo_pts["pseudo_left_waist"]),
        ]
    )

    # Hitung Homography Matrix M dari 4 pasang titik (src → dst)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # STEP 3: Warp setiap channel secara terpisah menggunakan matrix M yang sama
    # INTER_LINEAR = interpolasi bilinear (kualitas baik, kecepatan sedang)
    # BORDER_CONSTANT + value=0 = area di luar baju diisi transparan (alpha=0)
    warped_channels = [
        cv2.warpPerspective(
            cloth_clean[:, :, c],   # channel ke-c (0=R, 1=G, 2=B, 3=A)
            M,
            (TARGET_W, TARGET_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,          # isi luar dengan 0 (hitam/transparan)
        )
        for c in range(4)   # loop untuk R, G, B, A
    ]
    # Gabungkan kembali 4 channel menjadi satu gambar RGBA
    warped_rgba = np.stack(warped_channels, axis=-1)

    # STEP 4: Hard threshold alpha SETELAH warp
    # Interpolasi bilinear menghasilkan nilai alpha antara 0-255 di tepi.
    # Kita set threshold 180: >= 180 → 255 (baju), < 180 → 0 (transparan)
    # Ini menghilangkan pixel semi-transparan yang bisa menyebabkan tepi kabur
    _, final_alpha = cv2.threshold(
        warped_rgba[:, :, 3], 180, 255, cv2.THRESH_BINARY
    )
    warped_rgba = warped_rgba.copy()
    warped_rgba[:, :, 3] = final_alpha

    return warped_rgba, src_pts, dst_pts


# ------------------------------------------------------------------------------
# FUNGSI: composite_cloth_on_person  ←  TAHAP 4: BLENDING & SMOOTHING
# Tujuan : Menempelkan baju yang sudah di-warp ke foto orang dengan tepi
#          yang halus dan natural menggunakan soft alpha blending.
#
# MENGAPA TIDAK PAKAI seamlessClone?
#   seamlessClone (cv2.MIXED_CLONE) dirancang untuk menyatukan tekstur alami
#   (kulit, tanaman, awan). Untuk baju, mode ini justru "mencampur" warna baju
#   dengan latar sehingga baju terlihat transparan. Solusi: pure alpha blend.
#
# TEKNIK SOFT EDGE (menghilangkan tepi kaku):
#   Hard mask   → erode (zona inti penuh) + dilate (zona tepi)
#   Edge zone   = dilate - erode  → hanya area tepian saja
#   GaussianBlur(edge_zone) → gradient halus di area tepi
#   soft_alpha  = inti (1.0 solid) + tepi (gradient 0→1)
#
# RUMUS ALPHA BLENDING:
#   pixel_hasil = α × warna_baju + (1-α) × warna_orang
#   di mana α = soft_alpha (0.0 = transparan penuh, 1.0 = opaque penuh)
# ------------------------------------------------------------------------------
def composite_cloth_on_person(person_rgb, warped_rgba):
    # Pisahkan RGB dan alpha dari warped cloth
    warped_rgb = warped_rgba[:, :, :3].astype(np.float32)
    # Normalisasi alpha dari [0,255] ke [0.0,1.0] untuk operasi float
    warped_alpha = warped_rgba[:, :, 3].astype(np.float32) / 255.0

    # Buat hard mask binary: pixel baju (alpha > 0.5) = 255, lainnya = 0
    hard_alpha = (warped_alpha > 0.5).astype(np.uint8) * 255

    # Erode hard mask → zona INTI baju (pasti solid, jauh dari tepi)
    kernel_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # Dilate hard mask sedikit → perluasan zona tepi
    kernel_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(hard_alpha, kernel_in, iterations=2)    # zona inti
    dilated = cv2.dilate(hard_alpha, kernel_out, iterations=1)  # zona diperluas

    # Edge zone = selisih antara dilated dan eroded = hanya area tepian
    edge_zone = cv2.subtract(dilated, eroded)
    # Gaussian Blur pada edge zone → gradient halus (tidak patah tiba-tiba)
    edge_blur = cv2.GaussianBlur(edge_zone, (15, 15), 0)

    # Gabungkan: inti=1.0 solid, tepi=gradient blur
    # np.clip memastikan nilai tidak melebihi [0.0, 1.0]
    soft_alpha = np.clip(
        eroded.astype(np.float32) / 255.0 + edge_blur.astype(np.float32) / 255.0,
        0.0,
        1.0,
    )

    # Expand alpha 2D (H,W) → 3D (H,W,3) untuk operasi per-channel
    alpha_3ch = np.stack([soft_alpha] * 3, axis=-1)
    person_float = person_rgb.astype(np.float32)

    # ALPHA BLENDING: rumus Porter-Duff "over"
    # pixel_hasil = α × baju + (1-α) × orang
    result = alpha_3ch * warped_rgb + (1.0 - alpha_3ch) * person_float
    # Clip ke [0,255] dan konversi kembali ke uint8 untuk output gambar
    return np.clip(result, 0, 255).astype(np.uint8)


# ==============================================================================
# ████████████████████████████████████████████████████████████████████████████
# BAGIAN 2 — ANTARMUKA PENGGUNA STREAMLIT (UI)
# Semua komponen UI (upload, slider, tampilan gambar) ada di sini.
# Streamlit bekerja dengan cara: setiap interaksi pengguna (klik, drag slider)
# menjalankan ulang seluruh script dari atas → session_state menyimpan data
# antar run agar tidak hilang.
# ████████████████████████████████████████████████████████████████████████████
# ==============================================================================

# ------------------------------------------------------------------------------
# SESSION STATE — Penyimpanan Data Antar Interaksi
# Streamlit me-reload script setiap ada interaksi pengguna.
# session_state berfungsi seperti "memori" yang tetap ada selama sesi browser.
# Ketiga variabel ini menyimpan data yang dibutuhkan lintas tab.
# ------------------------------------------------------------------------------
if "cloth_rgba" not in st.session_state:
    st.session_state.cloth_rgba = None      # hasil preprocessing baju (RGBA)
if "raw_keypoints" not in st.session_state:
    st.session_state.raw_keypoints = None   # koordinat keypoint dari JSON
if "person_rgb" not in st.session_state:
    st.session_state.person_rgb = None      # gambar orang (RGB)

# ------------------------------------------------------------------------------
# LAYOUT TABS — 3 Tab Sesuai Pipeline 3 Tahap
# Pengguna mengisi tab secara berurutan: Tahap 1 → Tahap 2 → Tahap 3
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📁 TAHAP 1: Preprocess Baju", "👤 TAHAP 2: Keypoints", "✨ TAHAP 3: Virtual Try-On"]
)

# ==============================================================================
# TAB 1 — PREPROCESSING BAJU
# Pengguna upload foto baju → sistem hapus background → tampilkan hasil 3 panel
# ==============================================================================
with tab1:
    st.header("Langkah 1 — Background Removal Baju")

    # Widget upload file: menerima format jpg, jpeg, png
    uploaded_cloth = st.file_uploader(
        "Upload Foto Baju (Format .jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_cloth is not None:
        # Baca file upload sebagai bytes → decode ke numpy array BGR (format OpenCV)
        file_bytes = np.asarray(bytearray(uploaded_cloth.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Konversi BGR (OpenCV default) → RGB (format yang benar untuk tampilan)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize gambar ke ukuran target standar 768×1024
        img_resized = resize_to_target(img_rgb)

        # Jalankan fungsi background removal → hasilkan RGBA + mask
        cloth_rgba, cloth_mask = remove_white_background(img_resized)

        # Simpan hasil ke session_state agar bisa diakses di Tab 3
        st.session_state.cloth_rgba = cloth_rgba

        st.success("✅ Preprocessing Baju Selesai!")

        # Tampilkan 3 panel berdampingan: original, mask, hasil RGBA
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_resized, caption="① Original Cloth", use_container_width=True)
        with col2:
            # cloth_mask adalah grayscale 2D → Streamlit otomatis render sebagai grayscale
            st.image(
                cloth_mask,
                caption="② Cloth Mask (Grayscale)",
                use_container_width=True,
            )
        with col3:
            # cloth_rgba adalah RGBA 4-channel → Streamlit render dengan transparansi
            st.image(
                cloth_rgba,
                caption="③ Cloth RGBA (Transparan)",
                use_container_width=True,
            )

        # Info statistik gambar: dimensi, jumlah pixel baju vs background
        st.info(
            f"**Shape Output**: {cloth_rgba.shape} | **Pixel Baju**: {np.sum(cloth_mask > 0):,} px | **Pixel BG**: {np.sum(cloth_mask == 0):,} px"
        )
    else:
        st.warning("Silakan unggah foto baju terlebih dahulu.")

# ==============================================================================
# TAB 2 — EKSTRAKSI KEYPOINTS TUBUH
# Pengguna upload foto orang + JSON OpenPose →
# sistem parsing koordinat → visualisasi skeleton + titik penting
# ==============================================================================
with tab2:
    st.header("Langkah 2 — Deteksi Pose Manusia")

    # Dua kolom upload: kiri untuk foto, kanan untuk JSON
    col_u1, col_u2 = st.columns(2)

    with col_u1:
        uploaded_person = st.file_uploader(
            "Upload Foto Orang", type=["jpg", "jpeg", "png"]
        )
    with col_u2:
        uploaded_json = st.file_uploader(
            "Upload OpenPose JSON (.json)", type=["json"]
        )

    # Proses hanya jika KEDUA file sudah di-upload
    if uploaded_person is not None and uploaded_json is not None:
        # Load dan proses foto orang
        p_bytes   = np.asarray(bytearray(uploaded_person.read()), dtype=np.uint8)
        p_img_bgr = cv2.imdecode(p_bytes, cv2.IMREAD_COLOR)
        p_img_rgb = cv2.cvtColor(p_img_bgr, cv2.COLOR_BGR2RGB)
        # Resize ke target dan simpan ke session_state untuk Tab 3
        st.session_state.person_rgb = resize_to_target(p_img_rgb)

        # Parse file JSON OpenPose → ekstrak koordinat titik tubuh
        json_data = json.load(uploaded_json)
        raw_pts   = parse_keypoints_from_json(json_data)

        if raw_pts is None:
            # Keypoint wajib tidak terdeteksi → tidak bisa lanjut
            st.error("❌ Gagal memproses koordinat OpenPose. Pastikan Neck, Shoulders, Mid-Hip terdeteksi.")
        else:
            # Simpan keypoint ke session_state untuk dipakai di Tab 3
            st.session_state.raw_keypoints = raw_pts
            st.success("✅ File Pose JSON Berhasil Dimuat!")

            person_rgb = st.session_state.person_rgb
            h, w       = person_rgb.shape[:2]

            # --- Bangun visualisasi Panel ②: Skeleton OpenPose ---
            # Definisi koneksi antar titik untuk menggambar tulang/skeleton
            SKELETON_CONNECTIONS = [
                ("neck", "right_shoulder"), ("neck", "left_shoulder"),
                ("right_shoulder", "right_hip"), ("left_shoulder", "left_hip"),
                ("right_hip", "left_hip"), ("neck", "mid_hip"),
            ]
            # Canvas hitam sebagai background skeleton
            skeleton_canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # Gambar garis skeleton (hijau) antar titik yang terdeteksi
            for kp_a, kp_b in SKELETON_CONNECTIONS:
                pt_a, pt_b = raw_pts.get(kp_a), raw_pts.get(kp_b)
                if pt_a and pt_b:
                    cv2.line(skeleton_canvas, pt_a, pt_b, (100, 200, 100), 4)

            # Gambar lingkaran kuning di setiap titik keypoint
            for name, pt in raw_pts.items():
                if pt:
                    cv2.circle(skeleton_canvas, pt, 8, (255, 200, 0), -1)

            # --- Bangun visualisasi Panel ③: Foto orang + keypoint overlay ---
            overlay = person_rgb.copy()

            # Semua titik menggunakan warna hitam (dapat diubah per titik jika mau)
            kp_colors = {
                "neck":           (0,0,0),
                "left_shoulder":  (0,0,0),
                "right_shoulder": (0,0,0),
                "mid_hip":        (0,0,0),
                "left_hip":       (0,0,0),
                "right_hip":      (0,0,0),
            }

            # Gambar setiap titik keypoint beserta labelnya di foto orang
            for name, pt in raw_pts.items():
                if pt:
                    color = kp_colors.get(name, (255, 0, 0))
                    # Lingkaran solid berwarna + border putih agar terlihat di semua bg
                    cv2.circle(overlay, pt, 10, color, -1)
                    cv2.circle(overlay, pt, 10, (255, 255, 255), 2)

                    # Format label: ganti underscore dengan spasi
                    label = name.replace("_", " ")
                    font       = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness  = 1

                    # Hitung ukuran teks untuk posisi teks yang tepat (di atas titik)
                    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    text_x = pt[0] - text_w // 2        # tengahkan horizontal
                    text_y = pt[1] - 10 - 6             # posisi di atas lingkaran

                    # Gambar teks dua kali: shadow hitam tebal dulu, lalu teks merah
                    # Teknik ini membuat teks terbaca di background apapun
                    cv2.putText(overlay, label,
                                (text_x, text_y),
                                font, font_scale, (0, 0, 0), thickness + 1)   # shadow
                    cv2.putText(overlay, label,
                                (text_x, text_y),
                                font, font_scale, (255, 0, 0), thickness)      # teks merah

            # Gambar garis penghubung antar titik di foto orang (abu-abu tipis)
            connections = [
                ("neck", "left_shoulder"),
                ("neck", "right_shoulder"),
                ("left_shoulder", "left_hip"),
                ("right_shoulder", "right_hip"),
                ("left_hip", "right_hip"),
                ("neck", "mid_hip"),
            ]
            for kp_a, kp_b in connections:
                pt_a, pt_b = raw_pts.get(kp_a), raw_pts.get(kp_b)
                if pt_a and pt_b:
                    cv2.line(overlay, pt_a, pt_b, (180, 180, 180), 2)

            # Tampilkan 3 panel: original, skeleton, overlay keypoint
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.image(person_rgb,
                         caption="① Original Human",
                         use_container_width=True)
            with col_p2:
                st.image(skeleton_canvas,
                         caption="② OpenPose Skeleton",
                         use_container_width=True)
            with col_p3:
                st.image(overlay,
                         caption="③ Raw Keypoints dari JSON",
                         use_container_width=True)

            # Expander: tampilkan koordinat mentah dalam format JSON
            with st.expander("Lihat Koordinat Keypoints Mentah"):
                st.json(raw_pts)
    else:
        st.warning("Silakan unggah Foto Orang dan File OpenPose JSON.")

# ==============================================================================
# TAB 3 — VIRTUAL TRY-ON INTERAKTIF
# Menggabungkan hasil Tab 1 (baju RGBA) dan Tab 2 (keypoint orang)
# dengan parameter yang dapat disesuaikan real-time via slider.
# Setiap perubahan slider langsung men-trigger ulang komputasi warp + composite.
# ==============================================================================
with tab3:
    st.header("Langkah 3 — Interaktif Kontrol Fit & Warping")

    # Validasi: data dari Tab 1 dan Tab 2 harus sudah ada
    if (
        st.session_state.cloth_rgba is None
        or st.session_state.raw_keypoints is None
    ):
        # Tampilkan pesan error jika salah satu data belum tersedia
        st.error(
            "⚠️ Data Belum Lengkap! Pastikan Anda telah mengunggah berkas baju di Tahap 1 serta foto orang & JSON di Tahap 2."
        )
    else:
        # ── SLIDER PARAMETER ─────────────────────────────────────────────────
        # 3 slider untuk tuning posisi dan ukuran baju secara real-time.
        # Setiap perubahan nilai slider langsung menjalankan ulang warp + composite.
        st.subheader("🎛️ Parameter Kontrol Posisi Baju")
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            # extend_ratio: mengontrol seberapa jauh bahu di-extend ke samping
            # 0.1 = sempit, 0.6 = sangat lebar
            ext_ratio = st.slider(
                "Extend Ratio (Lebar Bahu)",
                min_value=0.1,
                max_value=0.6,
                value=0.35,
                step=0.01,
            )
        with col_s2:
            # waist_factor: mengontrol panjang baju ke bawah
            # 0.4 = sangat pendek (crop top), 0.9 = sangat panjang
            waist_fac = st.slider(
                "Waist Factor (Panjang Kebawah)",
                min_value=0.4,
                max_value=0.9,
                value=0.80,
                step=0.01,
            )
        with col_s3:
            # shoulder_lift: mengontrol seberapa tinggi baju diangkat
            # 0.0 = posisi asli, 0.4 = sangat tinggi ke atas
            sh_lift = st.slider(
                "Shoulder Lift (Geser Atas)",
                min_value=0.0,
                max_value=0.4,
                value=0.20,
                step=0.01,
            )

        # ── PIPELINE UTAMA ────────────────────────────────────────────────────
        # Jalankan seluruh pipeline dengan nilai slider terbaru.
        # Urutan: pseudo_pts → warp → composite (sesuai pipeline 4 tahap)

        # Hitung 4 pseudo-titik dari keypoint mentah + parameter slider
        pseudo_pts = compute_pseudo_points(
            st.session_state.raw_keypoints, ext_ratio, waist_fac, sh_lift
        )
        # Warp baju ke posisi tubuh menggunakan Perspective Transform
        warped_rgba, src_pts, dst_pts = warp_cloth_to_body(
            st.session_state.cloth_rgba, pseudo_pts
        )
        # Composite baju ke foto orang dengan soft alpha blending
        result_rgb = composite_cloth_on_person(
            st.session_state.person_rgb, warped_rgba
        )

        # ── PERSIAPAN VISUALISASI ─────────────────────────────────────────────

        # Panel 1: Gambar baju + kotak kuning menunjukkan area src warp
        cloth_display = st.session_state.cloth_rgba[:, :, :3].copy()
        cv2.polylines(
            cloth_display,
            [src_pts.astype(np.int32).reshape(-1, 1, 2)],
            True,           # isClosed = kotak tertutup
            (255, 200, 0),  # warna kuning
            4,              # ketebalan garis
        )

        # Panel 2: Foto orang + overlay titik dst warp dan kotak trapesoid
        person_overlay = st.session_state.person_rgb.copy()

        # Urutan dst_pts harus konsisten dengan urutan src_pts di warp_cloth_to_body
        dst_order = [
            "pseudo_left_shoulder",
            "pseudo_right_shoulder",
            "pseudo_right_waist",
            "pseudo_left_waist",
        ]
        # Gambar kotak kuning yang menunjukkan area tujuan warp di tubuh orang
        dst_box = np.array([pseudo_pts[k] for k in dst_order], dtype=np.int32)
        cv2.polylines(
            person_overlay, [dst_box.reshape(-1, 1, 2)], True, (255, 255, 0), 3
        )

        # Gambar titik penanda: merah untuk bahu, biru untuk pinggang
        for k in dst_order:
            pt = pseudo_pts[k]
            col = (255, 80, 80) if "shoulder" in k else (80, 80, 255)
            cv2.circle(person_overlay, pt, 12, col, -1)            # lingkaran berwarna
            cv2.circle(person_overlay, pt, 12, (255, 255, 255), 2) # border putih

        # ── RENDER 3 PANEL DENGAN MATPLOTLIB ─────────────────────────────────
        # Matplotlib digunakan (bukan st.image langsung) agar layout,
        # judul, dan ukuran panel konsisten dengan output di Google Colab
        fig, axes = plt.subplots(1, 3, figsize=(18, 7.5))
        fig.suptitle(
            "TAHAP 3 — Virtual Try-On Interaktif Hasil Simulasi",
            fontsize=14,
            fontweight="bold",
        )

        # Panel ① — Baju dengan kotak area warp sumber
        axes[0].imshow(cloth_display)
        axes[0].set_title(
            "① Cloth RGBA\n(kotak kuning = area warp src)", fontsize=11
        )
        axes[0].axis("off")  # sembunyikan axis/grid

        # Panel ② — Foto orang dengan titik-titik tujuan warp
        axes[1].imshow(person_overlay)
        axes[1].set_title(
            "② Foto Orang + Dst Points\n(merah=bahu, biru=waist)", fontsize=11
        )
        axes[1].axis("off")

        # Panel ③ — Hasil akhir Virtual Try-On
        axes[2].imshow(result_rgb)
        axes[2].set_title("③ HASIL Virtual Try-On ✅", fontsize=11)
        axes[2].axis("off")

        plt.tight_layout()  # atur jarak antar panel agar tidak saling tumpang tindih

        # Render figure matplotlib ke dalam halaman Streamlit
        st.pyplot(fig)