import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# Setup Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Virtual Try-On App — UAS PCD", layout="wide", page_icon="👕"
)

st.title("👕 Aplikasi Virtual Try-On Interaktif")
st.write("Implementasi Sistem Pengolahan Citra Digital untuk Virtual Try-On")
st.markdown("---")

# Konstanta Dimensi Target
TARGET_W, TARGET_H = 768, 1024


# ==============================================================================
# LOGIKA INTI PENGOLAHAN CITRA DIGITAL
# ==============================================================================

# Fungsi di tahap 1
def resize_to_target(img_array):
    img_pil = Image.fromarray(img_array)
    if img_pil.size != (TARGET_W, TARGET_H):
        img_pil = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
    return np.array(img_pil)


def remove_white_background(img_rgb, bright_thresh=200, gray_tolerance=20):
    h, w = img_rgb.shape[:2]
    r = img_rgb[:, :, 0].astype(int)
    g = img_rgb[:, :, 1].astype(int)
    b = img_rgb[:, :, 2].astype(int)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(int)
    is_bright = gray > bright_thresh

    is_neutral = (
        (np.abs(r - g) < gray_tolerance)
        & (np.abs(g - b) < gray_tolerance)
        & (np.abs(r - b) < gray_tolerance)
    )

    bg_mask = (is_bright & is_neutral).astype(np.uint8) * 255
    rough_mask = cv2.bitwise_not(bg_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rough_mask = cv2.morphologyEx(
        rough_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    contours, _ = cv2.findContours(
        rough_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        cloth_mask = rough_mask
    else:
        largest = max(contours, key=cv2.contourArea)
        cloth_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cloth_mask, [largest], -1, 255, thickness=cv2.FILLED)

    cloth_rgba = np.dstack([img_rgb, cloth_mask])
    return cloth_rgba, cloth_mask


def parse_keypoints_from_json(json_data):
    # Indeks keypoint OpenPose BODY_25
    KP_IDX = {
        # 'nose':           0,
        'neck':           1,
        'right_shoulder': 2,    # perspektif SUBJEK (= kiri di gambar)
        # 'right_elbow':    3,
        # 'right_wrist':    4,
        'left_shoulder':  5,    # perspektif SUBJEK (= kanan di gambar)
        # 'left_elbow':     6,
        # 'left_wrist':     7,
        'mid_hip':        8,
        'right_hip':      9,
        # 'right_knee':    10,
        # 'right_ankle':   11,
        'left_hip':      12,
        # 'left_knee':     13,
        # 'left_ankle':    14,
    }

    people = json_data.get("people", [])
    if not people:
        return None

    kps = people[0].get("pose_keypoints_2d", [])

    def get_pt(idx):
        x, y, conf = kps[idx * 3], kps[idx * 3 + 1], kps[idx * 3 + 2]
        return (int(x), int(y)) if conf > 0.1 else None

    raw = {name: get_pt(idx) for name, idx in KP_IDX.items()}

    mh = raw["mid_hip"]
    ls = raw["left_shoulder"]
    rs = raw["right_shoulder"]

    if mh:
        sh_half = abs(ls[0] - rs[0]) // 4 if ls and rs else 40
        if raw["left_hip"] is None and raw["right_hip"] is not None:
            raw["left_hip"] = (
                2 * mh[0] - raw["right_hip"][0],
                raw["right_hip"][1],
            )
        elif raw["right_hip"] is None and raw["left_hip"] is not None:
            raw["right_hip"] = (
                2 * mh[0] - raw["left_hip"][0],
                raw["left_hip"][1],
            )
        elif raw["left_hip"] is None and raw["right_hip"] is None:
            raw["left_hip"] = (mh[0] - sh_half, mh[1])
            raw["right_hip"] = (mh[0] + sh_half, mh[1])

    required = ["neck", "left_shoulder", "right_shoulder", "mid_hip"]
    missing = [k for k in required if raw.get(k) is None]
    if missing:
        return None

    return raw


def compute_pseudo_points(raw_pts, extend_ratio, waist_factor, shoulder_lift):
    neck = raw_pts["neck"]
    ls = raw_pts["left_shoulder"]
    rs = raw_pts["right_shoulder"]
    mh = raw_pts["mid_hip"]

    if ls[0] < rs[0]:
        img_left_sh, img_right_sh = ls, rs
    else:
        img_left_sh, img_right_sh = rs, ls

    shoulder_width = abs(img_right_sh[0] - img_left_sh[0])
    torso_height = mh[1] - neck[1]
    extend = shoulder_width * extend_ratio
    lift = int(torso_height * shoulder_lift)
    waist_y = int(neck[1] + torso_height * waist_factor)

    pseudo_left_sh = (int(img_left_sh[0] - extend), int(img_left_sh[1] - lift))
    pseudo_right_sh = (
        int(img_right_sh[0] + extend),
        int(img_right_sh[1] - lift),
    )

    return {
        "pseudo_left_shoulder": pseudo_left_sh,
        "pseudo_right_shoulder": pseudo_right_sh,
        "pseudo_left_waist": (pseudo_left_sh[0], waist_y),
        "pseudo_right_waist": (pseudo_right_sh[0], waist_y),
    }


def get_cloth_bbox(alpha):
    _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return 0, 0, TARGET_W, TARGET_H
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x + w, y + h


def clean_cloth_edges(cloth_rgba, erode_px=6, flood_iter=25):
    rgb = cloth_rgba[:, :, :3].astype(np.float32)
    alpha = cloth_rgba[:, :, 3].copy()

    _, hard = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    k_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
    )
    interior = cv2.erode(hard, k_erode, iterations=1)

    interior_color = rgb.copy()
    k_flood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for _ in range(flood_iter):
        dilated_color = np.stack(
            [
                cv2.dilate(interior_color[:, :, c], k_flood)
                for c in range(3)
            ],
            axis=-1,
        )
        not_interior = (interior == 0)[:, :, np.newaxis]
        interior_color = np.where(not_interior, dilated_color, interior_color)

    is_interior_3ch = np.stack([interior == 255] * 3, axis=-1)
    rgb_clean = np.where(is_interior_3ch, rgb, interior_color)

    result = cloth_rgba.copy().astype(np.float32)
    result[:, :, :3] = np.clip(rgb_clean, 0, 255)
    result[:, :, 3] = alpha
    return result.astype(np.uint8)


def warp_cloth_to_body(cloth_rgba, pseudo_pts):
    cloth_rgba = clean_cloth_edges(cloth_rgba, erode_px=6, flood_iter=25)

    alpha = cloth_rgba[:, :, 3]
    _, hard = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    k_trim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    trimmed_mask = cv2.erode(hard, k_trim, iterations=2)

    cloth_clean = cloth_rgba.copy()
    cloth_clean[:, :, 3] = trimmed_mask

    x_min, y_min, x_max, y_max = get_cloth_bbox(trimmed_mask)

    src_pts = np.float32(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    )
    dst_pts = np.float32(
        [
            list(pseudo_pts["pseudo_left_shoulder"]),
            list(pseudo_pts["pseudo_right_shoulder"]),
            list(pseudo_pts["pseudo_right_waist"]),
            list(pseudo_pts["pseudo_left_waist"]),
        ]
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_channels = [
        cv2.warpPerspective(
            cloth_clean[:, :, c],
            M,
            (TARGET_W, TARGET_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for c in range(4)
    ]
    warped_rgba = np.stack(warped_channels, axis=-1)

    _, final_alpha = cv2.threshold(
        warped_rgba[:, :, 3], 180, 255, cv2.THRESH_BINARY
    )
    warped_rgba = warped_rgba.copy()
    warped_rgba[:, :, 3] = final_alpha

    return warped_rgba, src_pts, dst_pts


def composite_cloth_on_person(person_rgb, warped_rgba):
    warped_rgb = warped_rgba[:, :, :3].astype(np.float32)
    warped_alpha = warped_rgba[:, :, 3].astype(np.float32) / 255.0

    hard_alpha = (warped_alpha > 0.5).astype(np.uint8) * 255

    kernel_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(hard_alpha, kernel_in, iterations=2)
    dilated = cv2.dilate(hard_alpha, kernel_out, iterations=1)
    edge_zone = cv2.subtract(dilated, eroded)
    edge_blur = cv2.GaussianBlur(edge_zone, (15, 15), 0)

    soft_alpha = np.clip(
        eroded.astype(np.float32) / 255.0 + edge_blur.astype(np.float32) / 255.0,
        0.0,
        1.0,
    )

    alpha_3ch = np.stack([soft_alpha] * 3, axis=-1)
    person_float = person_rgb.astype(np.float32)
    result = alpha_3ch * warped_rgb + (1.0 - alpha_3ch) * person_float
    return np.clip(result, 0, 255).astype(np.uint8)


# ==============================================================================
# ANTARMUKA MULTI-TAHAP (STREAMLIT INTERFACE)
# ==============================================================================

# Inisialisasi Session State agar data tetap tersimpan antar interaksi komponen
if "cloth_rgba" not in st.session_state:
    st.session_state.cloth_rgba = None
if "raw_keypoints" not in st.session_state:
    st.session_state.raw_keypoints = None
if "person_rgb" not in st.session_state:
    st.session_state.person_rgb = None

# Membuat 3 Tabs Menu Utama
tab1, tab2, tab3 = st.tabs(
    ["📁 TAHAP 1: Preprocess Baju", "👤 TAHAP 2: Keypoints", "✨ TAHAP 3: Virtual Try-On"]
)

# ------------------------------------------------------------------------------
# TAB 1: PREPROCESSING BAJU
# ------------------------------------------------------------------------------
with tab1:
    st.header("Langkah 1 — Background Removal Baju")
    uploaded_cloth = st.file_uploader(
        "Upload Foto Baju (Format .jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_cloth is not None:
        file_bytes = np.asarray(bytearray(uploaded_cloth.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize ke target 768x1024
        img_resized = resize_to_target(img_rgb)

        # Proses Segmentasi Background
        cloth_rgba, cloth_mask = remove_white_background(img_resized)

        # Simpan hasil pemrosesan ke session state
        st.session_state.cloth_rgba = cloth_rgba

        st.success("✅ Preprocessing Baju Selesai!")

        # Visualisasi Side-by-Side 3 Panel
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_resized, caption="① Original Cloth", use_container_width=True)
        with col2:
            st.image(
                cloth_mask,
                caption="② Cloth Mask (Grayscale)",
                use_container_width=True,
            )
        with col3:
            st.image(
                cloth_rgba,
                caption="③ Cloth RGBA (Transparan)",
                use_container_width=True,
            )

        # Print Info Matriks Metadata seperti di Colab
        st.info(
            f"**Shape Output**: {cloth_rgba.shape} | **Pixel Baju**: {np.sum(cloth_mask > 0):,} px | **Pixel BG**: {np.sum(cloth_mask == 0):,} px"
        )
    else:
        st.warning("Silakan unggah foto baju terlebih dahulu.")

# ------------------------------------------------------------------------------
# TAB 2: Ekstraksi Koordinat Tubuh dari OpenPose JSON
# ------------------------------------------------------------------------------
with tab2:
    st.header("Langkah 2 — Deteksi Pose Manusia")
    col_u1, col_u2 = st.columns(2)

    with col_u1:
        uploaded_person = st.file_uploader(
            "Upload Foto Orang", type=["jpg", "jpeg", "png"]
        )
    with col_u2:
        uploaded_json = st.file_uploader(
            "Upload OpenPose JSON (.json)", type=["json"]
        )

    if uploaded_person is not None and uploaded_json is not None:
        # Load foto orang
        p_bytes   = np.asarray(bytearray(uploaded_person.read()), dtype=np.uint8)
        p_img_bgr = cv2.imdecode(p_bytes, cv2.IMREAD_COLOR)
        p_img_rgb = cv2.cvtColor(p_img_bgr, cv2.COLOR_BGR2RGB)
        st.session_state.person_rgb = resize_to_target(p_img_rgb)

        # Load JSON
        json_data = json.load(uploaded_json)
        raw_pts   = parse_keypoints_from_json(json_data)

        if raw_pts is None:
            st.error("❌ Gagal memproses koordinat OpenPose. Pastikan Neck, Shoulders, Mid-Hip terdeteksi.")
        else:
            st.session_state.raw_keypoints = raw_pts
            st.success("✅ File Pose JSON Berhasil Dimuat!")

            person_rgb = st.session_state.person_rgb
            h, w       = person_rgb.shape[:2]

            # --- Panel ②: Skeleton di background hitam ---
            SKELETON_CONNECTIONS = [
                ("neck", "right_shoulder"), ("neck", "left_shoulder"),
                ("right_shoulder", "right_hip"), ("left_shoulder", "left_hip"),
                ("right_hip", "left_hip"), ("neck", "mid_hip"),
            ]
            skeleton_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            for kp_a, kp_b in SKELETON_CONNECTIONS:
                pt_a, pt_b = raw_pts.get(kp_a), raw_pts.get(kp_b)
                if pt_a and pt_b:
                    cv2.line(skeleton_canvas, pt_a, pt_b, (100, 200, 100), 4)
            for name, pt in raw_pts.items():
                if pt:
                    cv2.circle(skeleton_canvas, pt, 8, (255, 200, 0), -1)

            # --- Panel ③: Foto orang + raw keypoints dari JSON ---
            overlay = person_rgb.copy()

            # Warna per keypoint
            kp_colors = {
                "neck":           (0,0,0),   
                "left_shoulder":  (0,0,0),   
                "right_shoulder": (0,0,0),   
                "mid_hip":        (0,0,0),   
                "left_hip":       (0,0,0),   
                "right_hip":      (0,0,0),   
            }

            for name, pt in raw_pts.items():
                if pt:
                    color = kp_colors.get(name, (255, 0, 0))
                    cv2.circle(overlay, pt, 10, color, -1)
                    cv2.circle(overlay, pt, 10, (255, 255, 255), 2)

                    label = name.replace("_", " ")
                    font       = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness  = 1

                    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    text_x = pt[0] - text_w // 2       # horizontal: tengah
                    text_y = pt[1] - 10 - 6            # vertical: di atas titik (radius=10, padding=6)

                    cv2.putText(overlay, label,
                                (text_x, text_y),
                                font, font_scale, (0, 0, 0), thickness + 1)   # shadow hitam
                    cv2.putText(overlay, label,
                                (text_x, text_y),
                                font, font_scale, (255, 0, 0), thickness)

            # Garis penghubung antar keypoint
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

            # --- Tampilkan 3 panel ---
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
            # Menampilkan koordinat log
            with st.expander("Lihat Koordinat Keypoints Mentah"):
                st.json(raw_pts)
    else:
        st.warning("Silakan unggah Foto Orang dan File OpenPose JSON.")

# ------------------------------------------------------------------------------
# TAB 3: WARPING & VIRTUAL TRY-ON (EKSPEKTASI TAMPILAN SESUAI GAMBAR)
# ------------------------------------------------------------------------------
with tab3:
    st.header("Langkah 3 — Interaktif Kontrol Fit & Warping")

    # Validasi kesiapan data dari Tahap 1 & 2
    if (
        st.session_state.cloth_rgba is None
        or st.session_state.raw_keypoints is None
    ):
        st.error(
            "⚠️ Data Belum Lengkap! Pastikan Anda telah mengunggah berkas baju di Tahap 1 serta foto orang & JSON di Tahap 2."
        )
    else:
        # Input parameters controller dinamis menggunakan Sliders Streamlit
        st.subheader("🎛️ Parameter Kontrol Posisi Baju")
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            ext_ratio = st.slider(
                "Extend Ratio (Lebar Bahu)",
                min_value=0.1,
                max_value=0.6,
                value=0.35,
                step=0.01,
            )
        with col_s2:
            waist_fac = st.slider(
                "Waist Factor (Panjang Kebawah)",
                min_value=0.4,
                max_value=0.9,
                value=0.80,
                step=0.01,
            )
        with col_s3:
            sh_lift = st.slider(
                "Shoulder Lift (Geser Atas)",
                min_value=0.0,
                max_value=0.4,
                value=0.20,
                step=0.01,
            )

        # Jalankan Logika Komposisi dengan Parameter Baru
        pseudo_pts = compute_pseudo_points(
            st.session_state.raw_keypoints, ext_ratio, waist_fac, sh_lift
        )
        warped_rgba, src_pts, dst_pts = warp_cloth_to_body(
            st.session_state.cloth_rgba, pseudo_pts
        )
        result_rgb = composite_cloth_on_person(
            st.session_state.person_rgb, warped_rgba
        )

        # ----------------------------------------------------------------------
        # PROSES VISUALISASI MATPLOTLIB (SAMA PERSIS SEPERTI OUTPUT GAMBAR COLAB)
        # ----------------------------------------------------------------------

        # Panel 1: Cloth Bounding Box
        cloth_display = st.session_state.cloth_rgba[:, :, :3].copy()
        cv2.polylines(
            cloth_display,
            [src_pts.astype(np.int32).reshape(-1, 1, 2)],
            True,
            (255, 200, 0),
            4,
        )

        # Panel 2: Person + Target Overlay Bounding Box Box Trapesium
        person_overlay = st.session_state.person_rgb.copy()
        dst_order = [
            "pseudo_left_shoulder",
            "pseudo_right_shoulder",
            "pseudo_right_waist",
            "pseudo_left_waist",
        ]
        dst_box = np.array([pseudo_pts[k] for k in dst_order], dtype=np.int32)
        cv2.polylines(
            person_overlay, [dst_box.reshape(-1, 1, 2)], True, (255, 255, 0), 3
        )

        # Menggambar titik penanda
        for k in dst_order:
            pt = pseudo_pts[k]
            col = (255, 80, 80) if "shoulder" in k else (80, 80, 255)
            cv2.circle(person_overlay, pt, 12, col, -1)
            cv2.circle(person_overlay, pt, 12, (255, 255, 255), 2)

        # Render 3 Panel Utama Secara Berdampingan Horizontal menggunakan Matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(18, 7.5))
        fig.suptitle(
            "TAHAP 3 — Virtual Try-On Interaktif Hasil Simulasi",
            fontsize=14,
            fontweight="bold",
        )

        axes[0].imshow(cloth_display)
        axes[0].set_title(
            "① Cloth RGBA\n(kotak kuning = area warp src)", fontsize=11
        )
        axes[0].axis("off")

        axes[1].imshow(person_overlay)
        axes[1].set_title(
            "② Foto Orang + Dst Points\n(merah=bahu, biru=waist)", fontsize=11
        )
        axes[1].axis("off")

        axes[2].imshow(result_rgb)
        axes[2].set_title("③ HASIL Virtual Try-On ✅", fontsize=11)
        axes[2].axis("off")

        plt.tight_layout()

        # Tampilkan plot matplotlib ke Streamlit UI
        st.pyplot(fig)