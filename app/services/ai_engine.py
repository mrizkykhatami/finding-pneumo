"""
ai_engine.py — Antarmuka inferensi AI untuk PneumoScan.

╔══════════════════════════════════════════════════════════════════════════╗
║  STATUS SAAT INI: MODE MOCK (dummy).                                       ║
║  Belum memakai TensorFlow / file .h5 — tujuannya agar SELURUH alur web     ║
║  (upload → hasil → 3 heatmap → simpan → PDF) bisa diuji lebih dulu.        ║
║                                                                            ║
║  CARA MENGAKTIFKAN MODEL ASLI (Fase 3 - integrasi akhir):                  ║
║    1. Taruh 3 file .h5 di folder models/.                                   ║
║    2. Isi fungsi _predict_real() di bawah dengan loading model +           ║
║       preprocessing per-arsitektur + Grad-CAM (kode sudah disiapkan        ║
║       kerangkanya, tinggal di-uncomment & disesuaikan).                     ║
║    3. Set USE_REAL_MODELS = True.                                          ║
║                                                                            ║
║  Tanda tangan analyze_scan() TIDAK berubah, jadi web tidak perlu diubah.   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import hashlib
import os
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Ganti ke True setelah model .h5 siap & _predict_real() diisi.
USE_REAL_MODELS = False

# Label tiga arsitektur — urutan ini dipakai konsisten di seluruh aplikasi.
MODEL_NAMES = ("DenseNet121", "ResNet50", "InceptionV3")

# Ukuran kanvas heatmap yang ditampilkan di web (px).
_DISPLAY_SIZE = 420


# ──────────────────────────────────────────────────────────────────────────
#  API PUBLIK — dipakai oleh blueprint scan.py
# ──────────────────────────────────────────────────────────────────────────
def analyze_scan(original_abs_path: str, out_dir_abs: str, basename: str) -> dict:
    """
    Jalankan analisis lengkap pada satu gambar X-ray.

    Args:
        original_abs_path : path absolut file X-ray yang sudah disimpan.
        out_dir_abs       : folder absolut tempat menyimpan heatmap (mis. static/uploads).
        basename          : nama dasar unik untuk file output (tanpa ekstensi).

    Returns:
        dict {
          'label'      : 'PNEUMONIA' | 'NORMAL',
          'confidence' : float 0..1,
          'heatmaps'   : [path_relatif_1, path_relatif_2, path_relatif_3],  # urut MODEL_NAMES
          'is_mock'    : bool,
        }
        Path heatmap relatif terhadap folder static/ (mis. 'uploads/xxx_densenet.png').
    """
    if USE_REAL_MODELS:
        return _predict_real(original_abs_path, out_dir_abs, basename)
    return _predict_mock(original_abs_path, out_dir_abs, basename)


# ──────────────────────────────────────────────────────────────────────────
#  IMPLEMENTASI MOCK
# ──────────────────────────────────────────────────────────────────────────
def _predict_mock(original_abs_path: str, out_dir_abs: str, basename: str) -> dict:
    """
    Hasil dummy yang DETERMINISTIK (sama gambar → sama hasil) supaya demo stabil.
    Heatmap dibuat dengan menempel 'hotspot' warna di atas X-ray asli + watermark MOCK.
    """
    # Label & confidence diturunkan dari hash file → stabil, terlihat acak.
    digest = _file_digest(original_abs_path)
    seed = int(digest[:8], 16)
    is_pneumonia = (seed % 100) >= 45          # ~55% kemungkinan pneumonia
    base_conf = 0.62 + (seed % 37) / 100.0      # 0.62 .. 0.98
    confidence = round(min(base_conf, 0.98), 4)
    label = "PNEUMONIA" if is_pneumonia else "NORMAL"

    base = _load_base_image(original_abs_path)

    # Tiga "hotspot" berbeda posisi/warna agar tiap model terlihat khas.
    spots = [
        ((0.40, 0.55), (220, 60, 50)),    # DenseNet — merah, paru kiri-bawah
        ((0.62, 0.45), (240, 140, 30)),   # ResNet   — oranye, paru kanan-tengah
        ((0.50, 0.38), (250, 210, 40)),   # Inception— kuning, atas-tengah
    ]

    rel_paths = []
    for (name, ((cx, cy), color)) in zip(MODEL_NAMES, spots):
        # Untuk NORMAL, hotspot dibuat lebih lemah (seakan tak ada fokus patologis).
        intensity = 1.0 if is_pneumonia else 0.35
        heat = _compose_mock_heatmap(base, (cx, cy), color, intensity, name)
        out_name = f"{basename}_{name.lower()}.png"
        heat.save(os.path.join(out_dir_abs, out_name))
        rel_paths.append(f"uploads/{out_name}")

    return {
        "label": label,
        "confidence": confidence,
        "heatmaps": rel_paths,
        "is_mock": True,
    }


def _load_base_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    # Resize menjaga aspek lalu pad jadi kotak agar tampil rapi.
    img.thumbnail((_DISPLAY_SIZE, _DISPLAY_SIZE), Image.LANCZOS)
    canvas = Image.new("RGB", (_DISPLAY_SIZE, _DISPLAY_SIZE), (10, 12, 16))
    canvas.paste(img, ((_DISPLAY_SIZE - img.width) // 2, (_DISPLAY_SIZE - img.height) // 2))
    return canvas


def _compose_mock_heatmap(base: Image.Image, center, color, intensity: float, tag: str) -> Image.Image:
    """Tempel gradien radial (hotspot) di atas gambar + label MOCK."""
    w, h = base.size
    cx, cy = int(center[0] * w), int(center[1] * h)
    max_r = int(0.42 * w)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Lingkaran konsentris: makin ke dalam makin pekat → kesan heatmap.
    steps = 40
    for i in range(steps, 0, -1):
        r = int(max_r * i / steps)
        a = int(150 * intensity * (1 - i / steps) ** 1.5)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (a,))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max_r * 0.12))

    out = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    # Watermark "MOCK" supaya jelas ini bukan hasil model asli.
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    d.text((10, 10), f"{tag} · MOCK", fill=(255, 255, 255), font=font)
    return out


def _file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────
#  IMPLEMENTASI MODEL ASLI — diisi pada tahap integrasi akhir
# ──────────────────────────────────────────────────────────────────────────
def _predict_real(original_abs_path: str, out_dir_abs: str, basename: str) -> dict:
    """
    KERANGKA untuk model asli. Saat ini sengaja melempar error supaya tidak
    terpakai sebelum benar-benar diisi.

    Rencana implementasi (sesuai spesifikasi preprocessing yang sudah ditetapkan):
      - DenseNet121 : input (?, ?, 3), densenet.preprocess_input, Grad-CAM dari conv terakhir.
      - ResNet50    : input 256x256,   resnet50.preprocess_input, sub-model 'resnet50' + head.
      - InceptionV3 : input 299x299,   inception_v3.preprocess_input, Grad-CAM dari 'mixed10'.
      - Ensemble    : rata-rata confidence 3 model → label final.

    Catatan: ukuran input final DenseNet & Inception masih perlu dikonfirmasi
    dari model.input_shape (lihat catatan di chat).
    """
    raise NotImplementedError(
        "Model asli belum diaktifkan. Isi _predict_real() lalu set USE_REAL_MODELS=True."
    )
