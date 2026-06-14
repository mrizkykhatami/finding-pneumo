"""
ai_engine.py — Antarmuka inferensi AI untuk FindingPneumo.

Sekarang aplikasi SELALU memakai inferensi nyata: ensemble 3 model .keras asli
(DenseNet121 + ResNet50 + InceptionV3) + Grad-CAM per-arsitektur. Lihat
_predict_real().

CATATAN: Mode MOCK/demo (hasil & heatmap dummy tanpa TensorFlow) sudah TIDAK
DIPAKAI sejak ketiga model asli terpasang. Kodenya tetap disimpan di bawah dalam
bentuk komentar untuk referensi/laporan.

analyze_scan() adalah satu-satunya API yang dipakai web.
"""
from __future__ import annotations

import os
from PIL import Image

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
          'consensus'  : path_relatif | None,  # irisan (perkalian) 3 heatmap; None bila <2 model sukses
          'is_mock'    : bool,
        }
        Path heatmap relatif terhadap folder static/ (mis. 'uploads/xxx_densenet.png').
    """
    # Selalu pakai model asli. (Mode mock sudah dipensiunkan — lihat blok komentar di bawah.)
    return _predict_real(original_abs_path, out_dir_abs, basename)


# ──────────────────────────────────────────────────────────────────────────
#  IMPLEMENTASI MOCK  —  TIDAK DIPAKAI LAGI (disimpan sebagai komentar)
#  Dipensiunkan setelah 3 model .keras asli terpasang. Untuk mengaktifkan lagi,
#  uncomment blok ini, kembalikan import (hashlib, ImageDraw/ImageFilter/ImageFont),
#  lalu panggil _predict_mock() dari analyze_scan().
# ──────────────────────────────────────────────────────────────────────────
# def _predict_mock(original_abs_path: str, out_dir_abs: str, basename: str) -> dict:
#     """
#     Hasil dummy yang DETERMINISTIK (sama gambar → sama hasil) supaya demo stabil.
#     Heatmap dibuat dengan menempel 'hotspot' warna di atas X-ray asli + watermark MOCK.
#     """
#     # Label & confidence diturunkan dari hash file → stabil, terlihat acak.
#     digest = _file_digest(original_abs_path)
#     seed = int(digest[:8], 16)
#     is_pneumonia = (seed % 100) >= 45          # ~55% kemungkinan pneumonia
#     base_conf = 0.62 + (seed % 37) / 100.0      # 0.62 .. 0.98
#     confidence = round(min(base_conf, 0.98), 4)
#     label = "PNEUMONIA" if is_pneumonia else "NORMAL"
#
#     base = _load_base_image(original_abs_path)
#
#     # Tiga "hotspot" berbeda posisi/warna agar tiap model terlihat khas.
#     spots = [
#         ((0.40, 0.55), (220, 60, 50)),    # DenseNet — merah, paru kiri-bawah
#         ((0.62, 0.45), (240, 140, 30)),   # ResNet   — oranye, paru kanan-tengah
#         ((0.50, 0.38), (250, 210, 40)),   # Inception— kuning, atas-tengah
#     ]
#
#     rel_paths = []
#     for (name, ((cx, cy), color)) in zip(MODEL_NAMES, spots):
#         # Untuk NORMAL, hotspot dibuat lebih lemah (seakan tak ada fokus patologis).
#         intensity = 1.0 if is_pneumonia else 0.35
#         heat = _compose_mock_heatmap(base, (cx, cy), color, intensity, name)
#         out_name = f"{basename}_{name.lower()}.png"
#         heat.save(os.path.join(out_dir_abs, out_name))
#         rel_paths.append(f"uploads/{out_name}")
#
#     return {
#         "label": label,
#         "confidence": confidence,
#         "heatmaps": rel_paths,
#         "is_mock": True,
#     }
#
#
# def _load_base_image(path: str) -> Image.Image:
#     img = Image.open(path).convert("RGB")
#     # Resize menjaga aspek lalu pad jadi kotak agar tampil rapi.
#     img.thumbnail((_DISPLAY_SIZE, _DISPLAY_SIZE), Image.LANCZOS)
#     canvas = Image.new("RGB", (_DISPLAY_SIZE, _DISPLAY_SIZE), (10, 12, 16))
#     canvas.paste(img, ((_DISPLAY_SIZE - img.width) // 2, (_DISPLAY_SIZE - img.height) // 2))
#     return canvas
#
#
# def _compose_mock_heatmap(base: Image.Image, center, color, intensity: float, tag: str) -> Image.Image:
#     """Tempel gradien radial (hotspot) di atas gambar + label MOCK."""
#     w, h = base.size
#     cx, cy = int(center[0] * w), int(center[1] * h)
#     max_r = int(0.42 * w)
#
#     overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
#     draw = ImageDraw.Draw(overlay)
#     # Lingkaran konsentris: makin ke dalam makin pekat → kesan heatmap.
#     steps = 40
#     for i in range(steps, 0, -1):
#         r = int(max_r * i / steps)
#         a = int(150 * intensity * (1 - i / steps) ** 1.5)
#         draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (a,))
#     overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max_r * 0.12))
#
#     out = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
#
#     # Watermark "MOCK" supaya jelas ini bukan hasil model asli.
#     d = ImageDraw.Draw(out)
#     try:
#         font = ImageFont.truetype("arial.ttf", 16)
#     except OSError:
#         font = ImageFont.load_default()
#     d.text((10, 10), f"{tag} · MOCK", fill=(255, 255, 255), font=font)
#     return out
#
#
# def _file_digest(path: str) -> str:
#     h = hashlib.sha256()
#     with open(path, "rb") as f:
#         for chunk in iter(lambda: f.read(8192), b""):
#             h.update(chunk)
#     return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────
#  IMPLEMENTASI MODEL ASLI (ensemble 3 model + Grad-CAM per-arsitektur)
# ──────────────────────────────────────────────────────────────────────────
#
# Spesifikasi diturunkan dari inspeksi langsung tiap file .keras:
#   DenseNet121 : input 299, TIDAK ada layer normalisasi di dalam model (hanya
#                 augmentasi rotation/translation yang no-op saat inference) ->
#                 input WAJIB lewat densenet.preprocess_input (mode 'densenet').
#                 base 'densenet121', conv terakhir 'relu'.
#   ResNet50    : input 256, ada Lambda 'preprocess_input_layer' DI DALAM model,
#                 tapi untuk Grad-CAM kita panggil sub-model 'resnet50' yang butuh
#                 resnet50.preprocess_input. conv terakhir 'conv5_block3_out'.
#   InceptionV3 : input 299, model flat TANPA preprocessing -> feed /255.0
#                 (sesuai metadata 'normalization_scale: 1/255.0'). conv 'mixed10'.
#
# Urutan _SPECS = urutan MODEL_NAMES (DenseNet, ResNet, Inception).
_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "list_models",
)
_SPECS = [
    {"name": "DenseNet121", "file": os.path.join(_MODELS_DIR, "densenet121", "densenet121.keras"),
     "size": 299, "prep": "densenet", "base_layer": "densenet121", "conv_layer": "relu"},
    {"name": "ResNet50",    "file": os.path.join(_MODELS_DIR, "resnet50", "resnet50.keras"),
     "size": 256, "prep": "resnet", "base_layer": "resnet50",    "conv_layer": "conv5_block3_out"},
    {"name": "InceptionV3", "file": os.path.join(_MODELS_DIR, "inception", "inception.keras"),
     "size": 299, "prep": "div255", "base_layer": None,          "conv_layer": "mixed10"},
]


def _predict_real(original_abs_path: str, out_dir_abs: str, basename: str) -> dict:
    """
    Inferensi nyata: ensemble 3 model + Grad-CAM per-arsitektur.

    Strategi memori: model dimuat SATU per satu lalu DILEPAS (peak memori = 1 model,
    bukan ~420 MB sekaligus) -> menghindari OOM/segfault di RAM terbatas. Konsekuensi:
    tiap scan agak lebih lama karena model di-load ulang, tapi jauh lebih stabil.

    Tahan-banting: bila satu model gagal, ensemble tetap jalan dari model lainnya.
    """
    import gc
    import traceback
    import numpy as np
    import cv2
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    orig_pil = Image.open(original_abs_path).convert("RGB")

    probs, heatmap_paths, heats_raw = [], [], []
    for spec in _SPECS:
        name = spec["name"]
        out_name = f"{basename}_{name.lower()}.png"
        model = None
        try:
            model = _load_keras(spec["file"], name, tf)
            inp = _prep_input(orig_pil, spec["size"], spec["prep"], np, tf)
            prob, heat = _grad_cam(model, inp, spec, tf)
            probs.append(prob)
            _save_overlay(orig_pil, heat, os.path.join(out_dir_abs, out_name), cv2, np)
            heatmap_paths.append(f"uploads/{out_name}")
            heats_raw.append(heat)  # simpan heatmap 2D mentah utk consensus
        except Exception:  # noqa: BLE001 — satu model gagal tidak mematikan yang lain
            traceback.print_exc()
            heatmap_paths.append(None)
        finally:
            del model
            tf.keras.backend.clear_session()
            gc.collect()

    if not probs:
        raise RuntimeError(
            "Semua model gagal dimuat. Periksa file di models/list_models/."
        )

    # Consensus map: perkalian heatmap antar-model. Hanya titik yang SEMUA model
    # (yang sukses) sama-sama kuat yang bertahan -> menyaring aktivasi 'nyasar'
    # satu model. Butuh ≥2 model sukses agar bermakna sebagai 'irisan'.
    consensus_path = None
    if len(heats_raw) >= 2:
        try:
            consensus_path = _save_consensus(
                orig_pil, heats_raw, os.path.join(out_dir_abs, f"{basename}_consensus.png"),
                cv2, np,
            )
            consensus_path = f"uploads/{basename}_consensus.png"
        except Exception:  # noqa: BLE001 — consensus opsional, jangan matikan analisis
            traceback.print_exc()
            consensus_path = None

    avg = float(sum(probs) / len(probs))
    label = "PNEUMONIA" if avg >= 0.5 else "NORMAL"
    confidence = round(avg if label == "PNEUMONIA" else 1.0 - avg, 4)
    return {
        "label": label,
        "confidence": confidence,
        "heatmaps": heatmap_paths,
        "consensus": consensus_path,
        "is_mock": False,
    }


def _load_keras(path, name, tf):
    """Muat model .keras. ResNet butuh patch Lambda 'preprocess_input_layer'."""
    if name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input as rp
        orig_call = tf.keras.layers.Lambda.call
        orig_shape = tf.keras.layers.Lambda.compute_output_shape

        def patched(self, inputs, *a, **k):
            if self.name == "preprocess_input_layer":
                return rp(inputs)
            return orig_call(self, inputs, *a, **k)

        tf.keras.layers.Lambda.call = patched
        tf.keras.layers.Lambda.compute_output_shape = lambda self, s: s
        try:
            return tf.keras.models.load_model(path, safe_mode=False, compile=False)
        finally:
            tf.keras.layers.Lambda.call = orig_call
            tf.keras.layers.Lambda.compute_output_shape = orig_shape
    return tf.keras.models.load_model(path, safe_mode=False, compile=False)


def _prep_input(orig_pil, size, mode, np, tf):
    arr = np.array(orig_pil.resize((size, size)), dtype="float32")[None, ...]  # (1,H,W,3)
    if mode == "div255":
        arr = arr / 255.0
    elif mode == "resnet":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        arr = preprocess_input(arr)
    elif mode == "densenet":
        from tensorflow.keras.applications.densenet import preprocess_input
        arr = preprocess_input(arr)
    # 'raw' -> biarkan 0-255 (dipakai bila model sudah punya layer normalisasi sendiri)
    return tf.convert_to_tensor(arr)


def _grad_cam(model, inp, spec, tf):
    """Return (prob_pneumonia, heatmap_2d ternormalisasi 0..1)."""
    base_layer, conv_layer = spec["base_layer"], spec["conv_layer"]

    with tf.GradientTape() as tape:
        if base_layer is None:
            # Model flat (Inception): grad_model langsung dari input model.
            grad_model = tf.keras.models.Model(
                model.inputs, [model.get_layer(conv_layer).output, model.output])
            conv_out, pred = grad_model(inp, training=False)
            tape.watch(conv_out)
        else:
            # Model bersub-model (DenseNet/ResNet): hitung conv dari dalam base,
            # lalu rekonstruksi head (layer setelah base) agar tetap 1 graph.
            base = model.get_layer(base_layer)
            grad_model = tf.keras.models.Model(
                base.input, [base.get_layer(conv_layer).output, base.output])
            conv_out, base_out = grad_model(inp, training=False)
            tape.watch(conv_out)
            x, passed = base_out, False
            for layer in model.layers:
                if layer.name == base_layer:
                    passed = True
                    continue
                if passed:
                    x = layer(x, training=False)
            pred = x
        loss = pred[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heat = tf.nn.relu(heat)
    maxv = tf.reduce_max(heat)
    if maxv > 0:
        heat = heat / maxv
    return float(pred[0, 0]), heat.numpy()


def _save_overlay(orig_pil, heat, out_abs, cv2, np, disp=420, thr=0.4):
    """
    Tempel heatmap (colormap JET) di atas X-ray asli, simpan PNG.

    Perbaikan tampilan (tetap jujur, tanpa memalsukan):
    - Jaga aspect ratio asli (gambar TIDAK ditarik & TIDAK diberi bar hitam) ->
      tampil penuh seperti X-ray asli, heatmap sejajar dengan anatomi.
    - Hanya warnai aktivasi KUAT (>= thr dari max). Area lemah dibiarkan polos
      sehingga heatmap terkonsentrasi di titik penting (umumnya area dada).
    - Batasi heatmap HANYA di dalam area tubuh (body mask). Grad-CAM beresolusi
      kasar (mis. 9x9) -> saat di-resize tepinya bisa 'tumpah' ke latar hitam /
      luar dada. Mask membuang tumpahan itu tanpa mengubah aktivasi di dalam paru.
    """
    img = orig_pil.copy()
    img.thumbnail((disp, disp), Image.LANCZOS)  # batasi ukuran, rasio terjaga
    w, h = img.size
    base = np.array(img)[:, :, ::-1]  # RGB -> BGR utk cv2

    hm = np.clip(cv2.resize(heat.astype("float32"), (w, h)), 0, 1)
    hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(base, 0.6, hm_color, 0.4, 0)

    # Mask tubuh: piksel terang = tubuh/dada, gelap = latar. Otsu pisahkan otomatis,
    # morphological close menutup lubang, lalu ambil komponen terbesar (badan utama).
    body = _body_mask(base, cv2, np)

    # warnai hanya di area aktivasi kuat DAN di dalam tubuh; sisanya tetap X-ray asli
    overlay = base.copy()
    mask = (hm >= thr) & body
    overlay[mask] = blended[mask]

    cv2.imwrite(out_abs, overlay)  # simpan apa adanya (rasio asli, tanpa bar)


def _save_consensus(orig_pil, heats_raw, out_abs, cv2, np):
    """
    Bangun & simpan 'consensus map' = irisan aktivasi antar-model.

    Metode: PERKALIAN heatmap (semua sudah ternormalisasi 0..1). Titik yang kuat
    di SEMUA model -> hasil tinggi; titik yang hanya 1 model menyala (mis. aktivasi
    'nyasar' ke perut) -> hasil ~0 dan tersaring. Setelah dikalikan, di-normalisasi
    ulang lalu ditampilkan lewat _save_overlay (reuse body-mask + colormap JET).

    heats_raw: list heatmap 2D (ukuran grid bisa beda antar-arsitektur, mis. 9x9
    vs 8x8) -> diseragamkan dulu ke grid acuan lewat cv2.resize sebelum dikalikan.
    """
    ref_h, ref_w = heats_raw[0].shape[:2]
    cons = np.ones((ref_h, ref_w), dtype="float32")
    for heat in heats_raw:
        h = heat.astype("float32")
        if h.shape[:2] != (ref_h, ref_w):
            h = cv2.resize(h, (ref_w, ref_h))
        cons *= np.clip(h, 0, 1)

    maxv = float(cons.max())
    if maxv > 0:
        cons = cons / maxv  # re-normalisasi agar titik konsensus terkuat = 1.0

    _save_overlay(orig_pil, cons, out_abs, cv2, np)
    return out_abs


def _body_mask(base_bgr, cv2, np):
    """
    Bangun mask boolean (H,W) area tubuh/dada dari X-ray.

    Cara: X-ray punya latar (sangat) gelap dan tubuh lebih terang. Otsu memisah
    keduanya secara otomatis (tanpa angka ambang manual yang rapuh antar gambar),
    morphological close menutup lubang kecil, lalu komponen terhubung TERBESAR
    dipertahankan sebagai badan utama.

    PENTING: hanya OUTLINE LUAR tubuh yang dipakai sebagai batas. Area gelap DI
    DALAM dada (mis. ruang antar-rusuk, rongga paru) ikut diisi penuh (fill holes)
    -> mask tidak 'bolong-bolong' di dalam dada, hanya membuang latar di luar tubuh.

    Fallback: kalau hasil mask aneh (terlalu kecil/besar), kembalikan semua True
    agar tidak menyembunyikan aktivasi yang valid.
    """
    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bw = np.where(labels == biggest, 255, 0).astype("uint8")

    # Isi penuh kontur terluar -> buang lubang di DALAM tubuh (antar-rusuk dll).
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros_like(bw)
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        bw = filled

    frac = float((bw > 0).mean())
    if frac < 0.15 or frac > 0.97:  # mask tidak masuk akal -> jangan mask apa pun
        return np.ones(gray.shape, dtype=bool)
    return bw > 0
