# 🫁 FindingPneumo

Aplikasi web lokal untuk **skrining pneumonia dari citra X-ray dada** menggunakan
**ensemble 3 model deep learning** (DenseNet121 · ResNet50 · InceptionV3) dengan
visualisasi **Grad-CAM** dan laporan **PDF**.

> ⚠️ **Disclaimer medis.** FindingPneumo adalah **alat bantu skrining berbasis AI**,
> bukan alat diagnosis. Hasil prediksi tidak menggantikan pemeriksaan, diagnosis,
> maupun keputusan klinis dokter. Selalu korelasikan dengan kondisi klinis pasien.

---

## ✨ Fitur

- 🔐 **Autentikasi dokter** (registrasi/login, password ter-hash, role Radiolog / Dokter Umum)
- 🧑‍⚕️ **Manajemen pasien** — CRUD + pencarian + ID otomatis `PNM-XXX`
- 📊 **Dashboard** — metrik real (total pasien, scan hari ini, pneumonia rate) + scan terbaru
- 🤖 **Analisis AI** — upload X-ray → **ensemble 3 model** → label + confidence
- 🔥 **Grad-CAM** — 3 heatmap (satu per arsitektur) menunjukkan area yang diperhatikan model
- 📝 **Catatan dokter** + **laporan PDF** resmi siap unduh

## 🛠️ Teknologi

| Lapisan | Teknologi |
|---|---|
| Backend | Python · Flask (application factory + Blueprint) |
| Database | SQLite via SQLAlchemy |
| AI | TensorFlow / Keras (3 model `.keras`), OpenCV, Pillow, NumPy |
| Laporan | ReportLab |
| Frontend | HTML + Tailwind CSS (CDN) + Vanilla JS |

## 📁 Struktur proyek

```
pneumonia-detector/
├── run.py                  # entry point  -> python run.py
├── requirements.txt
├── README.md
├── models/                 # file .keras (TIDAK di-repo, lihat di bawah)
└── app/
    ├── __init__.py         # create_app() + route index & dashboard
    ├── extensions.py       # db, login_manager
    ├── models.py           # skema DB: User, Patient, Scan
    ├── routes/             # blueprint halaman: auth, patients, scan
    ├── services/           # logika non-web: ai_engine (inferensi), report (PDF)
    ├── templates/          # halaman HTML (Jinja2)
    └── static/uploads/     # X-ray & heatmap hasil (TIDAK di-repo)
```

## 🚀 Cara menjalankan (lokal)

```bash
# 1. Masuk folder & install dependency
cd pneumonia-detector
pip install -r requirements.txt

# 2. Taruh 3 file model (lihat bagian Model di bawah) ke:
#    models/list_models/densenet121/densenet121.keras
#    models/list_models/resnet50/resnet50.keras
#    models/list_models/inception/inception.keras

# 3. Jalankan
python run.py
```
Buka **http://127.0.0.1:5000** → Registrasi dokter → Login → Patients → New Scan.

> Scan pertama butuh ~10–30 detik karena memuat ketiga model. Itu normal.

## 🧠 Model

File model `.keras` (~420 MB total) **tidak disertakan di repository** karena
melebihi batas ukuran GitHub. Bagikan secara terpisah (Google Drive / dsb) dan
letakkan di `models/list_models/<arsitektur>/`.

Preprocessing per arsitektur (sudah ditangani otomatis di `app/services/ai_engine.py`):
- **DenseNet121** — input 299×299, raw 0–255 (preprocessing di dalam model), Grad-CAM `relu`
- **ResNet50** — input 256×256, `resnet50.preprocess_input`, Grad-CAM `conv5_block3_out`
- **InceptionV3** — input 299×299, normalisasi `/255`, Grad-CAM `mixed10`

Keputusan akhir = **rata-rata probabilitas** ketiga model (soft voting), ambang **0.5**.

## 🛡️ Guardrails (keamanan & keandalan)

- Semua halaman sensitif dilindungi **login** (`@login_required`).
- Upload divalidasi: **tipe** (PNG/JPG/JPEG/BMP/WEBP) & **ukuran** (maks 12 MB).
- **Tahan-banting**: bila satu model gagal dimuat, ensemble tetap berjalan dari sisanya.
- Model dimuat **satu per satu lalu dilepas** → menghindari kehabisan memori.
- **Peringatan confidence rendah** ditampilkan saat keyakinan model < 65%.
- **Disclaimer medis** tampil di UI dan pada setiap laporan PDF.
- `SECRET_KEY` & mode `debug` dapat diatur lewat environment variable.

## 👥 Tim

Proyek Akhir — Artificial Intelligence (Semester 4).

---

*Catatan: aplikasi ditujukan untuk lingkungan lokal/edukasi, bukan deployment produksi.*
