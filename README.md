# FindingPneumo

Aplikasi web untuk **skrining pneumonia dari citra X-ray dada** menggunakan **ensemble 3 model deep learning** (DenseNet121 · ResNet50 · InceptionV3) dengan visualisasi **Grad-CAM** dan laporan **PDF**.

> **Disclaimer medis.** FindingPneumo adalah **alat bantu skrining berbasis AI**,
> bukan alat diagnosis. Hasil prediksi tidak menggantikan pemeriksaan, diagnosis,
> maupun keputusan klinis dokter. Selalu korelasikan dengan kondisi klinis pasien.

---

## Fitur

- **Autentikasi dokter:** registrasi/login, password ter-hash, role Radiolog / Dokter Umum
- **Manajemen pasien:** CRUD + pencarian + ID otomatis `PNM-XXX`
- **Dashboard:** total pasien, scan hari ini, pneumonia rate, report generated
- **Analisis AI:** upload citra X-ray → ensemble 3 model → label + confidence 
- **Grad-CAM:** 4 heatmap (satu per arsitektur model + konsensus/irisan) menunjukkan area yang diperhatikan model
- **Catatan dokter** + **laporan PDF** siap unduh

## Teknologi

| Lapisan | Teknologi |
|---|---|
| Backend | Python · Flask (application factory + Blueprint) |
| Database | SQLite via SQLAlchemy |
| AI | TensorFlow / Keras (3 model `.keras`), OpenCV, Pillow, NumPy |
| Laporan | ReportLab |
| Frontend | HTML + Tailwind CSS (CDN) + Vanilla JS |

## Struktur proyek

```
finding-pneumo/
├── run.py                  # entry point  -> python run.py
├── requirements.txt
├── README.md
├── models/                 # file .keras
└── app/
    ├── __init__.py         # create_app() + route index & dashboard
    ├── extensions.py       # db, login_manager
    ├── models.py           # skema DB: User, Patient, Scan
    ├── routes/             # blueprint halaman: auth, patients, scan
    ├── services/           # logika non-web: ai_engine (inferensi), report (PDF)
    ├── templates/          # halaman HTML (Jinja2)
    └── static/uploads/     # X-ray & heatmap hasil (TIDAK di-repo)
```

## Cara menjalankan (lokal)

```bash
# 1. Masuk folder & install dependency
cd finding-pneumo
pip install -r requirements.txt

# 2. Taruh 3 file model (lihat bagian Model di bawah) ke:
#    models/list_models/densenet121/densenet121.keras
#    models/list_models/resnet50/resnet50.keras
#    models/list_models/inception/inception.keras

# 3. Jalankan
python run.py
```
Buka **http://127.0.0.1:5000** → Registrasi dokter → Login → Patients → New Scan.

> Scan pertama mungkin butuh waktu lebih lama karena memuat ketiga model.

## Model

File model `.keras` memiliki ukuran total yang cukup besar (~420 MB total) disimpan menggunakan Git Large File Storage (LFS) dan diletakkan pada `models/list_models/<arsitektur>/`.

Preprocessing per arsitektur (sudah ditangani otomatis di `app/services/ai_engine.py`):
- **DenseNet121:** input 299×299, raw 0-255 (preprocessing di dalam model), Grad-CAM `relu`
- **ResNet50:** input 256×256, `resnet50.preprocess_input`, Grad-CAM `conv5_block3_out`
- **InceptionV3:** input 299×299, normalisasi `/255`, Grad-CAM `mixed10`

Keputusan akhir = **rata-rata probabilitas** ketiga model (soft voting), ambang **0.5**.

## Guardrails (Keamanan & Keandalan)

- Semua halaman sensitif dilindungi **login** (`@login_required`).
- Upload divalidasi: **tipe** (PNG/JPG/JPEG/BMP/WEBP) & **ukuran** (maks 12 MB).
- Bila satu model gagal dimuat, ensemble tetap berjalan dari sisanya.
- Model dimuat **satu per satu lalu dilepas** → menghindari kehabisan memori.
- **Peringatan confidence rendah** ditampilkan saat keyakinan model < 65%.
- **Disclaimer medis** tampil di UI dan pada setiap laporan PDF.

## Tim

Proyek Artificial Intelligence
| NPM | Nama |
| --- | --- |
| 140810240029 | Hamzah Abdillah Gabriela |
| 140810240061 | Renadi Wilantara |
| 140810240073 | Muhammad Rizky Khatami |

Teknik Informatika  
Universitas Padjadjaran

---

