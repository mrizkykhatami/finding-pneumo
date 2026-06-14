"""
Skema database — 3 tabel sesuai spesifikasi project.

Catatan desain:
- Password TIDAK pernah disimpan plaintext. Disimpan sebagai hash (Werkzeug).
- Patient.id memakai format string 'PNM-XXX' (auto-generate diisi di Fase 2).
- Scan menyimpan 1 path gambar original + 3 path heatmap (DenseNet/ResNet/Inception)
  sesuai keputusan ensemble 3 model. Kolom AI masih nullable di Fase 1.
"""
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app.extensions import db


class User(UserMixin, db.Model):
    """Dokter yang login ke sistem (Radiolog / Dokter Umum)."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(30), nullable=False, default="Dokter Umum")  # 'Radiolog' | 'Dokter Umum'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relasi: satu dokter bisa membuat banyak scan
    scans = db.relationship("Scan", backref="doctor", lazy=True)

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

    def __repr__(self) -> str:
        return f"<User {self.email} ({self.role})>"


class Patient(db.Model):
    """Data pasien. ID berformat PNM-XXX (auto-generate di Fase 2)."""
    __tablename__ = "patients"

    id = db.Column(db.String(20), primary_key=True)  # contoh: 'PNM-001'
    nama = db.Column(db.String(120), nullable=False)
    umur = db.Column(db.Integer, nullable=True)
    jenis_kelamin = db.Column(db.String(20), nullable=True)  # 'L' | 'P'
    kontak = db.Column(db.String(50), nullable=True)
    gejala = db.Column(db.Text, nullable=True)
    riwayat_medis = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relasi: satu pasien bisa punya banyak scan
    scans = db.relationship("Scan", backref="patient", lazy=True)

    def __repr__(self) -> str:
        return f"<Patient {self.id} - {self.nama}>"


class Scan(db.Model):
    """Hasil pemindaian X-ray + output AI (diisi mulai Fase 3)."""
    __tablename__ = "scans"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), db.ForeignKey("patients.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    tanggal = db.Column(db.DateTime, default=datetime.utcnow)

    # Output AI — nullable karena baru terisi saat inferensi (Fase 3)
    hasil_prediksi = db.Column(db.String(30), nullable=True)      # 'PNEUMONIA' | 'NORMAL'
    confidence_score = db.Column(db.Float, nullable=True)         # 0.0 - 1.0

    # Path file gambar
    file_path_original = db.Column(db.String(255), nullable=True)
    file_path_heatmap_1 = db.Column(db.String(255), nullable=True)  # DenseNet121
    file_path_heatmap_2 = db.Column(db.String(255), nullable=True)  # ResNet50
    file_path_heatmap_3 = db.Column(db.String(255), nullable=True)  # InceptionV3
    file_path_heatmap_consensus = db.Column(db.String(255), nullable=True)  # irisan 3 model

    doctor_notes = db.Column(db.Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Scan {self.id} - {self.patient_id} - {self.hasil_prediksi}>"
