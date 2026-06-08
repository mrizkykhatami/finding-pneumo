"""
Blueprint manajemen pasien (Fase 2): list + search, create, edit, delete.

Catatan desain:
- ID pasien berformat 'PNM-XXX' dan di-generate otomatis (lihat next_patient_id).
- Tombol "New Scan" pada tiap baris mengarah ke halaman scan (Fase 3) sambil
  membawa patient_id lewat query string, sehingga form scan bisa langsung terisi.
"""
from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_required
from sqlalchemy import func

from extensions import db
from models import Patient

patients_bp = Blueprint("patients", __name__)

VALID_GENDER = {"L", "P"}


def next_patient_id() -> str:
    """
    Hasilkan ID pasien berikutnya berformat 'PNM-XXX' (3 digit, zero-padded).

    Strategi: ambil suffix numerik terbesar yang ada, lalu +1. Aman untuk
    aplikasi lokal single-user. Contoh: belum ada -> 'PNM-001', ada PNM-007 -> 'PNM-008'.
    """
    last = (
        Patient.query
        .filter(Patient.id.like("PNM-%"))
        .order_by(func.length(Patient.id).desc(), Patient.id.desc())
        .first()
    )
    if last is None:
        return "PNM-001"
    try:
        n = int(last.id.split("-", 1)[1])
    except (IndexError, ValueError):
        n = 0
    return f"PNM-{n + 1:03d}"


def _read_form():
    """Ambil & rapikan field form pasien menjadi dict."""
    return {
        "nama": request.form.get("nama", "").strip(),
        "umur": request.form.get("umur", "").strip(),
        "jenis_kelamin": request.form.get("jenis_kelamin", "").strip(),
        "kontak": request.form.get("kontak", "").strip(),
        "gejala": request.form.get("gejala", "").strip(),
        "riwayat_medis": request.form.get("riwayat_medis", "").strip(),
    }


def _validate(data) -> str | None:
    """Kembalikan pesan error pertama yang ditemukan, atau None jika valid."""
    if not data["nama"]:
        return "Nama pasien wajib diisi."
    if data["umur"]:
        if not data["umur"].isdigit() or not (0 <= int(data["umur"]) <= 150):
            return "Umur harus berupa angka yang masuk akal (0–150)."
    if data["jenis_kelamin"] and data["jenis_kelamin"] not in VALID_GENDER:
        return "Jenis kelamin tidak valid."
    return None


@patients_bp.route("/patients")
@login_required
def list_patients():
    """Tabel pasien dengan pencarian (nama / ID / kontak)."""
    q = request.args.get("q", "").strip()
    query = Patient.query
    if q:
        like = f"%{q}%"
        query = query.filter(
            db.or_(
                Patient.nama.ilike(like),
                Patient.id.ilike(like),
                Patient.kontak.ilike(like),
            )
        )
    patients = query.order_by(Patient.created_at.desc()).all()
    return render_template("patients.html", patients=patients, q=q)


@patients_bp.route("/patients/new", methods=["GET", "POST"])
@login_required
def create_patient():
    if request.method == "POST":
        data = _read_form()
        err = _validate(data)
        if err:
            flash(err, "error")
            return render_template(
                "patient_form.html", patient=data, mode="create", next_id=next_patient_id()
            )

        patient = Patient(
            id=next_patient_id(),
            nama=data["nama"],
            umur=int(data["umur"]) if data["umur"] else None,
            jenis_kelamin=data["jenis_kelamin"] or None,
            kontak=data["kontak"] or None,
            gejala=data["gejala"] or None,
            riwayat_medis=data["riwayat_medis"] or None,
        )
        db.session.add(patient)
        db.session.commit()
        flash(f"Pasien {patient.id} — {patient.nama} berhasil ditambahkan.", "success")
        return redirect(url_for("patients.list_patients"))

    # GET
    return render_template(
        "patient_form.html", patient=None, mode="create", next_id=next_patient_id()
    )


@patients_bp.route("/patients/<patient_id>/edit", methods=["GET", "POST"])
@login_required
def edit_patient(patient_id):
    patient = db.session.get(Patient, patient_id)
    if patient is None:
        flash("Pasien tidak ditemukan.", "error")
        return redirect(url_for("patients.list_patients"))

    if request.method == "POST":
        data = _read_form()
        err = _validate(data)
        if err:
            flash(err, "error")
            # pertahankan input user untuk diperbaiki
            data["id"] = patient.id
            return render_template("patient_form.html", patient=data, mode="edit")

        patient.nama = data["nama"]
        patient.umur = int(data["umur"]) if data["umur"] else None
        patient.jenis_kelamin = data["jenis_kelamin"] or None
        patient.kontak = data["kontak"] or None
        patient.gejala = data["gejala"] or None
        patient.riwayat_medis = data["riwayat_medis"] or None
        db.session.commit()
        flash(f"Data pasien {patient.id} diperbarui.", "success")
        return redirect(url_for("patients.list_patients"))

    # GET
    return render_template("patient_form.html", patient=patient, mode="edit")


@patients_bp.route("/patients/<patient_id>/delete", methods=["POST"])
@login_required
def delete_patient(patient_id):
    patient = db.session.get(Patient, patient_id)
    if patient is None:
        flash("Pasien tidak ditemukan.", "error")
        return redirect(url_for("patients.list_patients"))

    # Hapus scan terkait dulu agar tidak melanggar foreign key
    for scan in list(patient.scans):
        db.session.delete(scan)
    db.session.delete(patient)
    db.session.commit()
    flash(f"Pasien {patient_id} beserta data scan-nya telah dihapus.", "success")
    return redirect(url_for("patients.list_patients"))
