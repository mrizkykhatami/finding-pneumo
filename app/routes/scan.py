"""
Blueprint pemindaian (Fase 3 & 4):
  - Form upload X-ray (membawa patient_id dari halaman pasien).
  - Jalankan analisis AI (ensemble 3 model via ai_engine) → simpan 3 heatmap.
  - Halaman hasil: prediksi, confidence, 3 heatmap, catatan dokter.
  - Generate laporan PDF (Fase 4).
"""
import os
import uuid
from datetime import datetime

from flask import (
    Blueprint, render_template, redirect, url_for, request, flash,
    current_app, send_file, abort,
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from app.extensions import db
from app.models import Patient, Scan
from app.services.ai_engine import analyze_scan, MODEL_NAMES

scan_bp = Blueprint("scan", __name__)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_BYTES = 12 * 1024 * 1024  # 12 MB


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _upload_dir() -> str:
    """Folder absolut static/uploads (dibuat jika belum ada)."""
    d = os.path.join(current_app.root_path, "static", "uploads")
    os.makedirs(d, exist_ok=True)
    return d


@scan_bp.route("/scan/new")
@login_required
def new_scan():
    """Tampilkan form upload. patient_id opsional (dari tombol New Scan)."""
    patient_id = request.args.get("patient_id", "")
    selected = db.session.get(Patient, patient_id) if patient_id else None
    # Daftar pasien untuk dropdown (kalau user buka langsung tanpa pilih pasien).
    patients = Patient.query.order_by(Patient.nama).all()
    return render_template(
        "scan_new.html", selected=selected, patients=patients, model_names=MODEL_NAMES
    )


@scan_bp.route("/scan/analyze", methods=["POST"])
@login_required
def analyze():
    patient_id = request.form.get("patient_id", "").strip()
    patient = db.session.get(Patient, patient_id) if patient_id else None
    if patient is None:
        flash("Pilih pasien yang valid terlebih dahulu.", "error")
        return redirect(url_for("scan.new_scan"))

    file = request.files.get("xray")
    if file is None or file.filename == "":
        flash("Belum ada gambar X-ray yang dipilih.", "error")
        return redirect(url_for("scan.new_scan", patient_id=patient_id))

    if not _allowed(file.filename):
        flash("Format tidak didukung. Gunakan PNG / JPG / JPEG / BMP / WEBP.", "error")
        return redirect(url_for("scan.new_scan", patient_id=patient_id))

    # Simpan file original dengan nama unik.
    upload_dir = _upload_dir()
    ext = file.filename.rsplit(".", 1)[1].lower()
    basename = f"{patient_id}_{datetime.utcnow():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"
    original_name = f"{basename}.{ext}"
    original_abs = os.path.join(upload_dir, original_name)
    file.save(original_abs)

    # Validasi ukuran setelah tersimpan (hindari baca seluruh stream di memori).
    if os.path.getsize(original_abs) > MAX_BYTES:
        os.remove(original_abs)
        flash("Ukuran file melebihi 12 MB.", "error")
        return redirect(url_for("scan.new_scan", patient_id=patient_id))

    # Jalankan analisis AI (ensemble 3 model).
    try:
        result = analyze_scan(original_abs, upload_dir, basename)
    except Exception as exc:  # noqa: BLE001 — tampilkan error apa pun ke dokter dengan ramah
        current_app.logger.exception("Analisis gagal")
        flash(f"Analisis gagal: {exc}", "error")
        return redirect(url_for("scan.new_scan", patient_id=patient_id))

    # Simpan record scan.
    scan = Scan(
        patient_id=patient.id,
        doctor_id=current_user.id,
        hasil_prediksi=result["label"],
        confidence_score=result["confidence"],
        file_path_original=f"uploads/{original_name}",
        file_path_heatmap_1=result["heatmaps"][0],
        file_path_heatmap_2=result["heatmaps"][1],
        file_path_heatmap_3=result["heatmaps"][2],
        file_path_heatmap_consensus=result.get("consensus"),
    )
    db.session.add(scan)
    db.session.commit()

    flash("Analisis selesai.", "success")
    return redirect(url_for("scan.view_scan", scan_id=scan.id))


@scan_bp.route("/scan/<int:scan_id>")
@login_required
def view_scan(scan_id):
    scan = db.session.get(Scan, scan_id)
    if scan is None:
        abort(404)
    heatmaps = list(zip(MODEL_NAMES, [
        scan.file_path_heatmap_1, scan.file_path_heatmap_2, scan.file_path_heatmap_3,
    ]))
    return render_template("scan_result.html", scan=scan, heatmaps=heatmaps)


@scan_bp.route("/scan/<int:scan_id>/notes", methods=["POST"])
@login_required
def save_notes(scan_id):
    scan = db.session.get(Scan, scan_id)
    if scan is None:
        abort(404)
    scan.doctor_notes = request.form.get("doctor_notes", "").strip() or None
    db.session.commit()
    flash("Catatan dokter tersimpan.", "success")
    return redirect(url_for("scan.view_scan", scan_id=scan.id))


@scan_bp.route("/scan/<int:scan_id>/report.pdf")
@login_required
def report_pdf(scan_id):
    scan = db.session.get(Scan, scan_id)
    if scan is None:
        abort(404)
    from app.services.report import build_report_pdf  # impor lokal agar reportlab hanya dimuat saat perlu

    pdf_buffer = build_report_pdf(scan, current_app.root_path)
    filename = f"FindingPneumo_{scan.patient_id}_scan{scan.id}.pdf"
    return send_file(
        pdf_buffer, mimetype="application/pdf",
        as_attachment=True, download_name=filename,
    )
