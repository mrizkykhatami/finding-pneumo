"""
Application factory FindingPneumo (deteksi pneumonia, alat bantu skrining).

Struktur:
    app/__init__.py     -> create_app() + route utama (index, dashboard)
    app/extensions.py   -> db, login_manager
    app/models.py       -> skema database
    app/routes/         -> blueprint halaman (auth, patients, scan)
    app/services/       -> logika non-web (ai_engine, report)

Dijalankan lewat run.py di root proyek:  python run.py
"""
import os
from datetime import date

from flask import Flask, render_template, redirect, url_for
from flask_login import login_required, current_user
from sqlalchemy import func

from app.extensions import db, login_manager
from app.models import User, Patient, Scan  # untuk user_loader, metrik & create_all


def create_app() -> Flask:
    app = Flask(__name__)

    # --- Konfigurasi ---
    # SECRET_KEY untuk menandatangani session-cookie. Di produksi ambil dari env;
    # untuk tugas lokal, fallback string ini sudah cukup.
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-key-ganti-untuk-produksi")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///pneumoscan.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # --- Init ekstensi ---
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"  # redirect ke sini kalau akses halaman terkunci
    login_manager.login_message = "Silakan login terlebih dahulu."
    login_manager.login_message_category = "error"

    # --- Blueprint ---
    from app.routes.auth import auth_bp
    from app.routes.patients import patients_bp
    from app.routes.scan import scan_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(patients_bp)
    app.register_blueprint(scan_bp)

    # --- Route utama ---
    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        return redirect(url_for("auth.login"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        # --- Metrik real (Fase 2) ---
        total_patients = db.session.query(func.count(Patient.id)).scalar() or 0

        today = date.today()
        scans_today = (
            db.session.query(func.count(Scan.id))
            .filter(func.date(Scan.tanggal) == today.isoformat())
            .scalar()
            or 0
        )

        # Pneumonia rate: proporsi scan positif dari scan yang sudah punya hasil prediksi
        total_predicted = (
            db.session.query(func.count(Scan.id))
            .filter(Scan.hasil_prediksi.isnot(None))
            .scalar()
            or 0
        )
        pneumonia_count = (
            db.session.query(func.count(Scan.id))
            .filter(func.upper(Scan.hasil_prediksi) == "PNEUMONIA")
            .scalar()
            or 0
        )
        pneumonia_rate = (
            f"{(pneumonia_count / total_predicted * 100):.0f}%" if total_predicted else "—"
        )

        # Reports generated: scan yang sudah punya catatan dokter (proxy laporan PDF, Fase 4)
        reports_generated = (
            db.session.query(func.count(Scan.id))
            .filter(Scan.doctor_notes.isnot(None), Scan.doctor_notes != "")
            .scalar()
            or 0
        )

        recent_scans = (
            Scan.query.order_by(Scan.tanggal.desc()).limit(5).all()
        )

        metrics = {
            "total_patients": total_patients,
            "scans_today": scans_today,
            "pneumonia_rate": pneumonia_rate,
            "reports_generated": reports_generated,
        }
        return render_template(
            "dashboard.html", metrics=metrics, recent_scans=recent_scans
        )

    # --- Buat tabel DB saat pertama kali jalan ---
    with app.app_context():
        db.create_all()

    return app


@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))
