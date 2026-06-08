"""
Blueprint autentikasi: register, login, logout.
Memakai session-cookie via Flask-Login (sesuai spesifikasi).
"""
from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user

from app.extensions import db
from app.models import User

auth_bp = Blueprint("auth", __name__)

VALID_ROLES = {"Radiolog", "Dokter Umum"}


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    # Kalau sudah login, tidak perlu register lagi
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        nama = request.form.get("nama", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "Dokter Umum")

        # Validasi sederhana
        if not nama or not email or not password:
            flash("Semua field wajib diisi.", "error")
            return render_template("register.html")

        if role not in VALID_ROLES:
            flash("Role tidak valid.", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password minimal 6 karakter.", "error")
            return render_template("register.html")

        # Cek email sudah terdaftar
        if User.query.filter_by(email=email).first():
            flash("Email sudah terdaftar. Silakan login.", "error")
            return render_template("register.html")

        # Simpan user baru (password di-hash di dalam set_password)
        user = User(nama=nama, email=email, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash("Registrasi berhasil. Silakan login.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        # Pesan error sengaja generik (tidak membocorkan email mana yang ada/tidak)
        if user is None or not user.check_password(password):
            flash("Email atau password salah.", "error")
            return render_template("login.html")

        login_user(user)
        flash(f"Selamat datang, {user.nama}.", "success")

        # Redirect ke halaman tujuan jika ada (?next=...), kalau tidak ke dashboard
        next_page = request.args.get("next")
        return redirect(next_page or url_for("dashboard"))

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Anda telah logout.", "success")
    return redirect(url_for("auth.login"))
