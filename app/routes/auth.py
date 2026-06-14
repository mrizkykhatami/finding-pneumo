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
            flash("All fields are required.", "error")
            return render_template("register.html")

        if role not in VALID_ROLES:
            flash("Invalid role.", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")

        # Check if email is already registered
        if User.query.filter_by(email=email).first():
            flash("Email is already registered. Please login.", "error")
            return render_template("register.html")

        # Simpan user baru (password di-hash di dalam set_password)
        user = User(nama=nama, email=email, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash("Registration successful. Please login.", "success")
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

        # Error message is intentionally generic (do not reveal whether the email exists)
        if user is None or not user.check_password(password):
            flash("Email or password is incorrect.", "error")
            return render_template("login.html")

        login_user(user)
        flash(f"Welcome, {user.nama}.", "success")

        # Redirect ke halaman tujuan jika ada (?next=...), kalau tidak ke dashboard
        next_page = request.args.get("next")
        return redirect(next_page or url_for("dashboard"))

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("auth.login"))
