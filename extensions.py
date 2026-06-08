"""
Ekstensi Flask diinisialisasi terpisah di sini supaya tidak terjadi
circular import antara app.py, models.py, dan auth.py.
Pola standar: objek dibuat kosong di sini, lalu di-bind ke app di app.py.
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()
