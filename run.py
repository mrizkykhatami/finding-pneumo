"""
Entry point aplikasi PneumoScan.

Jalankan dari folder ini:
    python run.py
Lalu buka: http://127.0.0.1:5000
"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    # debug=True hanya untuk pengembangan lokal
    app.run(debug=True, host="127.0.0.1", port=5000)
