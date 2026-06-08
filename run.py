"""
Entry point aplikasi PneumoScan.

Jalankan dari folder ini:
    python run.py
Lalu buka: http://127.0.0.1:5000
"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    # use_reloader=False: cegah server restart saat heatmap ditulis ke static/uploads
    # (penyebab "connection reset" saat scan) + hemat memori (1 proses, bukan 2).
    # threaded=False: hindari beberapa inferensi TensorFlow berbarengan.
    app.run(debug=True, use_reloader=False, threaded=False,
            host="127.0.0.1", port=5000)
