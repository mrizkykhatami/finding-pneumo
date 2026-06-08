"""
Entry point aplikasi FindingPneumo.

Jalankan dari folder ini:
    python run.py
Lalu buka: http://127.0.0.1:5000
"""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    # debug dari environment (default aktif untuk dev lokal; set FLASK_DEBUG=0 di produksi)
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    # use_reloader=False: cegah server restart saat heatmap ditulis ke static/uploads
    # (penyebab "connection reset" saat scan) + hemat memori (1 proses, bukan 2).
    # threaded=False: hindari beberapa inferensi TensorFlow berbarengan.
    app.run(debug=debug, use_reloader=False, threaded=False,
            host="127.0.0.1", port=5000)
