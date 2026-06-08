"""
report.py — Pembuat laporan PDF resmi (Fase 4) memakai ReportLab.

build_report_pdf(scan, root_path) -> BytesIO berisi PDF siap diunduh.
Tata letak: header klinik, identitas pasien, ringkasan hasil AI, grid gambar
(original + 3 Grad-CAM), catatan dokter, dan disclaimer.
"""
import os
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
)

from app.services.ai_engine import MODEL_NAMES

# Palet selaras dengan UI web (teal klinis).
_TEAL = colors.HexColor("#0d9488")
_TEAL_DARK = colors.HexColor("#115e59")
_SLATE = colors.HexColor("#475569")
_SLATE_LIGHT = colors.HexColor("#f1f5f9")
_ROSE = colors.HexColor("#e11d48")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Brand", fontName="Helvetica-Bold", fontSize=18,
                          textColor=_TEAL_DARK, leading=20))
    ss.add(ParagraphStyle("BrandSub", fontName="Helvetica", fontSize=8,
                          textColor=_SLATE, leading=10))
    ss.add(ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=11,
                          textColor=_TEAL_DARK, spaceBefore=10, spaceAfter=4))
    ss.add(ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5,
                          textColor=colors.HexColor("#1e293b"), leading=14))
    ss.add(ParagraphStyle("Small", fontName="Helvetica", fontSize=7.5,
                          textColor=_SLATE, leading=10))
    ss.add(ParagraphStyle("Caption", fontName="Helvetica", fontSize=7.5,
                          textColor=_SLATE, alignment=TA_CENTER, leading=9))
    return ss


def _abs(root_path: str, rel: str | None) -> str | None:
    """Ubah path relatif-static (mis. 'uploads/x.png') jadi absolut yang ada."""
    if not rel:
        return None
    p = os.path.join(root_path, "static", rel.replace("/", os.sep))
    return p if os.path.exists(p) else None


def build_report_pdf(scan, root_path: str) -> BytesIO:
    ss = _styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"Laporan FindingPneumo {scan.patient_id}",
    )
    story = []

    # ── Header ──────────────────────────────────────────────────────────
    header = Table(
        [[
            Paragraph("FindingPneumo", ss["Brand"]),
            Paragraph(
                f"Laporan Skrining Pneumonia (AI)<br/>"
                f"No. Scan: <b>#{scan.id}</b><br/>"
                f"Tanggal: {scan.tanggal:%d %B %Y, %H:%M}" if scan.tanggal else "",
                ss["BrandSub"]),
        ]],
        colWidths=[90 * mm, 84 * mm],
    )
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, -1), 1.2, _TEAL),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [header, Spacer(1, 8)]

    # ── Identitas pasien ────────────────────────────────────────────────
    p = scan.patient
    doctor = scan.doctor
    gender = {"L": "Laki-laki", "P": "Perempuan"}.get(
        p.jenis_kelamin if p else None, "—")
    info = [
        ["ID Pasien", p.id if p else "—", "Nama", p.nama if p else "—"],
        ["Umur", str(p.umur) if p and p.umur is not None else "—", "Jenis Kelamin", gender],
        ["Kontak", (p.kontak or "—") if p else "—", "Pemeriksa",
         f"{doctor.nama} ({doctor.role})" if doctor else "—"],
    ]
    info_tbl = Table(info, colWidths=[26 * mm, 61 * mm, 26 * mm, 61 * mm])
    info_tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
        ("FONT", (2, 0), (2, -1), "Helvetica-Bold", 9),
        ("TEXTCOLOR", (0, 0), (0, -1), _SLATE),
        ("TEXTCOLOR", (2, 0), (2, -1), _SLATE),
        ("BACKGROUND", (0, 0), (-1, -1), _SLATE_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [Paragraph("Identitas Pasien", ss["H2"]), info_tbl, Spacer(1, 4)]

    if p and (p.gejala or p.riwayat_medis):
        story.append(Paragraph(
            f"<b>Gejala:</b> {p.gejala or '—'}<br/><b>Riwayat medis:</b> {p.riwayat_medis or '—'}",
            ss["Body"]))

    # ── Ringkasan hasil AI ──────────────────────────────────────────────
    is_pneumonia = (scan.hasil_prediksi or "").upper() == "PNEUMONIA"
    label_color = _ROSE if is_pneumonia else _TEAL
    conf_txt = f"{scan.confidence_score * 100:.1f}%" if scan.confidence_score is not None else "—"
    result_tbl = Table(
        [[
            Paragraph("HASIL PREDIKSI", ss["Small"]),
            Paragraph("TINGKAT KEYAKINAN", ss["Small"]),
        ], [
            Paragraph(f"<b>{scan.hasil_prediksi or '—'}</b>",
                      ParagraphStyle("R", parent=ss["Body"], fontSize=15,
                                     textColor=label_color, fontName="Helvetica-Bold")),
            Paragraph(f"<b>{conf_txt}</b>",
                      ParagraphStyle("C", parent=ss["Body"], fontSize=15,
                                     textColor=_TEAL_DARK, fontName="Helvetica-Bold")),
        ]],
        colWidths=[87 * mm, 87 * mm],
    )
    result_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, label_color),
        ("LINEBEFORE", (1, 0), (1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story += [Paragraph("Ringkasan Analisis AI (Ensemble 3 Model)", ss["H2"]),
              result_tbl, Spacer(1, 6)]

    # ── Grid gambar: original + 3 Grad-CAM ──────────────────────────────
    def _img_cell(rel_path, caption):
        abs_p = _abs(root_path, rel_path)
        if abs_p:
            img = RLImage(abs_p, width=40 * mm, height=40 * mm)
        else:
            img = Paragraph("(gambar tidak tersedia)", ss["Caption"])
        return [img, Paragraph(caption, ss["Caption"])]

    cells = [
        _img_cell(scan.file_path_original, "X-Ray Original"),
        _img_cell(scan.file_path_heatmap_1, f"Grad-CAM · {MODEL_NAMES[0]}"),
        _img_cell(scan.file_path_heatmap_2, f"Grad-CAM · {MODEL_NAMES[1]}"),
        _img_cell(scan.file_path_heatmap_3, f"Grad-CAM · {MODEL_NAMES[2]}"),
    ]
    # 4 kolom dalam 1 baris (gambar di atas, caption di bawah → 2 baris tabel).
    img_row = [c[0] for c in cells]
    cap_row = [c[1] for c in cells]
    grid = Table([img_row, cap_row], colWidths=[43 * mm] * 4)
    grid.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
    ]))
    story += [Paragraph("Peta Aktivasi (Grad-CAM)", ss["H2"]), grid, Spacer(1, 4)]

    # ── Catatan dokter ──────────────────────────────────────────────────
    notes = (scan.doctor_notes or "").strip() or "—"
    note_tbl = Table([[Paragraph(notes.replace("\n", "<br/>"), ss["Body"])]],
                     colWidths=[174 * mm])
    note_tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffdf5")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [Paragraph("Catatan & Interpretasi Dokter", ss["H2"]), note_tbl, Spacer(1, 10)]

    # ── Tanda tangan & disclaimer ───────────────────────────────────────
    sign = Table(
        [[Paragraph(
            f"Diperiksa oleh,<br/><br/><br/>"
            f"<b>{doctor.nama if doctor else '—'}</b><br/>"
            f"{doctor.role if doctor else ''}", ss["Body"])]],
        colWidths=[70 * mm], hAlign="RIGHT",
    )
    story += [sign, Spacer(1, 8)]

    story.append(Paragraph(
        "<b>Disclaimer:</b> Laporan ini dihasilkan oleh alat bantu skrining berbasis AI "
        "(FindingPneumo) dan bersifat pendukung. Hasil tidak menggantikan pemeriksaan, "
        "diagnosis, maupun keputusan klinis dokter. Korelasikan dengan kondisi klinis pasien.",
        ss["Small"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buf.seek(0)
    return buf


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(_SLATE)
    canvas.drawString(18 * mm, 10 * mm,
                      f"FindingPneumo · Dicetak {datetime.now():%d %b %Y %H:%M}")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Halaman {doc.page}")
    canvas.restoreState()
