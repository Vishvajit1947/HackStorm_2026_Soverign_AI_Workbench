"""generate_mrpl_corpus.py — Build the complete MRPL demo corpus for Phase 6.

Creates:
MRPL_DEMO_CORPUS/
├── native_pdfs/
│   ├── SOP_01.pdf
│   ├── SOP_02.pdf
│   ├── Technical_Procedure.pdf
│   └── P_and_ID_Guide.pdf
├── scanned_reports/
│   ├── inspection_report_01.pdf
│   ├── inspection_report_02.pdf
│   └── hydrotest_report.pdf
├── pids/
│   ├── PID_01.png
│   └── PID_02.png
├── handwritten/
│   └── handwritten_annotation.pdf
└── approval_notes/
    └── approval_note_reference.pdf
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf
from matplotlib import patches
from PIL import Image, ImageDraw, ImageEnhance
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "MRPL_DEMO_CORPUS"

# Set up subdirectories
DIR_NATIVE = CORPUS_DIR / "native_pdfs"
DIR_SCANNED = CORPUS_DIR / "scanned_reports"
DIR_PIDS = CORPUS_DIR / "pids"
DIR_HANDWRITTEN = CORPUS_DIR / "handwritten"
DIR_APPROVAL = CORPUS_DIR / "approval_notes"

for d in (DIR_NATIVE, DIR_SCANNED, DIR_PIDS, DIR_HANDWRITTEN, DIR_APPROVAL):
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Native Technical PDFs (ReportLab)
# ══════════════════════════════════════════════════════════════════════════════

def create_sop_01():
    target = DIR_NATIVE / "SOP_01.pdf"
    doc = SimpleDocTemplate(
        str(target), pagesize=A4,
        leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'HeaderTitle',
        parent=styles['Heading1'],
        fontSize=13, leading=16, textColor=colors.HexColor("#0f2a4a"),
        alignment=1, fontName="Helvetica-Bold"
    )
    subtitle_style = ParagraphStyle(
        'HeaderSubtitle',
        parent=styles['Normal'],
        fontSize=9, leading=12, textColor=colors.HexColor("#333333"),
        alignment=1, fontName="Helvetica"
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=10, leading=14, textColor=colors.HexColor("#0f2a4a"),
        fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=8.5, leading=12, textColor=colors.HexColor("#222222"),
        fontName="Helvetica"
    )

    story = []

    # Header Box
    header_data = [
        [
            Paragraph("<b>MANGALORE REFINERY AND PETROCHEMICALS LIMITED (MRPL)</b><br/>"
                      "OPERATIONS DIVISION — CRUDE DISTILLATION UNIT (CDU-II)", title_style),
        ],
        [
            Paragraph("STANDARD OPERATING PROCEDURE: HYDROCARBON FEED PUMP STARTUP & ISOLATION<br/>"
                      "DOC NO: <b>MRPL-SOP-CDU-0101</b> | REV: 04 | EFFECTIVE: 15-JAN-2026 | CLASSIFICATION: CONFIDENTIAL", subtitle_style),
        ]
    ]
    t_hdr = Table(header_data, colWidths=[515])
    t_hdr.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#0f2a4a")),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f4f8")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_hdr)
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. PURPOSE & APPLICABILITY", h2_style))
    story.append(Paragraph(
        "This standard operating procedure mandates safety requirements, alignment steps, and operational guidelines "
        "for the startup, continuous operation, and emergency isolation of Crude Booster Pumps <b>P-101A/B</b> serving "
        "atmospheric distillation train CDU-II at MRPL Mangalore complex. Applicable to all plant operators and shift engineers.",
        body_style
    ))

    story.append(Paragraph("2. SAFETY INTERLOCKS & OPERATIONAL THRESHOLDS", h2_style))
    story.append(Paragraph(
        "Prior to motor energization, field operations shall confirm that the DCS permissive interlock logic is SATISFIED "
        "and trip thresholds are active in accordance with Safety Integrity Level (SIL-2) allocation:",
        body_style
    ))

    interlock_data = [
        ["Tag", "Parameter", "Trip Setpoint", "Alarm Setting", "Interlock Logic"],
        ["PSLL-101", "Suction Pressure (P-101A)", "1.80 bar (g)", "2.05 bar (g)", "Trip Motor instantly; prevent dry-run"],
        ["PSHH-101", "Discharge Pressure (P-101A)", "18.50 bar (g)", "17.20 bar (g)", "Open bypass FCV-101 to 100%"],
        ["TSH-101", "DE Bearing Temperature", "85.0 °C", "75.0 °C", "Alarm annunciator CDU console"],
        ["TSHH-101", "NDE Bearing Temperature", "95.0 °C", "85.0 °C", "Trip Motor (2 out of 3 voting)"],
        ["PDSL-101", "Suction Basket Strainer ΔP", "0.45 bar", "0.30 bar", "Switch over to standby strainer"],
        ["FSLL-101", "Minimum Flow Bypass", "35.0 m³/h", "42.0 m³/h", "Open FCV-101; trip pump after 30s delay"]
    ]
    t_int = Table(interlock_data, colWidths=[65, 130, 85, 85, 150])
    t_int.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f2a4a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_int)

    story.append(Paragraph("3. PRE-STARTUP VERIFICATION CHECKLIST", h2_style))
    story.append(Paragraph(
        "• <b>Mechanical Seal Flush</b>: API Plan 53B barrier fluid reservoir pressure maintained at 3.5 bar (g), minimum 1.5 bar above seal chamber pressure.<br/>"
        "• <b>Lube Oil System</b>: Check constant level oilers at 50% mark with ISO VG 46 turbine oil. Cooling water flow confirmed through bearing housings.<br/>"
        "• <b>De-isolation Check</b>: Verify suction isolation valve MOV-101A is 100% locked open (LO). Confirm blind spectacle position is OPEN.<br/>"
        "• <b>Priming and Venting</b>: Vent pump casing high point valve V-101-V until bubble-free hydrocarbon liquid appears into closed drain header.",
        body_style
    ))

    story.append(Paragraph("4. STARTUP EXECUTION SEQUENCE", h2_style))
    story.append(Paragraph(
        "1. Request electrical clearance (Permit E-04) and ensure motor breaker racked IN.<br/>"
        "2. Ensure minimum recirculation bypass valve <b>FCV-101</b> is 100% open to feed drum V-101.<br/>"
        "3. Crack open discharge manual throttling valve MOV-101B to 10% position.<br/>"
        "4. Depress local START pushbutton while monitoring motor inrush current (must decay from ~380A to normal 68A within 6.5s).<br/>"
        "5. Observe discharge pressure gauge PI-1012 rise smoothly to shutoff head (~15.4 bar).<br/>"
        "6. Slowly ramp discharge valve MOV-101B over 90 seconds to target flow of 145 m³/h while DCS places FCV-101 on automatic flow control.",
        body_style
    ))

    story.append(Paragraph("5. EMERGENCY SHUTDOWN & ISOLATION", h2_style))
    story.append(Paragraph(
        "In the event of uncontrolled mechanical seal leakage, vapor lock, or fire alarm in battery limit: "
        "Press ESD-101 field button immediately. This executes: (a) Motor trip, (b) Immediate fail-close of suction valve MOV-101A and discharge valve MOV-101B, "
        "(c) Nitrogen purge sequence injection into pump stuffing box.",
        body_style
    ))

    doc.build(story)
    print(f"Created {target.name}")


def create_sop_02():
    target = DIR_NATIVE / "SOP_02.pdf"
    doc = SimpleDocTemplate(
        str(target), pagesize=A4,
        leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=12, leading=15, textColor=colors.HexColor("#8a1c14"), alignment=1, fontName="Helvetica-Bold")
    subtitle_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=colors.HexColor("#333333"), alignment=1, fontName="Helvetica")
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=9.5, leading=13, textColor=colors.HexColor("#8a1c14"), fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3)
    body_style = ParagraphStyle('B', parent=styles['Normal'], fontSize=8, leading=11, textColor=colors.HexColor("#222222"), fontName="Helvetica")

    story = []
    header_data = [
        [Paragraph("<b>MANGALORE REFINERY AND PETROCHEMICALS LIMITED (MRPL)</b><br/>HEALTH, SAFETY & ENVIRONMENT (HSE) DEPARTMENT", title_style)],
        [Paragraph("CORPORATE SAFETY PROCEDURE: HOT WORK, GAS TESTING & CONFINED SPACE ENTRY<br/>"
                   "DOC NO: <b>MRPL-SOP-HSE-0204</b> | REV: 06 | MANDATORY COMPLIANCE", subtitle_style)]
    ]
    t_hdr = Table(header_data, colWidths=[515])
    t_hdr.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#8a1c14")),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#fdf2f2")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_hdr)
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. PERMIT TO WORK (PTW) MANDATORY PROTOCOL", h2_style))
    story.append(Paragraph(
        "No hot work (welding, cutting, grinding, uncertified electrical tools) or confined space entry is permissible "
        "within 15 meters of hydrocarbon process equipment without a valid Class-A Hot Work Permit (Form HSE-F01) "
        "endorsed by the Shift In-charge (Operations) and the Area Safety Officer.",
        body_style
    ))

    story.append(Paragraph("2. ATMOSPHERIC TESTING LIMITS (GAS TESTING CRITERIA)", h2_style))
    gas_data = [
        ["Gas / Hazard", "Permissible Limit (Cold)", "Permissible Limit (Hot Work)", "Test Frequency"],
        ["Combustible Gas (LEL)", "< 1.0% LEL", "0.0% LEL (Strictly Nil)", "Continuous during hot work"],
        ["Oxygen (O₂)", "19.5% to 23.5%", "19.5% to 23.5%", "Prior to entry & every 2 hours"],
        ["Hydrogen Sulfide (H₂S)", "< 5.0 ppm", "< 5.0 ppm (Nil preferred)", "Continuous multi-gas detector"],
        ["Carbon Monoxide (CO)", "< 25.0 ppm", "< 25.0 ppm", "Prior to entry & post-welding"],
        ["Volatile Organics (VOC)", "< 10.0 ppm", "< 5.0 ppm", "Calibrated PID detector"]
    ]
    t_gas = Table(gas_data, colWidths=[120, 115, 140, 140])
    t_gas.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#8a1c14")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dddddd")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#fffafa")]),
    ]))
    story.append(t_gas)

    story.append(Paragraph("3. POSITIVE PHYSICAL ISOLATION (SPADING & BLINDING)", h2_style))
    story.append(Paragraph(
        "• Double Block and Bleed (DB&B) is acceptable ONLY for short-duration instrument calibration.<br/>"
        "• All line entries, hot work, and vessel entries REQUIRE positive isolation via spectacle blinds or slip spades "
        "rated for full piping class design pressure (ASME B16.48). Blinding list sign-off is mandatory before gas freeing.<br/>"
        "• Tag-out / Lock-out (LOTO): Padlocks with Master Key retention held by Shift Lead.",
        body_style
    ))

    story.append(Paragraph("4. FIRE PREVENTION & SURVEILLANCE", h2_style))
    story.append(Paragraph(
        "• Dedicated trained Fire Watch personnel equipped with 50 kg DCP fire extinguisher and continuous pressurized fire hose (10 bar).<br/>"
        "• Spark containment habitat / fire-resistant certified blankets (FM Global approved) covering all nearby sewers and drain openings within 15 meters.<br/>"
        "• Welding ground cable must be connected directly to the work piece, never to plant structural steel or pipe supports.",
        body_style
    ))

    doc.build(story)
    print(f"Created {target.name}")


def create_technical_procedure():
    target = DIR_NATIVE / "Technical_Procedure.pdf"
    doc = SimpleDocTemplate(
        str(target), pagesize=A4,
        leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=12, leading=15, textColor=colors.HexColor("#1b4d3e"), alignment=1, fontName="Helvetica-Bold")
    subtitle_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=colors.HexColor("#333333"), alignment=1, fontName="Helvetica")
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=9.5, leading=13, textColor=colors.HexColor("#1b4d3e"), fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3)
    body_style = ParagraphStyle('B', parent=styles['Normal'], fontSize=8, leading=11.5, textColor=colors.HexColor("#222222"), fontName="Helvetica")
    code_style = ParagraphStyle('C', parent=body_style, fontName="Courier-Bold", fontSize=8.5, leading=12, textColor=colors.HexColor("#0f2a4a"))

    story = []
    header_data = [
        [Paragraph("<b>MANGALORE REFINERY AND PETROCHEMICALS LIMITED (MRPL)</b><br/>TECHNICAL SERVICES DIVISION — ROTATING EQUIPMENT ENGINEERING", title_style)],
        [Paragraph("ENGINEERING TECHNICAL STANDARD: PUMP NPSH & CAVITATION MARGIN CRITERIA<br/>"
                   "DOC NO: <b>MRPL-ENG-STD-402</b> | REV: 03 | REFERENCE: API 610 / ISO 13709", subtitle_style)]
    ]
    t_hdr = Table(header_data, colWidths=[515])
    t_hdr.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#1b4d3e")),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f7f4")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_hdr)
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. GOVERNING NPSH FORMULATION", h2_style))
    story.append(Paragraph(
        "Net Positive Suction Head Available (NPSHa) represents the total suction head available at the pump suction centerline "
        "over and above the vapor pressure of the pumped liquid at operating temperature. The mandatory calculation is:",
        body_style
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>NPSHa = h_sp + h_s - h_f - h_vp</b>",
        code_style
    ))
    story.append(Paragraph(
        "Where:<br/>"
        "• <b>h_sp</b> = Absolute pressure on the liquid surface in suction vessel = (P_suct_abs) / (ρ · g) [meters]<br/>"
        "• <b>h_s</b> = Static liquid level above pump impeller centerline (positive if above, negative if below) [meters]<br/>"
        "• <b>h_f</b> = Total piping friction and dynamic head losses (suction pipe, strainer, valves, fittings) [meters]<br/>"
        "• <b>h_vp</b> = Vapor pressure of hydrocarbon mixture at maximum design pumping temperature = (P_vap) / (ρ · g) [meters]",
        body_style
    ))

    story.append(Paragraph("2. MANDATORY CAVITATION SAFETY MARGIN RULE", h2_style))
    story.append(Paragraph(
        "Per MRPL Engineering Standard and API 610 §6.1.8:<br/>"
        "<b>NPSHa ≥ NPSHr + 1.0 meter</b> (or <b>1.20 × NPSHr</b>, whichever margin is GREATER).<br/>"
        "For hydrocarbons with specific gravity < 0.80 operating near boiling point (bubble point), an additional 0.5 m vapor suppression contingency is enforced.",
        body_style
    ))

    story.append(Paragraph("3. VERIFIED DESIGN CALCULATION: FEED PUMP P-101A (NAPHTHA STABILIZER)", h2_style))
    calc_data = [
        ["Parameter", "Design Value", "Converted Head", "Notes"],
        ["Operating Temperature (T)", "65.0 °C (338.15 K)", "—", "Maximum summer ambient scenario"],
        ["Liquid Density (ρ)", "740.0 kg/m³", "—", "Light Naphtha fraction"],
        ["Suction Vessel Pressure (P_suct)", "1.80 bar(g) = 2.813 bar(a)", "38.74 m", "h_sp = (2.813 × 10⁵) / (740 × 9.81)"],
        ["Static Liquid Height (h_s)", "+4.20 meters", "+4.20 m", "Minimum operational low level (LLLL)"],
        ["Suction Piping Friction (h_f)", "0.85 bar loss equivalent", "-1.17 m", "Includes 100-mesh suction strainer"],
        ["Hydrocarbon Vapor Pressure (P_vp)", "0.420 bar(a)", "-5.78 m", "h_vp = (0.420 × 10⁵) / (740 × 9.81)"],
        ["<b>Calculated NPSHa</b>", "<b>—</b>", "<b>35.99 m</b>", "<b>NPSHa = 38.74 + 4.20 - 1.17 - 5.78</b>"],
        ["Vendor Certified NPSHr", "3.20 meters", "3.20 m", "At rated flow 145 m³/h (1480 RPM)"],
        ["Design Margin (NPSHa - NPSHr)", "—", "<b>+32.79 m</b>", "COMPLIES (Excess exceeds API 610 margin)"]
    ]
    t_calc = Table(calc_data, colWidths=[160, 115, 100, 140])
    t_calc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b4d3e")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f6fcf8")]),
    ]))
    story.append(t_calc)

    doc.build(story)
    print(f"Created {target.name}")


def create_pid_guide():
    target = DIR_NATIVE / "P_and_ID_Guide.pdf"
    doc = SimpleDocTemplate(
        str(target), pagesize=A4,
        leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=12, leading=15, textColor=colors.HexColor("#1f3a60"), alignment=1, fontName="Helvetica-Bold")
    subtitle_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=colors.HexColor("#333333"), alignment=1, fontName="Helvetica")
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=9.5, leading=13, textColor=colors.HexColor("#1f3a60"), fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3)
    body_style = ParagraphStyle('B', parent=styles['Normal'], fontSize=8, leading=11.5, textColor=colors.HexColor("#222222"), fontName="Helvetica")

    story = []
    header_data = [
        [Paragraph("<b>MANGALORE REFINERY AND PETROCHEMICALS LIMITED (MRPL)</b><br/>ENGINEERING DESIGN & DRAFTING DIVISION", title_style)],
        [Paragraph("P&ID DRAFTING GUIDE: INSTRUMENTATION & LINE SYMBOLOGY<br/>"
                   "DOC NO: <b>MRPL-DWG-STD-001</b> | REV: 05 | ANSI/ISA-5.1 STANDARD", subtitle_style)]
    ]
    t_hdr = Table(header_data, colWidths=[515])
    t_hdr.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#1f3a60")),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#eef3f8")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_hdr)
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. INSTRUMENT LETTER CODE IDENTIFICATION (ISA-5.1)", h2_style))
    story.append(Paragraph(
        "Instrument tag names consist of two components: Functional Identification (letters) and Loop Identification (numbers). "
        "First letter identifies the measured process variable; subsequent letters denote the readout or output function:",
        body_style
    ))

    code_table = [
        ["1st Letter (Variable)", "Meaning", "Succeeding Letter", "Meaning", "Example Tag", "Description"],
        ["P", "Pressure", "T", "Transmitter", "PT-101", "Pressure Transmitter (Electronic 4-20mA)"],
        ["F", "Flow Rate", "CV", "Control Valve", "FCV-101", "Pneumatic Flow Control Valve (Air to Open)"],
        ["T", "Temperature", "I", "Indicator (Local)", "TI-1014", "Bimetallic Dial Thermometer"],
        ["L", "Level", "SLL", "Switch Low Low", "LSLL-102", "Emergency Low Level Interlock Sensor"],
        ["I", "Interlock / Logic", "—", "Controller", "I-101", "DCS Logic Interlock Tripping Pump P-101"]
    ]
    t_sym = Table(code_table, colWidths=[90, 80, 85, 75, 75, 110])
    t_sym.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1f3a60")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.2),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dddddd")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t_sym)

    story.append(Paragraph("2. PIPING LINE SPECIFICATION CODING", h2_style))
    story.append(Paragraph(
        "All process piping tags follow standard format: <b>[Size]\"-[Fluid Code]-[Line Number]-[Pipe Spec]-[Insulation]</b><br/>"
        "• <b>Size</b>: Nominal Pipe Size in inches (e.g. 4\", 6\", 2\")<br/>"
        "• <b>Fluid Service</b>: HC = Hydrocarbon, CW = Cooling Water, STM = Steam, N2 = Nitrogen, DR = Drain<br/>"
        "• <b>Piping Class</b>: A1A = ASME Class 150 Carbon Steel, B1A = ASME Class 300, E1A = ASME Class 600<br/>"
        "• <b>Insulation</b>: H = Hot Conservation (Mineral wool), C = Cold Insulation (Polyurethane), P = Personnel Protection",
        body_style
    ))

    doc.build(story)
    print(f"Created {target.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Scanned Reports with Realistic Scan Artifacts (PyMuPDF / Pillow)
# ══════════════════════════════════════════════════════════════════════════════

def add_scan_effects(pil_img: Image.Image) -> Image.Image:
    """Apply realistic scanner noise, slight rotation, subtle paper tint, and contrast."""
    # Convert to RGB
    img = pil_img.convert("RGB")
    
    # Slight rotation (-0.4 to 0.4 degrees)
    angle = random.uniform(-0.35, 0.35)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(248, 246, 240))
    
    # Subtle paper tint / yellowing
    tint = Image.new("RGB", img.size, (250, 248, 242))
    img = Image.blend(img, tint, 0.12)
    
    # Contrast and sharpness tweaks
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.08)
    
    return img


def create_inspection_report_01():
    target = DIR_SCANNED / "inspection_report_01.pdf"
    
    # Render with ReportLab to buffer
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=11, leading=14, textColor=colors.HexColor("#222222"), alignment=1, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=8, leading=11, textColor=colors.HexColor("#444444"), alignment=1, fontName="Helvetica")
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=9, leading=12, textColor=colors.HexColor("#111111"), fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2)
    b = ParagraphStyle('B', parent=styles['Normal'], fontSize=7.5, leading=10, textColor=colors.HexColor("#222222"), fontName="Helvetica")

    story = []
    hdr_data = [
        [Paragraph("<b>MANGALORE REFINERY AND PETROCHEMICALS LIMITED (MRPL)</b><br/>INSPECTION & ASSET INTEGRITY DIVISION — NDT SERVICES", title_style)],
        [Paragraph("ULTRASONIC THICKNESS (UT) GAUGING & CORROSION RATE EVALUATION REPORT<br/>"
                   "REPORT NO: <b>MRPL-NDT-UT-2026-089</b> | DATE OF SURVEY: 28-FEB-2026 | STATUS: <b>ACTION REQUIRED</b>", sub_style)]
    ]
    t = Table(hdr_data, colWidths=[540])
    t.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e8e8e8")),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. ASSET DETAILS & SURVEY CONDITIONS", h2))
    info_data = [
        ["Asset Tag / ID:", "Feed Booster Pump P-101A & Suction Piping 4\"-HC-1012-A1A-H", "Plant Unit:", "CDU-II / NHT Feed Train"],
        ["Material / Spec:", "ASTM A106 Gr. B (Carbon Steel) / Schedule 40", "Operating Temp:", "65 °C / 1.8 bar (g)"],
        ["Nominal Thickness:", "6.02 mm (Schedule 40)", "Min. Required (t_min):", "3.20 mm (per ASME B31.3)"],
        ["NDT Equipment:", "Olympus 38DL-Plus Ultrasonic Gauge (Calib Due: Nov-2026)", "Couplant Used:", "High-Temp Silicone Gel"]
    ]
    t_info = Table(info_data, colWidths=[105, 185, 100, 150])
    t_info.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#999999")),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f0f0f0")),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor("#f0f0f0")),
        ('PADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_info)

    story.append(Paragraph("2. ULTRASONIC THICKNESS MEASUREMENT GRID", h2))
    ut_data = [
        ["Location / Point", "Description / Orientation", "Nominal (mm)", "Previous (mm)", "Current (mm)", "Loss (mm)", "Status"],
        ["P1", "Pump Suction Reducer Top", "6.02", "5.10", "4.85", "-0.25", "ACCEPTABLE"],
        ["P2", "Pump Suction Reducer Bottom (6 o'clock)", "6.02", "4.60", "4.15", "-0.45", "MONITOR"],
        ["P3", "Elbow #1 Extrados (4\"-HC-1012)", "6.02", "4.30", "3.85", "-0.45", "MONITOR"],
        ["P4", "Elbow #1 Intrados", "6.02", "5.30", "5.12", "-0.18", "ACCEPTABLE"],
        ["P5 (CRITICAL)", "Elbow #2 Extrados Impingement Zone", "6.02", "3.80", "3.42", "-0.38", "NEAR RETIREMENT"],
        ["P6", "Pump Volute Casing Inlet Throat", "8.50", "7.10", "6.68", "-0.42", "ACCEPTABLE"],
        ["P7", "Bypass Tie-in T-piece (2\"-HC-1015)", "5.54", "4.20", "3.90", "-0.30", "MONITOR"]
    ]
    t_ut = Table(ut_data, colWidths=[80, 160, 60, 60, 60, 50, 70])
    t_ut.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#333333")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 6.8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#777777")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f8f8")]),
        ('TEXTCOLOR', (6,5), (6,5), colors.red),
        ('FONTNAME', (0,5), (-1,5), 'Helvetica-Bold'),
    ]))
    story.append(t_ut)

    story.append(Paragraph("3. REMAINING LIFE & CORROSION RATE CALCULATION", h2))
    story.append(Paragraph(
        "• <b>Calculated Corrosion Rate (CR) at Point P5</b>: (3.80 mm - 3.42 mm) / 1.35 yrs = <b>0.281 mm/year</b>.<br/>"
        "• <b>Remaining Life (RL)</b>: (t_actual - t_min) / CR = (3.42 mm - 3.20 mm) / 0.281 mm/yr = <b>0.78 years (9.4 months)</b>.<br/>"
        "• <b>ENGINEERING RECOMMENDATION</b>: Spool section containing Elbow #2 MUST be scheduled for replacement during upcoming "
        "CDU-II mini-shutdown (May 2026). Continuous operational monitoring required. Derate hydrotest pressure to 16.5 bar.",
        b
    ))
    story.append(Spacer(1, 8))

    # Signature Block
    sig_data = [
        ["Inspected By: S. V. Shenoy (ASNT Level-II)", "Verified By: R. K. Sharma (Sr. Mgr NDT)", "Approved: Dy. GM (Inspection)"],
        ["[Signed 28/02/2026]", "[Signed 01/03/2026]", "[STAMPED & APPROVED]"]
    ]
    t_sig = Table(sig_data, colWidths=[180, 180, 180])
    t_sig.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Oblique'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LINEABOVE', (0,0), (-1,0), 0.5, colors.black),
    ]))
    story.append(t_sig)

    doc.build(story)
    buf.seek(0)
    
    # Convert PDF to raster image, apply scan effects, save as scanned PDF
    src_doc = pymupdf.open(stream=buf.read(), filetype="pdf")
    page = src_doc[0]
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes()))
    scanned_img = add_scan_effects(img)

    # Re-save with PyMuPDF
    img_buf = io.BytesIO()
    scanned_img.save(img_buf, format="JPEG", quality=88)
    img_buf.seek(0)

    out_doc = pymupdf.open()
    rect = page.rect
    new_page = out_doc.new_page(width=rect.width, height=rect.height)
    new_page.insert_image(rect, stream=img_buf.read())
    out_doc.save(str(target))
    print(f"Created {target.name} (scanned look)")


def create_inspection_report_02():
    target = DIR_SCANNED / "inspection_report_02.pdf"
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=11, leading=14, textColor=colors.HexColor("#222222"), alignment=1, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=8, leading=11, textColor=colors.HexColor("#444444"), alignment=1, fontName="Helvetica")
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=9, leading=12, textColor=colors.HexColor("#111111"), fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2)
    b = ParagraphStyle('B', parent=styles['Normal'], fontSize=7.5, leading=10, textColor=colors.HexColor("#222222"), fontName="Helvetica")

    story = [
        Table([
            [Paragraph("<b>MANGALORE REFINERY AND PETROCHEMICALS LIMITED (MRPL)</b><br/>RELIABILITY & CONDITION MONITORING CELL", title_style)],
            [Paragraph("ROTATING MACHINERY VIBRATION & SEAL DIAGNOSTIC REPORT<br/>"
                       "REPORT NO: <b>MRPL-RMC-VIB-2026-114</b> | DATE: 02-MAR-2026 | ASSET: <b>P-101B (STANDBY)</b>", sub_style)]
        ], colWidths=[540], style=[
            ('BOX', (0,0), (-1,-1), 1.5, colors.black),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e0e8f0")),
            ('PADDING', (0,0), (-1,-1), 4),
        ]),
        Spacer(1, 6),
        Paragraph("1. VIBRATION ANALYSIS DATA SUMMARY (ISO 10816-3)", h2),
        Table([
            ["Measurement Point", "Direction", "Overall RMS (mm/s)", "Velocity Peak (mm/s)", "Dominant Harmonic", "Zone Evaluation"],
            ["Pump DE Bearing", "Horizontal", "4.82", "6.12", "1X RPM (24.7 Hz)", "Zone B (Acceptable)"],
            ["Pump DE Bearing", "Vertical", "3.20", "4.05", "1X RPM", "Zone A (Good)"],
            ["Pump DE Bearing", "Axial", "5.85", "7.40", "1X + 2X RPM", "Zone C (Unsatisfactory)"],
            ["Pump NDE Bearing", "Horizontal", "7.65", "9.80", "2X Vane Pass (148 Hz)", "Zone C (RESTRICTED RUN)"],
            ["Motor DE Bearing", "Horizontal", "2.10", "2.80", "1X RPM", "Zone A (Good)"]
        ], colWidths=[95, 65, 95, 95, 100, 90], style=[
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#333333")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 6.8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#777777")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f8f8")]),
            ('TEXTCOLOR', (5,4), (5,4), colors.HexColor("#b30000")),
        ]),
        Paragraph("2. MECHANICAL SEAL & ANCILLARY INSPECTION", h2),
        Paragraph(
            "• High axial vibration on NDE bearing correlates with acoustic emissions indicating hydraulic recirculation.<br/>"
            "• Mechanical seal cartridge (API Plan 53B) observed with barrier fluid pressure drop (3.5 bar → 2.1 bar over 48h).<br/>"
            "• Pinhole corrosion pitting noted on seal flush recirculation tubing adjacent to orifice FO-101B.<br/>"
            "• <b>Action</b>: P-101B shall not be placed in continuous service. Overhaul seal assembly and replace flush tubing.",
            b
        ),
        Spacer(1, 8),
        Table([
            ["Vibration Analyst: K. M. Prabhu", "Maintenance Superintendent: D. A. Gowda"],
            ["[Recorded & Analysed 02/03/2026]", "[Action Plan Approved 03/03/2026]"]
        ], colWidths=[270, 270], style=[
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Oblique'),
            ('FONTSIZE', (0,0), (-1,-1), 7),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('LINEABOVE', (0,0), (-1,0), 0.5, colors.black),
        ])
    ]
    doc.build(story)
    buf.seek(0)
    
    src_doc = pymupdf.open(stream=buf.read(), filetype="pdf")
    page = src_doc[0]
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes()))
    scanned_img = add_scan_effects(img)

    img_buf = io.BytesIO()
    scanned_img.save(img_buf, format="JPEG", quality=88)
    img_buf.seek(0)

    out_doc = pymupdf.open()
    rect = page.rect
    new_page = out_doc.new_page(width=rect.width, height=rect.height)
    new_page.insert_image(rect, stream=img_buf.read())
    out_doc.save(str(target))
    print(f"Created {target.name} (scanned look)")


def create_hydrotest_report():
    target = DIR_SCANNED / "hydrotest_report.pdf"
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=11, leading=14, textColor=colors.HexColor("#222222"), alignment=1, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=8, leading=11, textColor=colors.HexColor("#444444"), alignment=1, fontName="Helvetica")
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=9, leading=12, textColor=colors.HexColor("#111111"), fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2)
    b = ParagraphStyle('B', parent=styles['Normal'], fontSize=7.5, leading=10, textColor=colors.HexColor("#222222"), fontName="Helvetica")

    story = [
        Table([
            [Paragraph("<b>MANGALORE REFINERY AND PETROCHEMICALS LIMITED (MRPL)</b><br/>QA/QC & INSPECTION CELL — PIPING FABRICATION", title_style)],
            [Paragraph("HYDROSTATIC PRESSURE TEST CERTIFICATE & CLEARANCE<br/>"
                       "CERTIFICATE NO: <b>MRPL-QC-HT-2026-042</b> | TEST DATE: 04-MAR-2026 | RESULT: <b>PASSED</b>", sub_style)]
        ], colWidths=[540], style=[
            ('BOX', (0,0), (-1,-1), 1.5, colors.black),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e8efe8")),
            ('PADDING', (0,0), (-1,-1), 4),
        ]),
        Spacer(1, 6),
        Paragraph("1. TEST PACKAGE SPECIFICATION", h2),
        Table([
            ["Piping Line Number:", "2\"-HC-1015-A1A (Minimum Flow Bypass Line)", "P&ID Reference:", "MRPL-CDU-PID-001 Rev 4"],
            ["Design Pressure:", "12.0 bar (g)", "Calculated Test Pressure:", "18.0 bar (g) [1.5 × P_des]"],
            ["Test Medium:", "DM Water with 50 ppm Hydrazine inhibitor", "Ambient Temp / Fluid Temp:", "31 °C / 29 °C"],
            ["Holding Time:", "4 Hours 00 Mins (10:00 to 14:00 IST)", "Pressure Gauges Used:", "PG-HT-01 & PG-HT-02 (Dual calibrated)"]
        ], colWidths=[110, 180, 110, 140], style=[
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#999999")),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 7),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f4f4f4")),
            ('BACKGROUND', (2,0), (2,-1), colors.HexColor("#f4f4f4")),
            ('PADDING', (0,0), (-1,-1), 2.5),
        ]),
        Paragraph("2. TEST LOG & WITNESS RECORD", h2),
        Table([
            ["Step", "Time (IST)", "Target Pressure", "Observed Pressure", "Hold Period", "Visual Examination"],
            ["Stage 1", "10:00", "6.0 bar (g)", "6.05 bar (g)", "15 mins", "Initial leak check: No weep"],
            ["Stage 2", "10:30", "12.0 bar (g)", "12.00 bar (g)", "15 mins", "Check all flanged joints: No leak"],
            ["Stage 3 (Full)", "11:00", "18.0 bar (g)", "18.10 bar (g)", "Begin 4h hold", "Full circumference inspection OK"],
            ["Final Reading", "14:00", "18.0 bar (g)", "18.05 bar (g)", "Completed", "ZERO pressure drop; No deformation"]
        ], colWidths=[55, 65, 85, 90, 85, 160], style=[
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2d5a27")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 6.8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#777777")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8faf8")]),
        ]),
        Paragraph("3. INSPECTOR CONCURRENCE & PUNCH-LIST SIGN-OFF", h2),
        Paragraph(
            "Piping spool 2\"-HC-1015-A1A successfully withstood the hydrotest pressure of 18.0 bar (g). "
            "Depressurization completed in controlled stages. Dewatering and air drying completed to dewpoint -20°C. "
            "Line cleared for reinstatement and final tie-in.",
            b
        ),
        Spacer(1, 10),
        Table([
            ["Quality Control Inspector", "Third Party Inspector (EIL)", "MRPL Operations Head"],
            ["N. B. Rao (QC Lead)", "T. S. Venkat (Engineers India Ltd)", "P. K. Verma (DGM Operations)"],
            ["[PASSED & SIGNED]", "[VERIFIED & STAMPED]", "[ACCEPTED FOR RESTORATION]"]
        ], colWidths=[180, 180, 180], style=[
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 6.8),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('LINEABOVE', (0,0), (-1,0), 0.5, colors.black),
        ])
    ]
    doc.build(story)
    buf.seek(0)
    
    src_doc = pymupdf.open(stream=buf.read(), filetype="pdf")
    page = src_doc[0]
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes()))
    scanned_img = add_scan_effects(img)

    img_buf = io.BytesIO()
    scanned_img.save(img_buf, format="JPEG", quality=88)
    img_buf.seek(0)

    out_doc = pymupdf.open()
    rect = page.rect
    new_page = out_doc.new_page(width=rect.width, height=rect.height)
    new_page.insert_image(rect, stream=img_buf.read())
    out_doc.save(str(target))
    print(f"Created {target.name} (scanned look)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. High-Resolution Industrial P&ID Diagrams (PID_01.png & PID_02.png)
# ══════════════════════════════════════════════════════════════════════════════

def draw_valve(ax, x, y, size=0.25, vtype="gate", orientation="h"):
    """Draw standard P&ID valve symbol."""
    half = size / 2
    if orientation == "h":
        # Two opposing triangles
        poly1 = patches.Polygon([[x - half, y - half/1.5], [x, y], [x - half, y + half/1.5]], closed=True, ec="black", fc="white", lw=1.2)
        poly2 = patches.Polygon([[x + half, y - half/1.5], [x, y], [x + half, y + half/1.5]], closed=True, ec="black", fc="white", lw=1.2)
        ax.add_patch(poly1)
        ax.add_patch(poly2)
        # Stem and actuator
        if vtype == "control":
            ax.plot([x, x], [y, y + half*1.5], color="black", lw=1.2)
            # Diaphragm actuator (dome)
            dome = patches.Arc((x, y + half*1.5), size*1.2, size*0.8, angle=0, theta1=0, theta2=180, color="black", lw=1.2)
            ax.add_patch(dome)
            ax.plot([x - half*0.6, x + half*0.6], [y + half*1.5, y + half*1.5], color="black", lw=1.2)
        elif vtype == "check":
            # Check valve circle or arrow
            circle = patches.Circle((x, y), size*0.18, ec="black", fc="black")
            ax.add_patch(circle)
    else: # vertical
        poly1 = patches.Polygon([[x - half/1.5, y - half], [x, y], [x + half/1.5, y - half]], closed=True, ec="black", fc="white", lw=1.2)
        poly2 = patches.Polygon([[x - half/1.5, y + half], [x, y], [x + half/1.5, y + half]], closed=True, ec="black", fc="white", lw=1.2)
        ax.add_patch(poly1)
        ax.add_patch(poly2)


def draw_instrument(ax, x, y, tag, radius=0.35, dcs=True):
    """Draw ISA-5.1 instrument bubble (circle with horizontal line for DCS)."""
    circle = patches.Circle((x, y), radius, ec="black", fc="white", lw=1.2, zorder=4)
    ax.add_patch(circle)
    if dcs:
        ax.plot([x - radius, x + radius], [y, y], color="black", lw=1.0, zorder=5)
    # Text
    parts = tag.split("-")
    if len(parts) == 2:
        ax.text(x, y + radius*0.35, parts[0], ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=6)
        ax.text(x, y - radius*0.35, parts[1], ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=6)
    else:
        ax.text(x, y, tag, ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=6)


def create_pid_01():
    target = DIR_PIDS / "PID_01.png"
    _fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Outer border and title block
    ax.plot([0.5, 19.5, 19.5, 0.5, 0.5], [0.5, 0.5, 11.5, 11.5, 0.5], color="black", lw=2)
    ax.plot([0.5, 19.5], [1.8, 1.8], color="black", lw=1.2)
    ax.plot([13.0, 13.0], [0.5, 1.8], color="black", lw=1.2)
    ax.plot([16.5, 16.5], [0.5, 1.8], color="black", lw=1.2)

    # Title Block Text
    ax.text(13.2, 1.5, "MANGALORE REFINERY & PETROCHEMICALS LTD", fontsize=8, fontweight="bold")
    ax.text(13.2, 1.25, "CRUDE DISTILLATION UNIT (CDU-II)", fontsize=7.5)
    ax.text(13.2, 1.0, "FEED BOOSTER PUMP STATION P-101A/B", fontsize=8, fontweight="bold", color="#0f2a4a")
    ax.text(13.2, 0.7, "DRAWING NO: MRPL-CDU-PID-001 | REV 4", fontsize=7.5)

    ax.text(16.7, 1.5, "DESIGN: EIL / MRPL", fontsize=7)
    ax.text(16.7, 1.25, "DATE: 15-JAN-2026", fontsize=7)
    ax.text(16.7, 1.0, "CAD REF: PID-CDU-01", fontsize=7)
    ax.text(16.7, 0.7, "STATUS: AS-BUILT", fontsize=7.5, fontweight="bold", color="#1b4d3e")

    # Revision block at top right
    ax.plot([14.5, 19.5, 19.5, 14.5, 14.5], [10.2, 10.2, 11.5, 11.5, 10.2], color="black", lw=1)
    ax.text(14.7, 11.2, "REV   DATE       REVISION DESCRIPTION", fontsize=6.5, fontweight="bold")
    ax.text(14.7, 10.9, "3     10-NOV-24  Re-routed min flow bypass", fontsize=6.5)
    ax.text(14.7, 10.6, "4     15-JAN-26  Added UT tapping P5 & FCV-101", fontsize=6.5, color="#8a1c14")

    # 1. Suction Feed Drum V-101 (left)
    drum = patches.FancyBboxPatch((1.5, 4.0), 2.2, 5.0, boxstyle="Round,pad=0.3", ec="black", fc="#eef4fa", lw=1.8)
    ax.add_patch(drum)
    ax.text(2.6, 6.5, "CRUDE FEED\nDRUM\nV-101", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(2.6, 4.5, "LLLL: 1.2m\nNLL: 3.5m\nHLLL: 5.2m", ha="center", va="center", fontsize=6.5, color="#555555")

    # Feed line entering V-101
    ax.plot([0.5, 1.5], [7.5, 7.5], color="black", lw=2)
    ax.text(0.7, 7.7, "8\"-CR-1001-A1A", fontsize=7, fontweight="bold")
    draw_valve(ax, 1.0, 7.5, size=0.3)

    # 2. Level Instruments on V-101
    draw_instrument(ax, 4.5, 8.2, "LT-101", radius=0.32)
    ax.plot([3.7, 4.18], [8.2, 8.2], color="black", lw=0.8, ls="--")
    draw_instrument(ax, 4.5, 5.0, "LSLL-101", radius=0.32)
    ax.plot([3.7, 4.18], [5.0, 5.0], color="black", lw=0.8, ls="--")

    # 3. Main Suction Header coming out of bottom of V-101
    ax.plot([2.6, 2.6], [4.0, 2.8], color="black", lw=2)
    ax.plot([2.6, 6.0], [2.8, 2.8], color="black", lw=2)
    ax.text(3.5, 3.0, "6\"-HC-1010-A1A", fontsize=7.5, fontweight="bold")

    # Split to P-101A (upper branch) and P-101B (lower branch)
    # Upper branch: y=4.5
    ax.plot([6.0, 6.0], [2.8, 4.5], color="black", lw=2)
    ax.plot([6.0, 8.0], [4.5, 4.5], color="black", lw=2)
    ax.text(6.5, 4.7, "4\"-HC-1012-A1A-H", fontsize=7, fontweight="bold")
    draw_valve(ax, 6.8, 4.5, size=0.3) # Suction isolation MOV-101A
    ax.text(6.8, 4.05, "MOV-101A", ha="center", fontsize=6.5)

    # Strainer S-101A
    strainer_box = patches.Rectangle((7.4, 4.3), 0.4, 0.4, ec="black", fc="#ffffcc", lw=1)
    ax.add_patch(strainer_box)
    ax.text(7.6, 4.5, "S", ha="center", va="center", fontsize=7, fontweight="bold")
    ax.text(7.6, 4.05, "S-101A", ha="center", fontsize=6.5)

    # Suction Pressure Transmitter PT-101A
    ax.plot([7.2, 7.2], [4.5, 5.5], color="black", lw=0.8, ls="--")
    draw_instrument(ax, 7.2, 5.8, "PT-101A", radius=0.32)

    # 4. Pump P-101A symbol (circle with tangential triangle nozzle)
    pump_a = patches.Circle((8.8, 4.5), 0.65, ec="black", fc="#eef4fa", lw=1.8)
    ax.add_patch(pump_a)
    ax.plot([8.0, 8.15], [4.5, 4.5], color="black", lw=2) # suction nozzle
    ax.plot([8.8, 8.8], [5.15, 6.8], color="black", lw=2) # discharge vertical
    ax.text(8.8, 4.5, "P-101A", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(8.8, 3.6, "145 m³/h | 14.2 bar", ha="center", fontsize=6.5)

    # Discharge line P-101A
    draw_valve(ax, 8.8, 5.7, size=0.3, vtype="check", orientation="v")
    ax.text(9.3, 5.7, "NRV", fontsize=6.5)
    draw_valve(ax, 8.8, 6.4, size=0.3, orientation="v") # discharge MOV-101B
    ax.text(9.5, 6.4, "MOV-101B", fontsize=6.5)

    # 5. Lower branch (P-101B standby) at y=2.0
    ax.plot([6.0, 6.0], [2.8, 2.0], color="black", lw=2)
    ax.plot([6.0, 8.0], [2.0, 2.0], color="black", lw=2)
    ax.text(6.5, 2.2, "4\"-HC-1013-A1A-H", fontsize=7, fontweight="bold")
    draw_valve(ax, 6.8, 2.0, size=0.3)
    ax.text(6.8, 1.55, "MOV-102A", ha="center", fontsize=6.5)
    
    strainer_b = patches.Rectangle((7.4, 1.8), 0.4, 0.4, ec="black", fc="#ffffcc", lw=1)
    ax.add_patch(strainer_b)
    ax.text(7.6, 2.0, "S", ha="center", va="center", fontsize=7, fontweight="bold")
    
    pump_b = patches.Circle((8.8, 2.0), 0.65, ec="black", fc="#eef4fa", lw=1.8)
    ax.add_patch(pump_b)
    ax.text(8.8, 2.0, "P-101B\n(STBY)", ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.plot([8.8, 8.8], [2.65, 6.8], color="black", lw=2)
    draw_valve(ax, 8.8, 3.2, size=0.3, vtype="check", orientation="v")

    # Join discharge headers at y=6.8
    ax.plot([8.8, 13.5], [6.8, 6.8], color="black", lw=2)
    ax.text(10.2, 7.0, "4\"-HC-1014-A1A-H (MAIN DISCHARGE)", fontsize=7.5, fontweight="bold")

    # Flow transmitter FT-101 on discharge
    ax.plot([10.8, 10.8], [6.8, 7.8], color="black", lw=0.8, ls="--")
    draw_instrument(ax, 10.8, 8.1, "FT-101", radius=0.32)

    # 6. Minimum Flow Recirculation Bypass line (y=8.8) looping back to V-101
    ax.plot([12.0, 12.0], [6.8, 9.2], color="black", lw=1.8)
    ax.plot([12.0, 3.0], [9.2, 9.2], color="black", lw=1.8)
    ax.plot([3.0, 3.0], [9.2, 9.0], color="black", lw=1.8)
    ax.text(7.0, 9.4, "2\"-HC-1015-A1A (MINIMUM FLOW RECIRCULATION LOOP)", fontsize=7.5, fontweight="bold", color="#8a1c14")

    # Control valve FCV-101 on bypass line
    draw_valve(ax, 9.0, 9.2, size=0.35, vtype="control")
    ax.text(9.0, 8.5, "FCV-101", ha="center", fontsize=7.5, fontweight="bold")
    ax.text(9.0, 8.2, "FO (Fail Open)", ha="center", fontsize=6.5)

    # Instrument line linking FT-101 to FIC-101 to FCV-101
    draw_instrument(ax, 10.8, 9.8, "FIC-101", radius=0.32)
    ax.plot([10.8, 10.8], [8.42, 9.48], color="black", lw=0.8, ls=":")
    ax.plot([10.48, 9.2], [9.8, 9.8], color="black", lw=0.8, ls=":")
    ax.plot([9.2, 9.0], [9.8, 9.5], color="black", lw=0.8, ls=":")

    # 7. To Distillation Column C-101 arrow at right
    ax.plot([13.5, 14.5], [6.8, 6.8], color="black", lw=2)
    ax.annotate("", xy=(14.5, 6.8), xytext=(14.2, 6.8), arrowprops={"arrowstyle": "->", "lw": 2, "color": "black"})
    ax.text(14.6, 6.7, "TO FRACTIONATION\nCOLUMN C-101", fontsize=7.5, fontweight="bold")

    # Inspection Tag Point 5 (Red Box highlighting UT thinning zone)
    ut_box = patches.Rectangle((6.2, 4.35), 0.3, 0.3, ec="red", fc="none", lw=1.5, ls="--")
    ax.add_patch(ut_box)
    ax.text(6.35, 5.0, "UT POINT P5\n(t=3.42mm)", color="red", fontsize=6.5, fontweight="bold", ha="center")

    plt.tight_layout()
    plt.savefig(str(target), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Created {target.name} (P&ID diagram)")


def create_pid_02():
    target = DIR_PIDS / "PID_02.png"
    _fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Border & Title block
    ax.plot([0.5, 19.5, 19.5, 0.5, 0.5], [0.5, 0.5, 11.5, 11.5, 0.5], color="black", lw=2)
    ax.plot([0.5, 19.5], [1.8, 1.8], color="black", lw=1.2)
    ax.plot([13.0, 13.0], [0.5, 1.8], color="black", lw=1.2)

    ax.text(13.2, 1.5, "MANGALORE REFINERY & PETROCHEMICALS LTD", fontsize=8, fontweight="bold")
    ax.text(13.2, 1.25, "CDU-II FRACTIONATION COLUMN C-101", fontsize=7.5)
    ax.text(13.2, 1.0, "REBOILER & BOTTOMS SYSTEM E-102", fontsize=8, fontweight="bold", color="#0f2a4a")
    ax.text(13.2, 0.7, "DRAWING NO: MRPL-CDU-PID-002 | REV 3", fontsize=7.5)

    # Column C-101 (tall vertical vessel)
    column = patches.FancyBboxPatch((5.0, 2.5), 3.0, 8.0, boxstyle="Round,pad=0.4", ec="black", fc="#fdfefe", lw=2)
    ax.add_patch(column)
    ax.text(6.5, 6.5, "ATMOSPHERIC\nFRACTIONATION\nCOLUMN\nC-101", ha="center", va="center", fontsize=9.5, fontweight="bold")

    # Feed inlet from PID-01
    ax.plot([1.0, 5.0], [7.0, 7.0], color="black", lw=2)
    ax.text(1.2, 7.2, "FROM P-101A/B: 4\"-HC-1014-A1A", fontsize=7.5, fontweight="bold")

    # Overhead vapor line
    ax.plot([6.5, 6.5], [10.5, 11.0], color="black", lw=2)
    ax.plot([6.5, 11.0], [11.0, 11.0], color="black", lw=2)
    ax.text(7.5, 11.2, "12\"-OVHD-1020-A1A", fontsize=7.5, fontweight="bold")
    draw_instrument(ax, 9.5, 10.3, "PT-102", radius=0.32)

    # Thermosyphon Reboiler E-102 at bottom
    reboiler = patches.Rectangle((10.0, 2.8), 3.0, 1.8, ec="black", fc="#eef4fa", lw=1.8)
    ax.add_patch(reboiler)
    ax.text(11.5, 3.7, "THERMOSYPHON\nREBOILER\nE-102", ha="center", va="center", fontsize=8.5, fontweight="bold")

    # Circulation piping from column bottom to reboiler
    ax.plot([6.5, 6.5], [2.5, 2.0], color="black", lw=2)
    ax.plot([6.5, 10.5], [2.0, 2.0], color="black", lw=2)
    ax.plot([10.5, 10.5], [2.0, 2.8], color="black", lw=2)
    ax.text(7.5, 2.2, "8\"-BTM-1030-A1A", fontsize=7.5, fontweight="bold")

    # Vapor return to column
    ax.plot([11.5, 11.5], [4.6, 5.2], color="black", lw=2)
    ax.plot([11.5, 8.0], [5.2, 5.2], color="black", lw=2)
    ax.text(9.0, 5.4, "8\"-VAP-1032-A1A", fontsize=7, fontweight="bold")

    # Steam supply to Reboiler
    ax.plot([14.5, 13.0], [4.2, 4.2], color="black", lw=1.8)
    ax.text(13.6, 4.4, "4\"-STM-2001", fontsize=7, fontweight="bold")
    draw_valve(ax, 13.7, 4.2, size=0.35, vtype="control")
    ax.text(13.7, 3.6, "TCV-102", ha="center", fontsize=7, fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(target), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Created {target.name} (P&ID diagram)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Handwritten Redline Field Notes (handwritten_annotation.pdf)
# ══════════════════════════════════════════════════════════════════════════════

def create_handwritten_annotation():
    target = DIR_HANDWRITTEN / "handwritten_annotation.pdf"
    
    # Create an authentic yellow-lined notepad background with cursive handwriting
    width, height = 1200, 1600
    img = Image.new("RGB", (width, height), (252, 248, 232)) # yellow legal pad
    draw = ImageDraw.Draw(img)

    # Draw ruled lines
    margin_x = 140
    # Left red margin line
    draw.line([(margin_x, 0), (margin_x, height)], fill=(220, 120, 120), width=3)
    draw.line([(margin_x + 4, 0), (margin_x + 4, height)], fill=(220, 120, 120), width=1)

    # Horizontal blue ledger lines
    line_spacing = 48
    for y in range(120, height, line_spacing):
        draw.line([(0, y), (width, y)], fill=(195, 215, 235), width=1)

    # Header in print style
    draw.text((margin_x + 30, 65), "MRPL CDU-II PRE-COMMISSIONING WALKDOWN NOTES", fill=(60, 60, 60))
    draw.text((width - 320, 65), "DATE: 04/09/2026", fill=(60, 60, 60))

    # Handwritten text simulation (slight jitter and blue ink)
    notes = [
        "1. Walkdown completed with Shift In-charge (Operations).",
        "2. Inspected P-101A suction flange & strainer S-101A.",
        "   - Noted slight gland weeping on bypass valve FCV-101.",
        "   - Action: Retorque bonnet studs before hydrotest sign-off.",
        "3. CRITICAL NDT FINDING:",
        "   - UT Point P5 on Elbow #2 extrados measured 3.42 mm.",
        "   - Minimum retirement thickness is 3.20 mm.",
        "   - Current corrosion rate = 0.28 mm/yr.",
        "   - Will reach retirement in approx 9 months.",
        "4. Standby pump P-101B vibration is high (7.65 mm/s on NDE).",
        "   - Seal flush Plan 53B bladder accumulator has pressure drop.",
        "   - DO NOT run P-101B continuously without seal overhaul.",
        "5. Minimum flow bypass line 2\"-HC-1015 successfully hydrotested",
        "   at 18.0 bar(g) for 4 hrs. Zero pressure drop recorded.",
        "",
        "Recommendations for GM Approval Note:",
        "-> Order replacement 4\" spool for Elbow #2 immediately.",
        "-> Authorize mechanical seal rebuild for P-101B.",
        "",
        "Field Engineer: R. K. Sharma (Sr. Mgr Mech, CDU-II)",
        "Witnessed: P. Verma (DGM Operations)"
    ]

    y_pos = 140
    for line in notes:
        # Add slight random ink color variation (ballpoint blue)
        ink = (25 + random.randint(-5, 5), 45 + random.randint(-5, 5), 135 + random.randint(-10, 10))
        # Add slight x, y jitter
        jx = random.randint(-2, 2)
        jy = random.randint(-1, 1)
        draw.text((margin_x + 25 + jx, y_pos + jy), line, fill=ink)
        y_pos += line_spacing

    # Convert to PDF via PyMuPDF
    img_buf = io.BytesIO()
    img.save(img_buf, format="JPEG", quality=90)
    img_buf.seek(0)

    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842) # A4
    page.insert_image(page.rect, stream=img_buf.read())
    pdf.save(str(target))
    print(f"Created {target.name} (handwritten notepad notes)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Approval Note Reference Document (approval_note_reference.pdf)
# ══════════════════════════════════════════════════════════════════════════════

def create_approval_note():
    target = DIR_APPROVAL / "approval_note_reference.pdf"
    doc = SimpleDocTemplate(
        str(target), pagesize=A4,
        leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=12, leading=15, textColor=colors.HexColor("#0f2a4a"), alignment=1, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=colors.HexColor("#333333"), alignment=1, fontName="Helvetica")
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=9.5, leading=13, textColor=colors.HexColor("#0f2a4a"), fontName="Helvetica-Bold", spaceBefore=7, spaceAfter=2)
    b = ParagraphStyle('B', parent=styles['Normal'], fontSize=8, leading=11.5, textColor=colors.HexColor("#222222"), fontName="Helvetica")

    story = [
        Table([
            [Paragraph("<b>MANGALORE REFINERY AND PETROCHEMICALS LIMITED (MRPL)</b><br/>TECHNICAL SERVICES & OPERATIONS DIVISION", title_style)],
            [Paragraph("TECHNICAL MEMORANDUM & NOTE FOR APPROVAL (NFA)<br/>"
                       "INTER-OFFICE MEMO | REF: <b>MRPL/CDU-II/NFA/2026/014</b> | CLASSIFICATION: <b>CONFIDENTIAL</b>", sub_style)]
        ], colWidths=[515], style=[
            ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#0f2a4a")),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f4f8")),
            ('PADDING', (0,0), (-1,-1), 4),
        ]),
        Spacer(1, 8),
        Table([
            ["TO:", "Chief General Manager (Technical Services & Operations)"],
            ["THROUGH:", "General Manager (Maintenance & Reliability)"],
            ["FROM:", "Senior Manager (Mechanical Maintenance, CDU-II)"],
            ["DATE:", "05-MAR-2026"],
            ["SUBJECT:", "Approval for Emergency Procurement of Spool Piece 4\"-HC-1012 & Overhaul of P-101B Seal Assembly"]
        ], colWidths=[80, 435], style=[
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
            ('PADDING', (0,0), (-1,-1), 2.5),
        ]),
        Spacer(1, 6),
        Paragraph("1. BACKGROUND & PROBLEM STATEMENT", h2),
        Paragraph(
            "During the ongoing baseline integrity assessment and pre-turnaround survey of the Crude Distillation Unit (CDU-II), "
            "comprehensive Non-Destructive Testing (NDT) was performed on Crude Feed Booster Pumps P-101A/B and associated piping. "
            "Ultrasonic gauging report MRPL-NDT-UT-2026-089 and vibration diagnostic report MRPL-RMC-VIB-2026-114 reveal two critical integrity threats "
            "requiring immediate management approval and procurement action prior to the monsoon turnaround.",
            b
        ),
        Paragraph("2. KEY TECHNICAL FINDINGS", h2),
        Paragraph(
            "• <b>Wall Thinning (P-101A Suction Line 4\"-HC-1012-A1A-H)</b>: Thickness at Elbow #2 extrados has degraded to <b>3.42 mm</b> "
            "(nominal 6.02 mm). The ASME B31.3 calculated retirement thickness is <b>3.20 mm</b>. With the current corrosion rate of 0.28 mm/year, "
            "remaining service life is calculated at only <b>9.4 months</b>.<br/>"
            "• <b>Standby Pump P-101B Unavailability</b>: Elevated axial vibration (7.65 mm/s RMS) and continuous pressure decay in API Plan 53B "
            "barrier system indicate imminent seal failure and hydraulic recirculation. P-101B cannot serve as a reliable standby.",
            b
        ),
        Paragraph("3. RISK ASSESSMENT & MITIGATION PROPOSAL", h2),
        Paragraph(
            "Operating beyond June 2026 without pipe replacement exposes the refinery to a catastrophic loss of containment (LOC) "
            "hazard of 145 m³/h hot hydrocarbon feed at 1.8 bar(g). The following actions are recommended for immediate approval:<br/>"
            "1. <b>Fabrication & Replacement of 4\" Spool</b>: Pre-fabricate replacement spool piece in ASTM A106 Gr. B with internal cladding.<br/>"
            "2. <b>OEM Mechanical Seal Overhaul</b>: Mobilize OEM service engineer for P-101B cartridge seal rebuild and accumulator recharge.<br/>"
            "3. <b>Estimated Outlay</b>: Total expenditure of INR 14.50 Lakhs against Operational Maintenance Budget (CDU-II OPEX).",
            b
        ),
        Paragraph("4. FINANCIAL & PROCUREMENT APPROVAL MATRIX", h2),
        Table([
            ["Item", "Scope / Description", "Vendor / Contractor", "Estimated Cost (INR)", "Budget Code"],
            ["1", "4\" Carbon Steel Spool (ASME B16.9 Elbows, cladded)", "L&T Heavy Engineering / Approved Fabricator", "₹ 6,80,000", "OPEX-CDU2-26-01"],
            ["2", "API Plan 53B Seal Cartridge Rebuild Kit", "Flowserve India Pvt. Ltd.", "₹ 5,20,000", "OPEX-CDU2-26-02"],
            ["3", "Site Radiography, NDT & Third Party EIL Inspection", "Engineers India Ltd (TPI)", "₹ 2,50,000", "OPEX-CDU2-26-03"],
            ["<b>TOTAL</b>", "<b>Emergency Turnaround Remediation</b>", "—", "<b>₹ 14,50,000</b>", "<b>APPROVED CAP</b>"]
        ], colWidths=[30, 225, 130, 70, 60], style=[
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f2a4a")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 6.8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ]),
        Spacer(1, 10),
        Paragraph("5. SIGN-OFF & INTER-DEPARTMENTAL CONCURRENCE", h2),
        Table([
            ["Initiated By", "Concurred By", "Finance Concurrence", "Approved By"],
            ["R. K. Sharma\nSr. Mgr (Mechanical)", "D. A. Gowda\nGM (Maintenance)", "S. Sengupta\nDGM (Finance)", "P. K. Verma\nCGM (Operations)"],
            ["[Signed 05/03/2026]", "[Signed 05/03/2026]", "[Concurred 06/03/2026]", "[APPROVED 06/03/2026]"]
        ], colWidths=[128, 128, 128, 131], style=[
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 6.5),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#999999")),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f4f8")),
        ])
    ]
    doc.build(story)
    print(f"Created {target.name}")


def main():
    print(f"Generating MRPL Demo Corpus in: {CORPUS_DIR}\n")
    create_sop_01()
    create_sop_02()
    create_technical_procedure()
    create_pid_guide()
    create_inspection_report_01()
    create_inspection_report_02()
    create_hydrotest_report()
    create_pid_01()
    create_pid_02()
    create_handwritten_annotation()
    create_approval_note()
    print("\nAll 11 demo corpus assets successfully created!")


if __name__ == "__main__":
    main()
