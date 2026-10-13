"""Fixtures for the knowledge-plane tests.

Not a conftest: these are helpers several test modules import explicitly, and
naming them makes it obvious in each test which fake is in play. A test that
silently picks up a fake image reader from a conftest is a test whose result you
cannot read off the page.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from kavach.knowledge import Extraction, ImagePathStatus


def native_pdf(path: Path, text: str, pages: int = 1) -> Path:
    """A PDF with a real text layer - the fast path through the fork."""
    doc = pymupdf.open()
    for index in range(pages):
        page = doc.new_page()
        body = text if pages == 1 else f"{text}\n\nPAGE MARKER {index + 1}"
        page.insert_textbox(
            pymupdf.Rect(40, 40, 550, 780), body, fontsize=9, fontname="helv"
        )
    doc.save(path)
    doc.close()
    return path


def scanned_pdf(path: Path, thin_text: str = "") -> Path:
    """A page with no usable text layer - a picture of text, as far as we know.

    `thin_text` stands in for the footer a scanner stamps on: enough characters
    to fool a `if page.get_text()` check, nowhere near enough to be the page.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    if thin_text:
        page.insert_text((40, 40), thin_text, fontsize=8, fontname="helv")
    doc.save(path)
    doc.close()
    return path


class FakeReader:
    """An ImageReader stand-in with a decided answer.

    Everything about the image path is a property of the box - whether tesseract
    is installed, whether the vision model has weights - so the tests state it
    outright instead of hoping the CI runner agrees.
    """

    def __init__(
        self,
        text: str = "",
        extraction: Extraction = Extraction.UNREADABLE,
        detail: str = "fake reader",
        raises: Exception | None = None,
    ):
        self.text = text
        self.extraction = extraction
        self.detail = detail
        self.raises = raises
        self.calls: list[str] = []

    def status(self) -> ImagePathStatus:
        return ImagePathStatus(
            ocr_engine="fake" if self.extraction is Extraction.OCR else None,
            vision_model="fake-vl" if self.extraction in
            (Extraction.VLM, Extraction.SIMULATED) else None,
            vision_is_simulated=self.extraction is Extraction.SIMULATED,
            simulated_allowed=self.extraction is Extraction.SIMULATED,
        )

    def read(self, png: bytes, *, hint: str = "") -> tuple[str, Extraction, str]:
        self.calls.append(hint)
        if self.raises is not None:
            # ImageReader.read is contractually non-raising; a fake that raises
            # proves extract_document does not depend on that contract holding.
            raise self.raises
        return self.text, self.extraction, self.detail


SOP_TEXT = """\
MANGALORE REFINERY AND PETROCHEMICALS LIMITED (MRPL)
STANDARD OPERATING PROCEDURE: HYDROCARBON FEED PUMP STARTUP
DOC NO: MRPL-SOP-CDU-0101 | REV: 04 | CLASSIFICATION: CONFIDENTIAL

1. PURPOSE AND APPLICABILITY
This procedure covers the startup and emergency isolation of Crude Booster
Pumps P-101A/B serving atmospheric distillation train CDU-II. It applies to all
plant operators and shift engineers on the unit at all times of operation.

2. SAFETY INTERLOCKS AND OPERATIONAL THRESHOLDS
PSLL-101 suction pressure trip setpoint is 1.80 bar gauge with an alarm setting
of 2.05 bar gauge. The interlock trips the motor instantly to prevent dry-run
damage to the mechanical seal faces and the pump bearing housing assembly.
PSHH-101 discharge pressure trips at 18.5 bar gauge on sustained deadhead.

3. STARTUP SEQUENCE
Confirm the suction valve is fully open before energising the motor. Observe the
discharge pressure gauge PI-1012 rise smoothly to shutoff head near 15.4 bar.
"""

HSE_TEXT = """\
MANGALORE REFINERY AND PETROCHEMICALS LIMITED (MRPL)
CORPORATE SAFETY PROCEDURE: HOT WORK AND CONFINED SPACE ENTRY
DOC NO: MRPL-SOP-HSE-0204 | REV: 06 | CLASSIFICATION: INTERNAL

1. PERMIT TO WORK PROTOCOL
No hot work may commence without a valid permit countersigned by the area
authority and the fire watch. Permits expire at the end of the issuing shift and
may not be carried over to a following shift under any circumstances.

2. GAS TESTING THRESHOLDS
Hydrogen sulfide must read below 5.0 ppm on a continuous multi-gas detector.
Carbon monoxide must read below 25.0 ppm prior to entry and again post-welding.
"""
