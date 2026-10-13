"""The ingest fork (§07).

The property under test throughout: a page Kavach could not read is recorded as a
page Kavach could not read. Every plausible-looking alternative - empty string,
skipped page, dropped document - produces a corpus that answers questions from a
hole it cannot describe.
"""

from __future__ import annotations

import pytest

from kavach.knowledge import Extraction, IngestError, extract_document
from kavach.knowledge.config import load_knowledge_config
from kavach.knowledge.extract import ImagePathStatus, ImageReader, detect_label
from kavach.types import Label
from kbfixtures import SOP_TEXT, FakeReader, native_pdf, scanned_pdf


@pytest.fixture()
def config():
    return load_knowledge_config()


# ---------------------------------------------------------------- the fork

def test_a_text_layer_is_read_exactly_and_never_transcribed(tmp_path, config):
    reader = FakeReader(text="TRANSCRIBED", extraction=Extraction.OCR)
    doc = extract_document(native_pdf(tmp_path / "sop.pdf", SOP_TEXT), config, reader=reader)

    assert doc.pages[0].extraction is Extraction.NATIVE
    assert "PSLL-101" in doc.pages[0].text
    # The image path must not have been consulted at all: transcription can be
    # wrong, and a page that carries its own characters has nothing to gain.
    assert reader.calls == []


def test_a_scan_with_no_reader_is_unreadable_not_empty(tmp_path, config):
    doc = extract_document(
        scanned_pdf(tmp_path / "scan.pdf"), config, reader=FakeReader()
    )
    page = doc.pages[0]

    assert page.extraction is Extraction.UNREADABLE
    assert page.text == ""
    # The page still exists in the record. An unreadable page that is silently
    # dropped is a gap nobody can report.
    assert doc.record.page_count == 1
    assert doc.record.unreadable_pages == [1]
    assert doc.record.readable_pages == 0


def test_a_thin_scanner_footer_does_not_count_as_a_text_layer(tmp_path, config):
    """`if page.get_text()` is true for a page whose only text is 'Page 1 of 7'."""
    doc = extract_document(
        scanned_pdf(tmp_path / "thin.pdf", thin_text="Page 1 of 7"),
        config,
        reader=FakeReader(),
    )
    page = doc.pages[0]

    assert page.extraction is Extraction.UNREADABLE
    # And the diagnosis says which case this was, because "no text at all" and
    # "a footer under the threshold" call for different fixes.
    assert "text layer" in page.detail
    assert "threshold" in page.detail


def test_ocr_output_is_recorded_as_ocr_not_native(tmp_path, config):
    doc = extract_document(
        scanned_pdf(tmp_path / "scan.pdf"),
        config,
        reader=FakeReader(text="TRIP SETPOINT 1.8 BAR", extraction=Extraction.OCR),
    )
    assert doc.pages[0].extraction is Extraction.OCR
    assert doc.pages[0].text == "TRIP SETPOINT 1.8 BAR"


def test_a_reader_that_blows_up_does_not_take_the_document_with_it(tmp_path, config):
    """One unreadable page must not abort a corpus load half way through."""
    reader = ImageReader()
    reader.status = lambda: ImagePathStatus(None, "vl", False, False)
    reader._vision_spec = lambda: (_ for _ in ()).throw(RuntimeError("no card"))

    doc = extract_document(scanned_pdf(tmp_path / "scan.pdf"), config, reader=reader)
    assert doc.pages[0].extraction is Extraction.UNREADABLE


def test_a_bare_image_takes_the_image_path_with_no_fork(tmp_path, config):
    png = tmp_path / "pid.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    reader = FakeReader()

    doc = extract_document(png, config, reader=reader)
    assert doc.pages[0].extraction is Extraction.UNREADABLE
    assert reader.calls == ["pid.png"]


def test_a_multi_page_pdf_forks_per_page(tmp_path, config):
    """The decision is per page, not per document - a report is often a native
    cover sheet stapled to scanned annexes."""
    doc = extract_document(
        native_pdf(tmp_path / "multi.pdf", SOP_TEXT, pages=3), config, reader=FakeReader()
    )
    assert [p.page for p in doc.pages] == [1, 2, 3]
    assert all(p.extraction is Extraction.NATIVE for p in doc.pages)


# ---------------------------------------------------------------- refusals

def test_an_unknown_file_type_names_what_it_can_open(tmp_path, config):
    junk = tmp_path / "notes.docx"
    junk.write_bytes(b"PK\x03\x04")
    with pytest.raises(IngestError, match="not a document type"):
        extract_document(junk, config, reader=FakeReader())


def test_a_missing_file_is_an_ingest_error(tmp_path, config):
    with pytest.raises(IngestError, match="does not exist"):
        extract_document(tmp_path / "ghost.pdf", config, reader=FakeReader())


# ---------------------------------------------------------------- classification

def test_the_marking_on_the_document_wins(tmp_path, config):
    doc = extract_document(
        native_pdf(tmp_path / "sop.pdf", SOP_TEXT), config, reader=FakeReader()
    )
    assert doc.record.label is Label.CONFIDENTIAL


def test_detect_label_ignores_prose_that_merely_mentions_a_label():
    assert detect_label("this document is not confidential in nature") is None
    assert detect_label("CLASSIFICATION: RESTRICTED") is Label.RESTRICTED


def test_an_unread_page_over_classifies_rather_than_under_classifies(tmp_path, config):
    """The marking usually sits in the header of the page we failed to read.

    Defaulting that document to INTERNAL hands an unread RESTRICTED scan to an
    operator, and nothing downstream would ever notice.
    """
    doc = extract_document(
        scanned_pdf(tmp_path / "scan.pdf"), config, reader=FakeReader()
    )
    assert doc.record.label is Label.CONFIDENTIAL
    assert doc.record.unreadable_pages == [1]


def test_an_explicit_label_overrides_the_marking(tmp_path, config):
    doc = extract_document(
        native_pdf(tmp_path / "sop.pdf", SOP_TEXT),
        config,
        reader=FakeReader(),
        label=Label.RESTRICTED,
    )
    assert doc.record.label is Label.RESTRICTED


def test_the_same_bytes_hash_the_same_twice(tmp_path, config):
    path = native_pdf(tmp_path / "sop.pdf", SOP_TEXT)
    first = extract_document(path, config, reader=FakeReader()).record
    second = extract_document(path, config, reader=FakeReader()).record
    assert first.sha256 == second.sha256 and len(first.sha256) == 64


# ---------------------------------------------------------------- image path status

def test_a_simulated_vision_model_is_not_an_image_path():
    status = ImagePathStatus(
        ocr_engine=None, vision_model="qwen3-vl-8b-q4",
        vision_is_simulated=True, simulated_allowed=False,
    )
    assert not status.available
    assert status.mode is Extraction.UNREADABLE
    # And it says what would change the answer, rather than just failing.
    assert "KAVACH_KB_ALLOW_SIMULATED_OCR" in status.detail


def test_the_simulated_path_must_be_asked_for_by_name():
    status = ImagePathStatus(
        ocr_engine=None, vision_model="qwen3-vl-8b-q4",
        vision_is_simulated=True, simulated_allowed=True,
    )
    assert status.available
    assert status.mode is Extraction.SIMULATED
    assert "may not be quoted" in status.detail


def test_a_real_ocr_engine_beats_a_vision_model():
    """OCR is deterministic and cheap; a VLM can hallucinate a plausible tag."""
    status = ImagePathStatus(
        ocr_engine="tesseract", vision_model="qwen3-vl-8b-q4",
        vision_is_simulated=False, simulated_allowed=False,
    )
    assert status.mode is Extraction.OCR


def test_extraction_kinds_know_what_may_be_quoted():
    assert Extraction.NATIVE.is_real and Extraction.OCR.is_real
    assert not Extraction.SIMULATED.is_real
    assert not Extraction.UNREADABLE.is_real
    # SIMULATED is indexable behind an opt-in; UNREADABLE has no text to index.
    assert Extraction.SIMULATED.is_indexed
    assert not Extraction.UNREADABLE.is_indexed


# ------------------------------------------------- finding the OCR engine


def test_tesseract_is_found_when_it_is_installed_but_not_on_path(tmp_path, monkeypatch):
    """The Windows installer does NOT add itself to PATH unless the box is
    ticked, and almost nobody ticks it.

    Checking PATH alone reported "not installed" on a machine where Tesseract
    was properly installed - so the panel told its owner to install the thing
    they had just installed, and every scan went to a slow vision model
    instead.
    """
    from kavach.knowledge import extract

    fake = tmp_path / "Tesseract-OCR" / "tesseract.exe"
    fake.parent.mkdir(parents=True)
    fake.write_text("", encoding="utf-8")

    monkeypatch.setattr(extract.shutil, "which", lambda _name: None)
    monkeypatch.setattr(extract, "_TESSERACT_DEFAULTS", (str(fake),))
    monkeypatch.delenv("KAVACH_TESSERACT", raising=False)

    assert extract.tesseract_binary() == str(fake)
    assert extract.tesseract_available() is True


def test_the_setting_points_at_either_the_folder_or_the_executable(tmp_path, monkeypatch):
    """An operator setting this will reasonably do either."""
    from kavach.knowledge import extract

    binary = tmp_path / "tesseract.exe"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(extract.shutil, "which", lambda _name: None)
    monkeypatch.setattr(extract, "_TESSERACT_DEFAULTS", ())

    for value in (str(tmp_path), str(binary)):
        monkeypatch.setenv("KAVACH_TESSERACT", value)
        assert extract.tesseract_binary() == str(binary), value


def test_path_still_wins_over_everything(tmp_path, monkeypatch):
    """A machine that put it on PATH deliberately must not be second-guessed by
    a default location from a list in our source."""
    from kavach.knowledge import extract

    monkeypatch.setattr(extract.shutil, "which", lambda _name: "/usr/bin/tesseract")
    monkeypatch.setenv("KAVACH_TESSERACT", str(tmp_path))
    assert extract.tesseract_binary() == "/usr/bin/tesseract"


def test_no_engine_anywhere_is_reported_as_absent(monkeypatch):
    """The honest half: it must still be able to say no."""
    from kavach.knowledge import extract

    monkeypatch.setattr(extract.shutil, "which", lambda _name: None)
    monkeypatch.setattr(extract, "_TESSERACT_DEFAULTS", ("/nowhere/at/all",))
    monkeypatch.delenv("KAVACH_TESSERACT", raising=False)

    assert extract.tesseract_binary() is None
    assert extract.tesseract_available() is False
