"""Versioned, book-specific crop templates.

Coordinates are relative to the normalized full face.  A template is never
silently shared with another book: unknown books deliberately fall back to the
legacy generic layout and remain identifiable by a different layout id.
"""

from __future__ import annotations

from dataclasses import dataclass

from .record_regions import BBox, bbox_linha_nome, bbox_numero_termo, bbox_registro


@dataclass(frozen=True)
class BookLayout:
    layout_id: str
    version: int
    records_per_face: int
    record_bboxes: tuple[BBox, ...]
    name_bboxes: tuple[BBox, ...]
    term_bboxes: tuple[BBox, ...]
    confidence: float
    calibration: str


# Confirmed visually on leaf 1 front/back, leaf 150 front/back and leaf 300
# front.  The right edge (0.735) stops before the printed annotations column.
A16_LAYOUT = BookLayout(
    layout_id="a-16-nascimentos-v1",
    version=1,
    records_per_face=2,
    record_bboxes=((0.055, 0.010, 0.735, 0.493), (0.055, 0.500, 0.735, 0.990)),
    # Faixas finais calibradas nas cinco faces. Incluem o rotulo impresso e a
    # escrita da linha, mas nao alcancam sexo/filiacao nem a averbacao. Isso
    # permite enviar o A-16 direto ao Qwen sem uma passada RapidOCR anterior.
    name_bboxes=((0.155, 0.170, 0.735, 0.225), (0.155, 0.645, 0.735, 0.710)),
    term_bboxes=((0.030, 0.035, 0.205, 0.090), (0.030, 0.520, 0.205, 0.575)),
    confidence=0.96,
    calibration="5_faces_confirmadas_2026-08-10",
)


def layout_for_book(book_code: str | None) -> BookLayout | None:
    normalized = str(book_code or "").strip().upper().replace("_", "-")
    if normalized == "A-16":
        return A16_LAYOUT
    return None


def record_bbox_for_book(book_code: str | None, index: int, total: int) -> BBox:
    layout = layout_for_book(book_code)
    if layout and int(total) == layout.records_per_face:
        return layout.record_bboxes[max(0, min(int(index), total - 1))]
    return bbox_registro(index, total)


def name_bbox_for_book(book_code: str | None, index: int, total: int) -> BBox:
    layout = layout_for_book(book_code)
    if layout and int(total) == layout.records_per_face:
        return layout.name_bboxes[max(0, min(int(index), total - 1))]
    return bbox_linha_nome(index, total)


def term_bbox_for_book(book_code: str | None, index: int, total: int) -> BBox:
    layout = layout_for_book(book_code)
    if layout and int(total) == layout.records_per_face:
        return layout.term_bboxes[max(0, min(int(index), total - 1))]
    return bbox_numero_termo(index, total)
