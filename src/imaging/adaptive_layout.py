"""Descoberta segura de layouts e quantidade de assentos por face.

O modulo usa somente evidencias estruturais baratas do OpenCV. Ele nao tenta
ler manuscrito e nao chama Qwen. A leitura continua sendo responsabilidade dos
recortes produzidos pelo pipeline de OCR/Qwen.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .record_regions import (
    BBox,
    bbox_linha_nome,
    bbox_numero_termo,
    bbox_registro,
    recortar_bbox,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayoutObservation:
    layout_id: str | None
    records_per_face: int
    confidence: float
    method: str
    needs_review: bool
    reason: str
    separator_y: float | None
    record_bboxes: tuple[BBox, ...]
    name_bboxes: tuple[BBox, ...]
    term_bboxes: tuple[BBox, ...]
    signature: tuple[float, ...]
    page_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout_id,
            "records_per_face": self.records_per_face,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "needs_review": self.needs_review,
            "reason": self.reason,
            "separator_y": self.separator_y,
            "record_bboxes": [list(bbox) for bbox in self.record_bboxes],
            "name_bboxes": [list(bbox) for bbox in self.name_bboxes],
            "term_bboxes": [list(bbox) for bbox in self.term_bboxes],
            "page_number": self.page_number,
        }


@dataclass
class LayoutTemplate:
    layout_id: str
    records_per_face: int
    signature: list[float]
    sample_pages: list[int] = field(default_factory=list)
    mean_confidence: float = 0.0
    reference_image: str = ""
    candidate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout_id,
            "records_per_face": self.records_per_face,
            "signature": self.signature,
            "sample_pages": self.sample_pages,
            "mean_confidence": round(self.mean_confidence, 4),
            "reference_image": self.reference_image,
            "candidate": self.candidate,
        }


def _similaridade(left: tuple[float, ...] | list[float], right: tuple[float, ...] | list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    a = np.asarray(left[:size], dtype=np.float32)
    b = np.asarray(right[:size], dtype=np.float32)
    distance = float(np.mean(np.abs(a - b)))
    return max(0.0, min(1.0, 1.0 - distance / 0.45))


class LayoutStore:
    """Persistencia JSON dos templates por livro, sem apagar banco legado."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.templates: list[LayoutTemplate] = []
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Templates de layout invalidos em %s: %s", self.path, exc)
            return
        for item in raw.get("templates", []) if isinstance(raw, dict) else []:
            try:
                self.templates.append(
                    LayoutTemplate(
                        layout_id=str(item["layout"]),
                        records_per_face=max(1, int(item["records_per_face"])),
                        signature=[float(value) for value in item.get("signature", [])],
                        sample_pages=[int(value) for value in item.get("sample_pages", [])],
                        mean_confidence=float(item.get("mean_confidence") or 0),
                        reference_image=str(item.get("reference_image") or ""),
                        candidate=bool(item.get("candidate")),
                    )
                )
            except (TypeError, ValueError, KeyError):
                logger.warning("Template de layout ignorado: %r", item)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "templates": [template.to_dict() for template in self.templates],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def add(self, observation: LayoutObservation, page_number: int | None) -> LayoutTemplate:
        layout_id = f"layout_{len(self.templates) + 1:03d}"
        template = LayoutTemplate(
            layout_id=layout_id,
            records_per_face=observation.records_per_face,
            signature=list(observation.signature),
            sample_pages=[page_number] if page_number is not None else [],
            mean_confidence=observation.confidence,
            candidate=True,
        )
        self.templates.append(template)
        self.save()
        return template

    def match(self, observation: LayoutObservation) -> tuple[LayoutTemplate | None, float]:
        candidates = [
            (template, _similaridade(observation.signature, template.signature))
            for template in self.templates
            if template.records_per_face == observation.records_per_face
        ]
        if not candidates:
            return None, 0.0
        return max(candidates, key=lambda item: item[1])

    def register_sample(self, template: LayoutTemplate, page_number: int | None) -> None:
        if page_number is not None and page_number not in template.sample_pages:
            template.sample_pages.append(page_number)
        # Um template de livro novo só fica ativo depois das cinco faces
        # representativas exigidas pela calibração humana.
        template.candidate = len(template.sample_pages) < 5
        self.save()


class AdaptiveLayoutDetector:
    """Classifica cada face e cria candidatos quando a estrutura muda."""

    AUTO_THRESHOLD = 0.95
    REVIEW_THRESHOLD = 0.80
    STRUCTURE_MIN = 0.72

    def __init__(self, store_path: Path | None = None) -> None:
        self.store = LayoutStore(store_path) if store_path else None

    @staticmethod
    def _signature(image: np.ndarray) -> tuple[float, ...]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        width = min(800, gray.shape[1])
        scale = width / max(1, gray.shape[1])
        small = cv2.resize(
            gray,
            (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        dark = (small < 170).astype(np.uint8) * 255
        horizontal = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(small.shape[1] * 0.28)), 1)),
        )
        vertical = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(small.shape[0] * 0.18)))),
        )
        rows = cv2.resize((horizontal > 0).mean(axis=1).reshape(-1, 1), (1, 32), interpolation=cv2.INTER_AREA).reshape(-1)
        cols = cv2.resize((vertical > 0).mean(axis=0).reshape(1, -1), (32, 1), interpolation=cv2.INTER_AREA).reshape(-1)
        ink_rows = cv2.resize((dark > 0).mean(axis=1).reshape(-1, 1), (1, 32), interpolation=cv2.INTER_AREA).reshape(-1)
        return tuple(float(value) for value in np.concatenate((rows, cols, ink_rows)))

    @classmethod
    def observar(cls, image: np.ndarray, page_number: int | None = None) -> LayoutObservation:
        if image is None or image.size == 0:
            raise ValueError("Imagem vazia")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        width = min(800, gray.shape[1])
        scale = width / max(1, gray.shape[1])
        small = cv2.resize(
            gray,
            (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        dark = (small < 170).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(30, int(small.shape[1] * 0.30)), 1)
        )
        horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
        strength = (horizontal > 0).mean(axis=1)
        start = int(len(strength) * 0.35)
        end = int(len(strength) * 0.65)
        center = strength[start:end]
        peak_index = int(np.argmax(center)) + start if len(center) else -1
        peak = float(strength[peak_index]) if peak_index >= 0 else 0.0
        background = float(np.median(strength)) if len(strength) else 0.0
        ratio = peak / max(background, 0.05)
        is_two = (
            peak >= 0.55
            and ratio >= 1.35
            and 0.40 <= peak_index / max(1, len(strength)) <= 0.60
        )
        records = 2 if is_two else 1
        separator_y = peak_index / max(1, len(strength)) if is_two else None

        if is_two:
            structure_confidence = min(0.99, 0.70 + min(0.29, peak * 0.28))
            reason = "separador_horizontal_central_confirmado"
        elif (peak < 0.48 or ratio < 1.15) and (
            float(np.max(strength)) if len(strength) else 0.0
        ) >= 0.25:
            structure_confidence = 0.86
            reason = "ausencia_de_separador_central; face_de_um_assento"
        else:
            structure_confidence = 0.66
            reason = "estrutura_central_ambigua"

        if records == 2:
            record_bboxes = (bbox_registro(0, 2), bbox_registro(1, 2))
            name_bboxes = (bbox_linha_nome(0, 2), bbox_linha_nome(1, 2))
            term_bboxes = (bbox_numero_termo(0, 2), bbox_numero_termo(1, 2))
        else:
            record_bboxes = (bbox_registro(0, 1),)
            name_bboxes = (bbox_linha_nome(0, 1),)
            term_bboxes = (bbox_numero_termo(0, 1),)

        return LayoutObservation(
            layout_id=None,
            records_per_face=records,
            confidence=structure_confidence,
            method="opencv_linhas_horizontais",
            needs_review=structure_confidence < cls.STRUCTURE_MIN,
            reason=reason,
            separator_y=separator_y,
            record_bboxes=record_bboxes,
            name_bboxes=name_bboxes,
            term_bboxes=term_bboxes,
            signature=cls._signature(image),
            page_number=page_number,
        )

    def classificar(
        self,
        image: np.ndarray,
        *,
        page_number: int | None = None,
        expected_records: int | None = None,
        force_expected: bool = False,
    ) -> LayoutObservation:
        observation = self.observar(image, page_number=page_number)
        if expected_records and observation.records_per_face != int(expected_records):
            detected = observation.records_per_face
            values = {
                **observation.__dict__,
                "needs_review": True,
                "reason": f"quantidade_detectada_{detected}_diferente_do_padrao_{expected_records}",
            }
            if force_expected:
                total = max(1, int(expected_records))
                values.update({
                    "records_per_face": total,
                    "separator_y": 0.488 if total == 2 else None,
                    "record_bboxes": tuple(bbox_registro(i, total) for i in range(total)),
                    "name_bboxes": tuple(bbox_linha_nome(i, total) for i in range(total)),
                    "term_bboxes": tuple(bbox_numero_termo(i, total) for i in range(total)),
                    "reason": (
                        f"calibracao_humana_forcou_{total}_registros; detector_observou_{detected}"
                    ),
                })
            observation = LayoutObservation(**values)
        if observation.confidence < self.STRUCTURE_MIN:
            # Uma divergencia com o padrao configurado merece ser registrada
            # como candidato, mas nao pode alterar a contagem usada pela
            # sequencia enquanto a estrutura permanecer ambigua.
            if self.store is not None and expected_records and observation.records_per_face != int(expected_records):
                template = self.store.add(observation, page_number)
                return LayoutObservation(
                    **{**observation.__dict__, "layout_id": template.layout_id,
                       "reason": "estrutura_ambigua; candidato_para_validacao_manual"}
                )
            return observation
        if self.store is None:
            return observation

        template, similarity = self.store.match(observation)
        if template is None:
            template = self.store.add(observation, page_number)
            return LayoutObservation(
                **{**observation.__dict__, "layout_id": template.layout_id, "needs_review": True,
                   "reason": "novo_layout_candidato; validar_com_paginas_seguidas"}
            )
        if similarity >= self.AUTO_THRESHOLD:
            self.store.register_sample(template, page_number)
            confirmado = not template.candidate
            return LayoutObservation(
                **{**observation.__dict__, "layout_id": template.layout_id,
                   "confidence": min(observation.confidence, similarity),
                   "needs_review": observation.needs_review or not confirmado,
                   "reason": "template_confirmado" if confirmado else "template_candidato_confirmacao"}
            )
        if similarity >= self.REVIEW_THRESHOLD:
            self.store.register_sample(template, page_number)
            return LayoutObservation(
                **{**observation.__dict__, "layout_id": template.layout_id,
                   "confidence": min(observation.confidence, similarity), "needs_review": True,
                   "reason": "template_com_confianca_media"}
            )
        novo = self.store.add(observation, page_number)
        return LayoutObservation(
            **{**observation.__dict__, "layout_id": novo.layout_id, "needs_review": True,
               "reason": "layout_desconhecido; novo_template_candidato"}
        )


def _pixels(bbox: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(bbox[0] * width)))
    y1 = max(0, min(height - 1, int(bbox[1] * height)))
    x2 = max(x1 + 1, min(width, int(bbox[2] * width)))
    y2 = max(y1 + 1, min(height, int(bbox[3] * height)))
    return x1, y1, x2, y2


def desenhar_diagnostico(image: np.ndarray, observation: LayoutObservation) -> np.ndarray:
    """Desenha caixas candidatas sem alterar a fotografia original."""
    output = image.copy()
    height, width = output.shape[:2]
    if observation.separator_y is not None:
        y = int(observation.separator_y * height)
        cv2.line(output, (0, y), (width, y), (0, 165, 255), max(2, width // 1200))
    for index, bbox in enumerate(observation.record_bboxes):
        x1, y1, x2, y2 = _pixels(bbox, width, height)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), max(2, width // 1000))
        for label, field_bbox, color in (
            ("nome", observation.name_bboxes[index], (255, 0, 0)),
            ("termo", observation.term_bboxes[index], (0, 180, 0)),
        ):
            fx1, fy1, fx2, fy2 = _pixels(field_bbox, width, height)
            cv2.rectangle(output, (fx1, fy1), (fx2, fy2), color, max(2, width // 1200))
            cv2.putText(output, f"{index + 1}:{label}", (fx1, max(18, fy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    texto = f"{observation.layout_id or 'layout_desconhecido'} | {observation.records_per_face} registro(s) | {observation.confidence:.2f}"
    cv2.putText(output, texto, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255) if observation.needs_review else (0, 130, 0), 2, cv2.LINE_AA)
    return output


def diagnosticos_da_observacao(image: np.ndarray, observation: LayoutObservation, output_dir: Path, stem: str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    debug_dir = output_dir / "debug"
    crops_dir = output_dir / "crops"
    debug_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{stem}_layout.jpg"
    cv2.imwrite(str(debug_path), desenhar_diagnostico(image, observation), [cv2.IMWRITE_JPEG_QUALITY, 95])
    crops: list[dict[str, str]] = []
    for index, (name_bbox, term_bbox) in enumerate(zip(observation.name_bboxes, observation.term_bboxes), start=1):
        name_path = crops_dir / f"{stem}_registro_{index:02d}_nome.jpg"
        term_path = crops_dir / f"{stem}_registro_{index:02d}_termo.jpg"
        cv2.imwrite(str(name_path), recortar_bbox(image, name_bbox), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(term_path), recortar_bbox(image, term_bbox), [cv2.IMWRITE_JPEG_QUALITY, 95])
        crops.append({"registro": str(index), "nome": str(name_path), "termo": str(term_path)})
    return {"debug": str(debug_path), "crops": crops}
