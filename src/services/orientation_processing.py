from __future__ import annotations

"""Orientacao retomavel por coorte para livros fotografados em sequencia.

O detector individual continua fornecendo a evidencia, mas uma resposta fraca
de 180 graus nao pode inverter sozinha uma foto de uma mesma sessao de captura.
Originais nunca sao regravados: apenas a rotacao logica e a evidencia ficam no
banco.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..database.repository import Repository
from ..imaging.page_orientation import OrientationDetector, OrientationResult


@dataclass(frozen=True)
class CohortDecision:
    rotation: int
    confidence: float
    agreement: float
    approved: bool
    reason: str


def decidir_orientacao_coorte(results: list[OrientationResult]) -> CohortDecision:
    if not results:
        return CohortDecision(0, 0.0, 0.0, False, "coorte sem amostras")
    votes = defaultdict(float)
    for result in results:
        confidence = float(result.confidence or 0.0)
        weight = 4.0 if confidence >= 0.85 else 2.0 if confidence >= 0.60 else 0.25
        votes[int(result.rotation) % 360] += weight
    rotation = max(votes, key=votes.get)
    total = sum(votes.values()) or 1.0
    agreement = votes[rotation] / total
    strong = sum(
        1 for result in results
        if int(result.rotation) % 360 == rotation and float(result.confidence or 0) >= 0.60
    )
    conflict = any(
        int(result.rotation) % 360 != rotation and float(result.confidence or 0) >= 0.85
        for result in results
    )
    approved = agreement >= 0.85 and strong >= 2 and not conflict
    confidence = min(0.99, 0.85 + 0.14 * agreement) if approved else min(0.84, agreement)
    observations = ", ".join(
        f"{int(result.rotation) % 360}g/{float(result.confidence):.2f}"
        for result in results
    )
    reason = (
        f"coorte: {len(results)} amostras; acordo ponderado={agreement:.1%}; "
        f"fortes={strong}; resultados=[{observations}]"
    )
    if conflict:
        reason += "; conflito individual >=85%"
    return CohortDecision(rotation, confidence, agreement, approved, reason)


def _read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _sample_rows(rows: list[dict], sample_count: int, interval: int) -> list[dict]:
    if not rows:
        return []
    count = max(1, min(int(sample_count), len(rows)))
    if count == 1:
        indices = {0}
    else:
        indices = {
            round(position * (len(rows) - 1) / (count - 1))
            for position in range(count)
        }
    if interval > 0:
        indices.update(range(0, len(rows), int(interval)))
        indices.add(len(rows) - 1)
    return [rows[index] for index in sorted(indices)]


class OrientationBatchRunner:
    def __init__(
        self,
        repo: Repository,
        *,
        detector: OrientationDetector | None = None,
        sample_count: int = 5,
        validation_interval: int = 25,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self.repo = repo
        self.detector = detector or OrientationDetector()
        self.sample_count = max(3, int(sample_count))
        self.validation_interval = max(0, int(validation_interval))
        self.on_progress = on_progress or (lambda _message: None)

    def run(self, book_code: str, *, apply: bool = False) -> dict:
        book = self.repo.db.fetchone(
            "SELECT id,codigo FROM livro WHERE upper(codigo)=upper(?)", (book_code,)
        )
        if not book:
            raise RuntimeError(f"livro {book_code!r} nao encontrado")
        rows = self.repo.db.fetchall(
            """
            SELECT i.id AS imagem_id, i.caminho_original, i.sha256, i.face,
                   i.folha_estimada, ii.id AS importacao_item_id
            FROM imagem i
            LEFT JOIN importacao_item ii ON ii.imagem_id=i.id
            WHERE i.livro_id=? AND COALESCE(i.tipo_documento,'registro')='registro'
              AND COALESCE(i.duplicidade_status,'')!='duplicata_confirmada'
            ORDER BY lower(COALESCE(i.face,'')), i.folha_estimada, i.id
            """,
            (int(book["id"]),),
        )
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            path = Path(row.get("caminho_original") or "")
            groups[(str(row.get("face") or "indeterminado"), str(path.parent).casefold())].append(row)

        decisions = []
        approved_image_ids: list[int] = []
        import_item_ids: list[int] = []
        for (face, folder), group in groups.items():
            samples = _sample_rows(group, self.sample_count, self.validation_interval)
            evidence: list[OrientationResult] = []
            errors: list[str] = []
            for position, row in enumerate(samples, 1):
                path = Path(row.get("caminho_original") or "")
                self.on_progress(f"{face}: amostra {position}/{len(samples)} - {path.name}")
                image = _read_image(path)
                if image is None:
                    errors.append(f"{path.name}: imagem invalida")
                    continue
                evidence.append(self.detector.detect(image))
            decision = decidir_orientacao_coorte(evidence)
            if errors:
                decision = CohortDecision(
                    decision.rotation, min(decision.confidence, 0.84), decision.agreement,
                    False, decision.reason + "; " + "; ".join(errors),
                )
            decisions.append({
                "face": face,
                "folder": folder,
                "images": len(group),
                "samples": len(samples),
                **asdict(decision),
            })
            if not (apply and decision.approved):
                continue
            approved_image_ids.extend(int(row["imagem_id"]) for row in group)
            import_item_ids.extend(
                int(row["importacao_item_id"])
                for row in group if row.get("importacao_item_id") is not None
            )
            normalization = json.dumps({
                "status": "orientacao_coorte_aprovada",
                "rotation": decision.rotation,
                "confidence": decision.confidence,
                "agreement": decision.agreement,
                "original_immutable": True,
            }, ensure_ascii=False)
            self.repo.db.executemany(
                """
                UPDATE imagem
                SET rotacao_visualizacao=?, orientacao_confianca=?,
                    orientacao_metodo='coorte_linhas_rapidocr_v1',
                    orientacao_motivo=?, normalizacao_json=?
                WHERE id=?
                """,
                [
                    (
                        decision.rotation, decision.confidence, decision.reason,
                        normalization, int(row["imagem_id"]),
                    )
                    for row in group
                ],
            )

        if apply and approved_image_ids:
            self.repo.db.executemany(
                """
                UPDATE processamento_item
                SET status='pendente', erro=NULL, iniciado_em=NULL,
                    concluido_em=NULL, updated_at=datetime('now')
                WHERE imagem_id=? AND etapa='ocr_nome_rapido'
                  AND status IN ('pausado','pendente')
                """,
                [(image_id,) for image_id in approved_image_ids],
            )
            self.repo.db.executemany(
                """
                UPDATE importacao_item
                SET orientacao=(SELECT rotacao_visualizacao FROM imagem WHERE id=importacao_item.imagem_id),
                    orientacao_confianca=(SELECT orientacao_confianca FROM imagem WHERE id=importacao_item.imagem_id),
                    orientacao_metodo='coorte_linhas_rapidocr_v1', status='orientado_coorte',
                    erro=NULL, updated_at=datetime('now')
                WHERE id=?
                """,
                [(item_id,) for item_id in import_item_ids],
            )
            self.repo.db.executemany(
                "UPDATE revisao SET resolvida=1 WHERE imagem_id=? AND tipo='orientacao_incerta'",
                [(image_id,) for image_id in approved_image_ids],
            )
            self.repo.db.executemany(
                """
                UPDATE imagem SET precisa_revisao=CASE WHEN EXISTS (
                    SELECT 1 FROM revisao r WHERE r.imagem_id=imagem.id AND r.resolvida=0
                ) THEN 1 ELSE 0 END WHERE id=?
                """,
                [(image_id,) for image_id in approved_image_ids],
            )
        return {
            "book": str(book["codigo"]),
            "apply": bool(apply),
            "images": len(rows),
            "approved_images": len(approved_image_ids),
            "groups": decisions,
        }
