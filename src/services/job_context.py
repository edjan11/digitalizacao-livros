from __future__ import annotations

from dataclasses import asdict, dataclass

from ..imaging.record_regions import (
    BBox,
    bbox_contido_no_registro,
    bbox_corresponde_registro,
)


@dataclass(frozen=True)
class QwenJobContext:
    registro_id: int
    imagem_id: int
    termo: int | None
    indice_na_imagem: int
    total_na_imagem: int
    bbox: BBox
    imagem_sha256: str
    tipo: str

    @classmethod
    def from_registro(
        cls,
        registro: dict,
        bbox,
        *,
        total: int,
        tipo: str,
    ) -> "QwenJobContext":
        return cls(
            registro_id=int(registro["registro_id"]),
            imagem_id=int(registro["imagem_id"]),
            termo=int(registro["termo"]) if registro.get("termo") is not None else None,
            indice_na_imagem=int(registro.get("indice_na_imagem") or 0),
            total_na_imagem=max(1, int(total)),
            bbox=tuple(float(valor) for valor in bbox),
            imagem_sha256=str(registro.get("sha256") or ""),
            tipo=tipo,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def validar_contexto_qwen(repo, contexto: QwenJobContext) -> dict | None:
    """Confere novamente a identidade antes de persistir uma inferência lenta."""
    registro = repo.db.fetchone(
        """
        SELECT r.id AS registro_id, r.termo, r.indice_na_imagem,
               i.id AS imagem_id, i.sha256,
               (SELECT COUNT(*) FROM registro rr WHERE rr.imagem_id=i.id) AS total
        FROM registro r JOIN imagem i ON i.id=r.imagem_id
        WHERE r.id=?
        """,
        (contexto.registro_id,),
    )
    if not registro or int(registro["imagem_id"]) != contexto.imagem_id:
        return None
    if contexto.imagem_sha256 and str(registro.get("sha256") or "") != contexto.imagem_sha256:
        return None
    if int(registro.get("indice_na_imagem") or 0) != contexto.indice_na_imagem:
        return None
    if int(registro.get("total") or 1) != contexto.total_na_imagem:
        return None
    valido = (
        bbox_corresponde_registro(
            contexto.bbox, contexto.indice_na_imagem, contexto.total_na_imagem
        )
        if contexto.tipo == "registro"
        else bbox_contido_no_registro(
            contexto.bbox, contexto.indice_na_imagem, contexto.total_na_imagem
        )
    )
    return registro if valido else None
