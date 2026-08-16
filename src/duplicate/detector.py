from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

from .hashing import compute_phash, compute_dhash, hash_distance
from .comparator import compute_registered_similarity

logger = logging.getLogger(__name__)

PHASH_THRESHOLD = 5
DHASH_THRESHOLD = 8
SSIM_THRESHOLD = 0.95
SSIM_SUSPECT = 0.90
PHASH_CANDIDATE = 8
DHASH_CANDIDATE = 6
MAX_RECENT_IMAGES = 2


def detectar_duplicidade(
    imagem_atual: np.ndarray,
    imagens_do_livro: list[dict],
    image_loader: Callable[[str], np.ndarray | None],
    phash_threshold: int = PHASH_THRESHOLD,
    dhash_threshold: int = DHASH_THRESHOLD,
    ssim_threshold: float = SSIM_THRESHOLD,
    ssim_suspect: float = SSIM_SUSPECT,
    sha256_atual: str | None = None,
    phash_atual: str | None = None,
    dhash_atual: str | None = None,
) -> dict:
    if imagem_atual is None or not imagens_do_livro:
        return {"status": "unico", "confianca": 0.0, "duplicata_de": None}

    phash_atual = phash_atual or compute_phash(imagem_atual)
    dhash_atual = dhash_atual or compute_dhash(imagem_atual)

    if not phash_atual:
        return {"status": "erro", "confianca": 0.0, "duplicata_de": None}

    if sha256_atual:
        for img_data in imagens_do_livro:
            if img_data.get("sha256") == sha256_atual:
                return {
                    "status": "duplicata_confirmada",
                    "confianca": 1.0,
                    "duplicata_de": img_data["id"],
                    "criterio": "arquivo_identico",
                }

    # Recapturas visuais normalmente acontecem logo depois da primeira foto.
    # Limitar a janela evita comparar centenas de formularios quase iguais e
    # mantem a verificacao rapida durante a captura.
    suspeitos = []
    for img_data in imagens_do_livro[-MAX_RECENT_IMAGES:]:
        phash_existente = img_data.get("hash_perceptual", "")
        if not phash_existente:
            continue
        ph_dist = hash_distance(phash_atual, phash_existente)
        dhash_existente = img_data.get("dhash", "")
        dh_dist = hash_distance(dhash_atual, dhash_existente) if dhash_existente else 0
        hash_normal = ph_dist <= max(phash_threshold, PHASH_CANDIDATE) and dh_dist <= max(
            dhash_threshold, DHASH_CANDIDATE
        )
        # Uma mao grande muda bastante o pHash. Ainda assim, a proxima foto
        # limpa pode ser uma refotografia da mesma pagina.
        anterior_ocluida = float(img_data.get("qualidade_oclusao") or 0) >= 0.20
        hash_com_oclusao = anterior_ocluida and ph_dist <= 18 and dh_dist <= 10
        if hash_normal or hash_com_oclusao:
            suspeitos.append((img_data, ph_dist, dh_dist))

    if not suspeitos:
        return {"status": "unico", "confianca": 1.0, "duplicata_de": None}

    suspeitos.sort(key=lambda x: (x[1] + x[2], x[1]))

    melhor_possivel = None
    for img_data, ph_dist, dh_dist in suspeitos:
        caminho = img_data.get("caminho_original", "")
        if caminho:
            img_existente = image_loader(caminho)
            if img_existente is not None:
                metricas = compute_registered_similarity(imagem_atual, img_existente)
                ssim_val = metricas["ssim"]
                alinhamento = metricas["alinhamento"]
                tinta = metricas["tinta"]

                anterior_ocluida = float(img_data.get("qualidade_oclusao") or 0) >= 0.20
                if anterior_ocluida:
                    atual_dir = imagem_atual[:, int(imagem_atual.shape[1] * 0.42):]
                    anterior_dir = img_existente[:, int(img_existente.shape[1] * 0.42):]
                    parcial = compute_registered_similarity(atual_dir, anterior_dir)
                    if (
                        parcial["ssim"] >= 0.75
                        and parcial["alinhamento"] >= 0.963
                        and parcial["tinta"] >= 0.865
                    ):
                        return {
                            "status": "duplicata_confirmada",
                            "confianca": max(parcial["ssim"], parcial["tinta"]),
                            "duplicata_de": img_data["id"],
                            "ssim": parcial["ssim"],
                            "similaridade_tinta": parcial["tinta"],
                            "alinhamento": parcial["alinhamento"],
                            "phash_dist": ph_dist,
                            "dhash_dist": dh_dist,
                            "criterio": "regiao_sem_oclusao",
                        }

                confirmada = ssim_val >= 0.94 or (alinhamento >= 0.975 and tinta >= 0.89)
                possivel = alinhamento >= 0.96 and tinta >= 0.878
                if confirmada:
                    return {
                        "status": "duplicata_confirmada",
                        "confianca": max(ssim_val, tinta),
                        "duplicata_de": img_data["id"],
                        "ssim": ssim_val,
                        "similaridade_tinta": tinta,
                        "alinhamento": alinhamento,
                        "phash_dist": ph_dist,
                        "dhash_dist": dh_dist,
                    }
                if possivel:
                    candidato = {
                        "status": "possivel_duplicata",
                        "confianca": max(ssim_val, tinta),
                        "duplicata_de": img_data["id"],
                        "ssim": ssim_val,
                        "similaridade_tinta": tinta,
                        "alinhamento": alinhamento,
                        "phash_dist": ph_dist,
                        "dhash_dist": dh_dist,
                    }
                    if not melhor_possivel or candidato["confianca"] > melhor_possivel["confianca"]:
                        melhor_possivel = candidato
            else:
                candidato = {
                    "status": "possivel_duplicata",
                    "confianca": 1.0 - (ph_dist / 64.0),
                    "duplicata_de": img_data["id"],
                    "phash_dist": ph_dist,
                }
                if not melhor_possivel or candidato["confianca"] > melhor_possivel["confianca"]:
                    melhor_possivel = candidato

    if melhor_possivel:
        return melhor_possivel
    return {"status": "unico", "confianca": 0.8, "duplicata_de": None}
