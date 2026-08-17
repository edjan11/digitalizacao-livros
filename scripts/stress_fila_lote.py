"""Stress da TOPOLOGIA ATUAL de producao (M1-T04).

- 1 worker + lock exclusivo, banco temporario, carga sintetica com OCR falso
  (mede o maquinario da fila, nao a inferencia);
- escalas 1x/2x/5x/10x: throughput, backlog maximo, RSS, recuperacao;
- interrupcao no meio + retomada: zero duplicados/perdidos/associacao errada;
- 2 workers no MESMO banco: segundo bloqueado pelo lock (experimento isolado,
  nao muda a regra de producao).

Uso: .venv\\Scripts\\python.exe scripts\\stress_fila_lote.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import Settings
from src.database.connection import Database
from src.database.repository import Repository
from src.ocr.base import OCRResult
from src.services.name_processing import NameBatchRunner, _exclusive_worker_lock

IMAGENS_BASE = 5  # 10 assentos por escala 1x


class ProvedorFalso:
    def __init__(self, name, atraso_s=0.0):
        self.name = name
        self.atraso_s = atraso_s

    def recognize(self, _imagem, fast=True):
        if self.atraso_s:
            time.sleep(self.atraso_s)
        return OCRResult(
            motor=self.name,
            texto_bruto="que recebeu o nome de Ana Beatriz de Souza",
            tempo_ms=2.0,
        )


QWEN_ATRASO_S = 0.05


class QwenFalso:
    def __init__(self, **_kwargs):
        pass

    def analisar_nome(self, _imagem):
        time.sleep(QWEN_ATRASO_S)  # simula o custo de inferencia p/ medir backlog
        return "Ana Beatriz de Souza", OCRResult(
            motor="qwen-falso", texto_bruto="Ana Beatriz de Souza", tempo_ms=50.0
        )

    def liberar(self):
        pass


def _geometria_falsa(image):
    return SimpleNamespace(
        image=image, confidence=1.0, left_line=None, right_line=None,
        reason="geometria simulada",
    )


def _mockar(atraso_s=0.0):
    import src.services.name_processing as np_
    np_._provedores = lambda _s: [ProvedorFalso("ocr-falso-a", atraso_s), ProvedorFalso("ocr-falso-b", atraso_s)]
    np_.retificar_formulario = _geometria_falsa
    np_._localizar_linhas_nome_pagina = lambda _i, _p, total: {
        indice: ((0.28, (0.004 + indice * 0.49), 0.76, 0.496 + indice * 0.49), "rotulo")
        for indice in range(total)
    }
    np_.modelo_qwen_instalado = lambda _p=None: True
    np_.QwenRecordAnalyzer = QwenFalso
    return np_


def _criar_acervo(base: Path, n_imagens: int) -> tuple[Path, int]:
    db = Database(base / "stress.db")
    db.connect()
    repo = Repository(db)
    livro_id = repo.criar_livro(
        oficio_id=6, tipo_id=1, codigo="A-07", nome_capa="Nascimentos",
        registros_por_face=2, termo_inicial=6801,
        termo_final=6801 + 2 * n_imagens - 1,
    )
    for indice in range(n_imagens):
        path = base / f"pagina_{indice}.jpg"
        imagem = np.full((600, 400, 3), 235, np.uint8)
        for y in range(40, 580, 25):
            cv2.line(imagem, (20, y), (380, y), (50, 50, 50), 1)
        cv2.imwrite(str(path), imagem)
        termo_a = 6801 + 2 * indice
        imagem_id = repo.registrar_imagem(
            livro_id=livro_id, ordem_captura=indice + 1,
            caminho_original=str(path), caminho_thumb=str(path),
            folha_estimada=indice + 1, face="frente",
            termo_inicial=termo_a, termo_final=termo_a + 1,
            duplicidade_status="unico",
        )
        repo.sincronizar_registros_imagem(imagem_id)
    db.close()
    return db.path, livro_id


def _config(settings: Settings, direto=False) -> None:
    settings.set("ocr", "name_qwen_threshold", 0.95)
    settings.set("ocr", "name_direct_qwen_books", ["A-07"] if direto else [])
    settings.set("telemetry", "enabled", False)


def _medir_backlog(db_path: Path, lote_id: int, parar: threading.Event, medidas: dict) -> None:
    """Amostra a fila a cada 50 ms; registra maximo em ``medidas``."""
    db = Database(db_path)
    db.connect()
    repo = Repository(db)
    maximo = 0
    amostras = 0
    while not parar.is_set():
        try:
            linha = repo.db.fetchone(
                """
                SELECT COUNT(*) AS n FROM processamento_item
                WHERE lote_id=? AND status IN ('pendente','processando')
                """,
                (int(lote_id),),
            ) or {"n": 0}
            maximo = max(maximo, int(linha["n"]))
            amostras += 1
        except Exception:
            pass
        time.sleep(0.05)
    medidas["maximo"] = maximo
    medidas["amostras"] = amostras
    db.close()


def _execucoes_por_registro(db_path: Path, motor: str) -> dict[int, int]:
    db = Database(db_path)
    db.connect()
    linhas = db.fetchall(
        "SELECT registro_id, COUNT(*) AS n FROM ocr_execucao "
        "WHERE motor=? AND registro_id IS NOT NULL GROUP BY registro_id",
        (motor,),
    )
    db.close()
    return {int(l["registro_id"]): int(l["n"]) for l in linhas}


def escala(n_imagens: int, nome: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="stress_") as tmp:
        base = Path(tmp)
        db_path, livro_id = _criar_acervo(base, n_imagens)
        _mockar()
        settings = Settings(base / "config.yaml")
        _config(settings)
        db = Database(db_path)
        db.connect()
        lote = Repository(db).criar_ou_sincronizar_lote_nomes(livro_id)
        total_itens = len(Repository(db).listar_itens_processamento(
            int(lote["id"]), etapa="ocr_nome_rapido"
        ))
        db.close()

        antes = psutil.Process().memory_info().rss / 1024 / 1024
        inicio = time.perf_counter()
        parar_backlog = threading.Event()
        medidas: dict = {"maximo": 0, "amostras": 0}
        medidor = threading.Thread(
            target=_medir_backlog,
            args=(db_path, int(lote["id"]), parar_backlog, medidas),
            daemon=True,
        )
        medidor.start()
        resumo = NameBatchRunner(
            db_path=db_path, settings=settings, lote_id=int(lote["id"]),
            max_workers=1,
        ).run()
        parar_backlog.set()
        medidor.join(timeout=5)
        tempo_total = time.perf_counter() - inicio
        depois = psutil.Process().memory_info().rss / 1024 / 1024

        qwen_por_registro = _execucoes_por_registro(db_path, "qwen-nome-faixa-v3")
        itens_qwen = len(qwen_por_registro)
        return {
            "escala": nome,
            "itens": total_itens,
            "tempo_s": round(tempo_total, 2),
            "throughput_itens_s": round(total_itens / max(0.001, tempo_total), 2),
            "status": resumo["status"],
            "backlog_max": medidas["maximo"],
            "backlog_amostras": medidas["amostras"],
            "rss_inicio_mb": round(antes, 1),
            "rss_fim_mb": round(depois, 1),
            "itens_qwen": itens_qwen,
        }


def interrupcao_e_retomada() -> dict:
    with tempfile.TemporaryDirectory(prefix="stress_kill_") as tmp:
        base = Path(tmp)
        db_path, livro_id = _criar_acervo(base, IMAGENS_BASE * 2)  # 20 assentos
        _mockar()
        settings = Settings(base / "config.yaml")
        _config(settings)
        db = Database(db_path)
        db.connect()
        repo = Repository(db)
        lote = repo.criar_ou_sincronizar_lote_nomes(livro_id)
        total = len(repo.listar_itens_processamento(int(lote["id"]), etapa="ocr_nome_rapido"))
        db.close()

        contador = {"n": 0}
        alvo = int(total * 0.6)

        def parar_apos_60():
            return contador["n"] >= alvo

        def on_progress(_resumo, _rotulo):
            contador["n"] += 1

        r1 = NameBatchRunner(
            db_path=db_path, settings=settings, lote_id=int(lote["id"]),
            max_workers=1, should_stop=parar_apos_60, on_progress=on_progress,
        ).run()
        db = Database(db_path)
        db.connect()
        repo = Repository(db)
        r1_itens = repo.listar_itens_processamento(int(lote["id"]), etapa="ocr_nome_rapido")
        db.close()
        concluidos_1 = sum(1 for i in r1_itens if i["status"] == "sugestao")

        r2 = NameBatchRunner(
            db_path=db_path, settings=settings, lote_id=int(lote["id"]),
            max_workers=1,
        ).run()

        rapido = _execucoes_por_registro(db_path, "ocr-nomes-rapido-v2")
        qwen = _execucoes_por_registro(db_path, "qwen-nome-faixa-v3")
        db = Database(db_path)
        db.connect()
        repo = Repository(db)
        pendentes = repo.listar_itens_processamento(
            int(lote["id"]), etapa="ocr_nome_rapido",
            statuses=("pendente", "processando"),
        )
        qwen_pendentes = repo.listar_itens_processamento(
            int(lote["id"]), etapa="qwen_nome", statuses=("pendente", "processando"),
        )
        db.close()
        return {
            "total": total,
            "status_fase1": r1["status"],
            "concluidos_fase1": concluidos_1,
            "status_fase2": r2["status"],
            "rapido_duplicados": sum(1 for n in rapido.values() if n != 1),
            "qwen_duplicados": sum(1 for n in qwen.values() if n != 1),
            "rapido_total_execucoes": len(rapido),
            "qwen_total_execucoes": len(qwen),
            "pendentes_restantes": len(pendentes) + len(qwen_pendentes),
        }


def dois_workers_mesmo_banco() -> dict:
    with tempfile.TemporaryDirectory(prefix="stress_2w_") as tmp:
        base = Path(tmp)
        db_path, livro_id = _criar_acervo(base, IMAGENS_BASE * 4)  # 40 assentos
        _mockar(atraso_s=0.2)  # mantem o worker A ocupado tempo suficiente
        settings = Settings(base / "config.yaml")
        _config(settings)
        db = Database(db_path)
        db.connect()
        lote = Repository(db).criar_ou_sincronizar_lote_nomes(livro_id)
        db.close()

        resultado_a: dict = {}
        erro_b: dict = {}

        def executar_a():
            try:
                resultado_a["resumo"] = NameBatchRunner(
                    db_path=db_path, settings=settings, lote_id=int(lote["id"]),
                    max_workers=1,
                ).run()
            except Exception as exc:
                resultado_a["erro"] = str(exc)

        def executar_b():
            try:
                NameBatchRunner(
                    db_path=db_path, settings=settings, lote_id=int(lote["id"]),
                    max_workers=1,
                ).run()
                erro_b["rodou"] = True
            except Exception as exc:
                erro_b["erro"] = str(exc)

        thread_a = threading.Thread(target=executar_a)
        thread_b = threading.Thread(target=executar_b)
        thread_a.start()
        time.sleep(0.3)  # garante que A segurou o lock
        thread_b.start()
        thread_a.join(timeout=120)
        thread_b.join(timeout=60)
        return {
            "a_terminou": "resumo" in resultado_a,
            "b_bloqueado": bool(erro_b.get("erro")) and "trabalhador" in str(erro_b.get("erro")),
            "b_erro": str(erro_b.get("erro"))[:120],
            "b_rodou": bool(erro_b.get("rodou")),
        }


def main() -> None:
    relatorio: dict = {"escalas": [], "interrupcao": {}, "dois_workers": {}}
    for multiplicador, nome in ((1, "1x"), (2, "2x"), (5, "5x"), (10, "10x")):
        print(f"Escala {nome}...", flush=True)
        resultado = escala(IMAGENS_BASE * multiplicador, nome)
        relatorio["escalas"].append(resultado)
        print(json.dumps({k: v for k, v in resultado.items()}, ensure_ascii=False), flush=True)

    print("Interrupcao + retomada...", flush=True)
    relatorio["interrupcao"] = interrupcao_e_retomada()
    print(json.dumps(relatorio["interrupcao"], ensure_ascii=False), flush=True)

    print("Dois workers no mesmo banco...", flush=True)
    relatorio["dois_workers"] = dois_workers_mesmo_banco()
    print(json.dumps(relatorio["dois_workers"], ensure_ascii=False), flush=True)

    saida = ROOT / ".tmp_stress_fila_lote.json"
    saida.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULTADO={saida}")


if __name__ == "__main__":
    main()
