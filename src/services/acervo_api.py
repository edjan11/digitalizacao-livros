"""API HTTP local, somente leitura, para integrar o acervo a outros sistemas.

O servidor deliberadamente escuta em 127.0.0.1 por padrão. Assim o balcão de
consulta pode fornecer JSON e a fotografia original sem duplicar arquivos ou
abrir permissões de rede sem o operador perceber.
"""

from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class _AcervoHandler(BaseHTTPRequestHandler):
    server_version = "DigitalizadorLivrosAPI/1.0"

    def log_message(self, _format: str, *_args) -> None:
        # A consulta não deve poluir o console/log a cada miniatura acessada.
        return

    @property
    def acervo(self):
        return self.server.acervo_api  # type: ignore[attr-defined]

    def _responder_json(self, payload: object, status: int = 200) -> None:
        dados = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def _responder_imagem(self, arquivo: Path, dados: bytes) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(str(arquivo))[0] or "application/octet-stream",
        )
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def _responder_erro(self, mensagem: str, status: int = 404) -> None:
        self._responder_json({"erro": mensagem}, status)

    def _registro_por_id(self, registro_id: int) -> dict | None:
        row = self.acervo.repo.db.fetchone(
            """
            SELECT r.id AS registro_id, r.termo, r.folha, r.face,
                   r.indice_na_imagem, r.status AS registro_status,
                   i.id AS imagem_id, i.caminho_original, i.caminho_thumb,
                   i.sha256,
                   i.qualidade_status, i.termo_inicial, i.termo_final,
                   i.rotacao_visualizacao, i.tipo_documento,
                   l.id AS livro_id, l.codigo AS livro_codigo,
                   l.nome_capa AS livro_nome, a.nome AS acervo_nome,
                   o.nome AS oficio_nome, t.nome AS tipo_nome
            FROM registro r
            JOIN imagem i ON i.id=r.imagem_id
            JOIN livro l ON l.id=r.livro_id
            LEFT JOIN acervo a ON a.id=l.acervo_id
            JOIN oficio o ON o.id=l.oficio_id
            JOIN tipo_registro t ON t.id=l.tipo_id
            WHERE r.id=?
            """,
            (registro_id,),
        )
        if row:
            row["metadados"] = self.acervo.repo.listar_metadados_registro(registro_id)
        return row

    def _registro_publico(self, row: dict) -> dict:
        item = dict(row)
        imagem_id = item.get("imagem_id")
        registro_id = item.get("registro_id")
        caminho = str(item.get("caminho_original") or "")
        arquivo = Path(caminho) if caminho else None
        foto_url = f"{self.acervo.base_url}/api/v1/imagens/{imagem_id}"
        # Campos legados continuam no topo; o bloco imagem facilita integrar
        # a foto ao lado de uma tela de lavratura de segunda via.
        item["foto_url"] = foto_url
        item["imagem_url"] = foto_url
        item["caminho_imagem"] = caminho
        item["imagem_nome"] = arquivo.name if arquivo else ""
        item["imagem_mime"] = (
            mimetypes.guess_type(caminho)[0] if caminho else None
        ) or "application/octet-stream"
        item["imagem_tamanho_bytes"] = (
            arquivo.stat().st_size if arquivo and arquivo.is_file() else None
        )
        item["imagem"] = {
            "id": imagem_id,
            "url": foto_url,
            "path": caminho,
            "nome": item["imagem_nome"],
            "mime": item["imagem_mime"],
            "tamanho_bytes": item["imagem_tamanho_bytes"],
            "sha256": item.get("sha256"),
            "rotacao_visualizacao": item.get("rotacao_visualizacao") or 0,
        }
        item["registro_url"] = f"{self.acervo.base_url}/api/v1/registros/{registro_id}"
        if "metadados" not in item and registro_id:
            item["metadados"] = self.acervo.repo.listar_metadados_registro(registro_id)
        metadados = list(item.get("metadados") or [])
        nomes = [
            meta for meta in metadados
            if meta.get("tipo") == "nome_registrado" and meta.get("ativo", 1)
            and meta.get("status") not in {"superado", "descartado"}
        ]
        confirmados = [
            meta for meta in nomes
            if meta.get("status") in {"confirmado", "corrigido"}
        ]
        sugestoes = [meta for meta in nomes if meta not in confirmados]
        confirmados.sort(key=lambda meta: int(meta.get("id") or 0), reverse=True)
        prioridade = {"qwen_nome_correcao": 0, "qwen_registro": 1, "ocr_nome_rapido": 2}
        sugestoes.sort(
            key=lambda meta: (
                prioridade.get(str(meta.get("fonte") or ""), 9),
                -float(meta.get("confianca") or 0),
                -int(meta.get("id") or 0),
            )
        )
        confirmado = str(item.get("nome_confirmado") or "")
        sugerido = str(item.get("nome_sugerido") or "")
        escolhido = None
        if not confirmado and confirmados:
            escolhido = confirmados[0]
            confirmado = str(escolhido.get("valor_tratado") or "")
        if not sugerido and sugestoes:
            sugerido = str(sugestoes[0].get("valor_tratado") or "")
        escolhido = escolhido or (confirmados[0] if confirmados else (sugestoes[0] if sugestoes else None))
        item["nome_confirmado"] = confirmado
        item["nome_sugerido"] = sugerido
        item["nome_registrado"] = confirmado or sugerido
        item["nome_status"] = (
            escolhido.get("status") if escolhido else item.get("nome_status") or ""
        )
        item["nome_confianca"] = float(
            escolhido.get("confianca") if escolhido else item.get("nome_confianca") or 0
        )
        item["nome_fonte"] = (
            escolhido.get("fonte") if escolhido else item.get("nome_fonte") or ""
        )
        item["nome_eh_confirmado"] = bool(confirmado)
        return item

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        caminho = parsed.path.rstrip("/")
        if caminho == "/api/v1/health":
            self._responder_json({"ok": True, "servico": "consulta-acervo"})
            return
        if caminho == "/api/v1/registros":
            query = parse_qs(parsed.query)
            texto = query.get("texto", query.get("q", [""]))[0]
            termo_raw = query.get("termo", [""])[0]
            livro_raw = query.get("livro_id", [""])[0]
            acervo_raw = query.get("acervo_id", [""])[0]
            oficio_raw = query.get("oficio_id", [""])[0]
            limite_raw = query.get("limite", ["100"])[0]
            try:
                termo = int(termo_raw) if termo_raw else None
                livro_id = int(livro_raw) if livro_raw else None
                acervo_id = int(acervo_raw) if acervo_raw else None
                oficio_id = int(oficio_raw) if oficio_raw else None
                limite = max(1, min(2000, int(limite_raw)))
            except ValueError:
                self._responder_erro("filtros numéricos inválidos", 400)
                return
            rows = self.acervo.repo.buscar_registros(
                texto=texto,
                termo=termo,
                livro_id=livro_id,
                acervo_id=acervo_id,
                oficio_id=oficio_id,
                limite=limite,
            )
            self._responder_json({
                "total": len(rows),
                "itens": [self._registro_publico(row) for row in rows],
            })
            return
        if caminho.startswith("/api/v1/registros/"):
            try:
                registro_id = int(caminho.rsplit("/", 1)[1])
            except ValueError:
                self._responder_erro("registro inválido", 400)
                return
            row = self._registro_por_id(registro_id)
            if row is None:
                self._responder_erro("registro não encontrado")
            else:
                self._responder_json(self._registro_publico(row))
            return
        if caminho.startswith("/api/v1/imagens/"):
            try:
                imagem_id = int(caminho.rsplit("/", 1)[1])
            except ValueError:
                self._responder_erro("imagem inválida", 400)
                return
            row = self.acervo.repo.db.fetchone(
                "SELECT caminho_original FROM imagem WHERE id=?", (imagem_id,)
            )
            arquivo = Path(row.get("caminho_original") or "") if row else None
            if arquivo is None or not arquivo.is_file():
                self._responder_erro("arquivo original não encontrado")
                return
            # O caminho só pode vir da coluna imagem.caminho_original; não há
            # parâmetro de arquivo livre e, portanto, não há travessia de pasta.
            dados = arquivo.read_bytes()
            self._responder_imagem(arquivo, dados)
            return
        self._responder_erro("rota não encontrada")


    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


class AcervoApiServer:
    """Servidor local controlado pelo ciclo de vida da janela de consulta."""

    def __init__(self, repo, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.repo = repo
        self.host = host
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host = "localhost" if self.host == "127.0.0.1" else self.host
        return f"http://{host}:{self.port}"

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self) -> None:
        if self._server is not None:
            return
        server = ThreadingHTTPServer((self.host, self.port), _AcervoHandler)
        server.daemon_threads = True
        server.acervo_api = self  # type: ignore[attr-defined]
        self._server = server
        self.port = int(server.server_address[1])
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="acervo-api",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
