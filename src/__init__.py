"""Nucleo do Digitalizador de Livros."""

# Mantem o contrato comum do monitor no mesmo lugar usado pelo OrganizadorFirmas.
from pathlib import Path
import sys

_shared_root = Path(__file__).resolve().parents[2] / "componentes_compartilhados"
if _shared_root.exists() and str(_shared_root) not in sys.path:
    sys.path.insert(0, str(_shared_root))
