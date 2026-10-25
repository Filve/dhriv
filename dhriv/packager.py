"""
packager.py — Pacchetti portabili per macOS, Windows e Linux.

Genera uno zip autosufficiente (runtime Dhriv + programma `.dhrb` + launcher) per ciascun
sistema operativo. Richiede Python >= 3.9 sulla macchina di destinazione. NON produce
installer nativi (.dmg/.msi/.deb): sono nella roadmap.

Copyright (c) 2026 Francesco Simeoni (@Filve). Tutti i diritti riservati.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

from .compiler import verify_binary

TARGETS = ("macos", "windows", "linux", "universal")

_FIXED_DATE = (2026, 1, 1, 0, 0, 0)


def _runtime_files() -> List[Tuple[str, bytes]]:
    pkg = Path(__file__).resolve().parent
    out: List[Tuple[str, bytes]] = []
    for path in sorted(pkg.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        out.append(("dhriv/" + path.relative_to(pkg).as_posix(), path.read_bytes()))
    return out


def _plist(app_name: str) -> bytes:
    ident = "".join(c for c in app_name.lower() if c.isalnum()) or "app"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>{app_name}</string>
<key>CFBundleIdentifier</key><string>dev.dhriv.{ident}</string>
<key>CFBundleExecutable</key><string>launcher</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleVersion</key><string>0.1.0</string>
</dict></plist>
""".encode("utf-8")


def build_package(dhrb_path: Path, app_name: str, target: str, out_dir: Path) -> Path:
    if target not in TARGETS:
        raise ValueError(f"Target '{target}' non valido. Disponibili: {', '.join(TARGETS)}")
    blob = Path(dhrb_path).read_bytes()
    if not verify_binary(blob):
        raise ValueError("Il file .dhrb non è valido o è stato manomesso: pacchetto rifiutato.")

    runtime = _runtime_files()
    entries: List[Tuple[str, bytes, int]] = []

    if target in ("macos", "universal"):
        base = f"macos/{app_name}.app/Contents"
        launcher = (
            '#!/bin/sh\nDIR="$(cd "$(dirname "$0")/../Resources" && pwd)"\n'
            'cd "$DIR" && exec python3 -m dhriv run program.dhrb "$@"\n'
        ).encode("utf-8")
        entries.append((f"{base}/Info.plist", _plist(app_name), 0o644))
        entries.append((f"{base}/MacOS/launcher", launcher, 0o755))
        entries.append((f"{base}/Resources/program.dhrb", blob, 0o644))
        entries += [(f"{base}/Resources/{n}", d, 0o644) for n, d in runtime]

    if target in ("windows", "universal"):
        base = "windows"
        bat = '@echo off\r\ncd /d "%~dp0"\r\npython -m dhriv run program.dhrb %*\r\n'.encode("utf-8")
        entries.append((f"{base}/run.bat", bat, 0o644))
        entries.append((f"{base}/program.dhrb", blob, 0o644))
        entries += [(f"{base}/{n}", d, 0o644) for n, d in runtime]

    if target in ("linux", "universal"):
        base = "linux"
        sh = b'#!/bin/sh\ncd "$(dirname "$0")" && exec python3 -m dhriv run program.dhrb "$@"\n'
        entries.append((f"{base}/run.sh", sh, 0o755))
        entries.append((f"{base}/program.dhrb", blob, 0o644))
        entries += [(f"{base}/{n}", d, 0o644) for n, d in runtime]

    manifest: Dict[str, object] = {
        "app": app_name,
        "target": target,
        "requires": "python>=3.9",
        "program_sha3_256": hashlib.sha3_256(blob).hexdigest(),
        "files": {name: hashlib.sha3_256(data).hexdigest() for name, data, _ in entries},
    }
    entries.append(("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"), 0o644))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{app_name}-{target}.zip"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name, date_time=_FIXED_DATE)
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return out_path
