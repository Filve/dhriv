# Dhriv

> **Autore e creatore originario:** Francesco Simeoni (`@Filve`) — diritti riservati, vedi [LICENSE.md](LICENSE.md) e [AUTHORS.md](AUTHORS.md).
> Branch di sviluppo: `dev_main_gemini`.

**Dhriv** è un linguaggio di programmazione orientato agli oggetti su **tre assi**: **Verticale**, **Orizzontale**, **Diagonale**. Ogni entità vive in un reticolo `(H, V, D)` e ogni ricerca di membri, metodi o nodi server si muove lungo questi assi, come una ricerca su larga scala in un cluster.

Versione attuale: `0.1.0-genesis` — **un nucleo funzionante e testato, non ancora il linguaggio "completo"**. Cosa c'è e cosa manca è dichiarato apertamente in [docs/ROADMAP.md](docs/ROADMAP.md).

```text
entity EdgeNode vertical Server horizontal Telemetry diagonal Kernel { }
let n = new EdgeNode("edge-01", 250)
print(n.describe(), n.report(), n.syscall(7))   // risolti su V, H e D
```

## Avvio rapido (solo Python ≥ 3.9, zero dipendenze)

```bash
python3 -m dhriv run examples/triaxial.dhr          # OOP tri-assiale + ricerca V/H/D
python3 -m dhriv run examples/waves.dhr             # onde, bande EM, modem con rumore
python3 -m dhriv run examples/ciao_it.dhr --report  # sintassi italiana + auto-apprendimento
python3 -m unittest discover -s tests -v            # 38 test
```

## Cosa c'è oggi (tutto coperto da test)

| Area | Modulo | Stato |
|---|---|---|
| Lexer/Parser/AST con errori su riga:colonna | [`lexer.py`](dhriv/lexer.py), [`parser.py`](dhriv/parser.py), [`ast_nodes.py`](dhriv/ast_nodes.py) | Funzionante |
| **OOP Verticale / Orizzontale / Diagonale**: ordine di risoluzione `Self → V → H → D`, cache di dispatch, link dinamici `<->` e `</>` tra istanze | [`triaxial_oop.py`](dhriv/triaxial_oop.py) | Funzionante |
| **Ricerca su larga scala** orizzontale (per livello), verticale (per shard), diagonale (rette `h−v` e `h+v`) con indici O(1) | [`triaxial_oop.py`](dhriv/triaxial_oop.py) | Funzionante (in memoria) |
| Interprete con budget di passi, profondità massima, pipe `\|>` | [`interpreter.py`](dhriv/interpreter.py) | Funzionante |
| **Poliglotta**: parole chiave in `it`/`en`/`es`/`fr` + termini legacy C/Java/JS/Python; rilevamento lingua e **riscrittura automatica** del sorgente nella lingua scelta (commenti e stringhe intatti) | [`lexer.py`](dhriv/lexer.py), [`polyglot.py`](dhriv/polyglot.py) | Funzionante |
| **Binario `.dhrb`** (header, SHA3-256, CRC32, AST compresso) e **auto-compilazione incrementale** per hash | [`compiler.py`](dhriv/compiler.py) | Funzionante |
| **WatchGuardian**: sigillo SHA3-256 del runtime, rilevamento modifiche, verifica `.dhrb`, scansione chiamate sensibili | [`guardians.py`](dhriv/guardians.py) | Funzionante |
| **WatchDog**: esecuzione supervisionata (timeout, stallo, riavvii solo per errori non deterministici), ricompilazione automatica al cambio file | [`guardians.py`](dhriv/guardians.py) | Funzionante |
| **Auto-apprendimento**: analisi statica di purezza + memoizzazione appresa e persistita tra esecuzioni | [`selflearn.py`](dhriv/selflearn.py) | Funzionante (ambito: funzioni pure) |
| **Onde/EM**: bande, lunghezza d'onda, FSPL, sintesi, spettro, modem BFSK/BPSK con Hamming(7,4)+CRC32, export `.wav` e `.iq` (SDR) | [`wave_electromagnetic.py`](dhriv/wave_electromagnetic.py) | Funzionante in simulazione/file |
| **Retrocompatibilità**: `legacy_run` invoca software esistente (senza shell, timeout, opt-in `--allow-legacy`) | [`compat.py`](dhriv/compat.py) | Funzionante |
| **Pacchetti portabili** macOS (`.app`), Windows (`run.bat`), Linux (`run.sh`) | [`packager.py`](dhriv/packager.py) | Funzionante (richiede Python sul target) |

## CLI

```bash
python3 -m dhriv run file.dhr|file.dhrb [--watchdog] [--allow-legacy] [--report]
python3 -m dhriv compile file.dhr [-o out.dhrb]
python3 -m dhriv render file.dhr --to it|en|es|fr|auto
python3 -m dhriv detect file.dhr
python3 -m dhriv guard seal|verify
python3 -m dhriv watch file.dhr [--cycles N]
python3 -m dhriv package file.dhr --target macos|windows|linux|universal
python3 -m dhriv transmit file.dhrb --wav out.wav   # binario -> onda
python3 -m dhriv receive out.wav -o back.dhrb       # onda -> binario
python3 -m dhriv repl
```

Specifica completa del linguaggio: [docs/LANGUAGE_SPEC.md](docs/LANGUAGE_SPEC.md).

## Licenza

Proprietaria, tutti i diritti riservati a Francesco Simeoni. Vedi [LICENSE.md](LICENSE.md).
