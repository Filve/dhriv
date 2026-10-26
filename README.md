# Dhriv

> **Autore e creatore originario:** Francesco Simeoni (`@Filve`) — diritti riservati, vedi [LICENSE.md](LICENSE.md) e [AUTHORS.md](AUTHORS.md).
> Branch di sviluppo: `dev_main_gemini`.

**Dhriv** è un linguaggio di programmazione orientato agli oggetti su **tre assi**: **Verticale**, **Orizzontale**, **Diagonale**, progettato metodicamente **dal singolo bit (`0`/`1`) fino all'Intelligenza Artificiale nativa e alle onde elettromagnetiche**. Ogni entità, tensore o nodo server vive in un reticolo `(H, V, D)` e si muove lungo questi assi.

Versione attuale: `0.2.0-synapse` — **zero dipendenze esterne (solo Python ≥ 3.9), testato su 53 test automatici**. Cosa c'è e cosa resta nei passi futuri è documentato con trasparenza in [docs/ROADMAP.md](docs/ROADMAP.md).

```text
// 1. Dal livello binario puro e ALU a porte logiche...
let hw = alu("ADD", 0b0010_1010, 0x10, 8)

// 2. ...all'OOP Tri-Assiale invocabile in pipeline verticali...
entity EdgeNode vertical Server horizontal Telemetry diagonal Kernel {
    fn forward(x) => x * 2
}

// 3. ...fino a Reti Neurali Tri-Assiali (V/H/D), Self-Attention e Autograd nativo!
let brain = nn_triaxial(2, 8, 1)
let pred = tensor([[0, 1], [1, 0]]) |> brain
```

## Avvio rapido (solo Python ≥ 3.9, zero dipendenze)

```bash
python3 -m dhriv run examples/binary_to_ai.dhr      # bit -> ALU -> rete neurale XOR -> onda EM
python3 -m dhriv run examples/triaxial_mind_it.dhr  # OOP V/H/D + Self-Attention + memoria in italiano
python3 -m dhriv run examples/triaxial.dhr          # OOP tri-assiale + ricerca V/H/D
python3 -m dhriv run examples/waves.dhr             # onde, bande EM, modem con rumore
python3 -m dhriv disasm examples/ciao_it.dhr --bits # disassembla in istruzioni binarie a 64 bit
python3 -m unittest discover -s tests -v            # 53 test
```

## Architettura Metodica (5 Livelli, tutto coperto da test)

| Livello / Area | Modulo | Stato |
|---|---|---|
| **0. Fondamenta Binarie & ALU**: porte logiche derivate da `NAND` (`NOT`, `AND`, `OR`, `NOR`, `XOR`, `XNOR`), `half_adder`, `full_adder`, `BinaryALU` ripple-carry a N bit con flag `Z/C/N/V`, letterali `0b`/`0x`/`0o` e operatori bitwise `& \| ^ ~ << >>` | [`binary_core.py`](dhriv/binary_core.py), [`lexer.py`](dhriv/lexer.py) | Funzionante |
| **1. ISA a 64 bit, Bytecode & `DhrivVM`**: parola binaria fissa `[8b OP \| 2b AXIS \| 6b FLAGS \| 16b A \| 32b B]`, constant folding, contenitore firmato `.dhrb` v2 (`SHA3-256` + `CRC32` + sezione `DHBC`), disassemblatore `--bits` | [`bytecode_vm.py`](dhriv/bytecode_vm.py), [`compiler.py`](dhriv/compiler.py) | Funzionante |
| **2. OOP Verticale / Orizzontale / Diagonale 2.0**: risoluzione `Self → V → H → D`, cache inline O(1), entità invocabili in pipe `\|>` (`forward`), dispatch assiale esplicito (`axis_call`), sciame multi-istanza (`axis_broadcast`), lambda `fn(x) => ...` | [`triaxial_oop.py`](dhriv/triaxial_oop.py), [`interpreter.py`](dhriv/interpreter.py) | Funzionante |
| **3. Intelligenza Artificiale & Neuro-Simbolico**: `Tensor` 2D con **Autograd reverse-mode** verificato analiticamente, operatori `</>` (`matmul`) e `<->` (fusione tensoriale), `nn_linear`, **Transformer `nn_attention`**, **`TriAxialNetwork`** (stack V + gate H + skip-bridge D), ottimizzatori `SGD`/`Adam`, memoria associativa coseno `MindMemory` e codec pesi binari IEEE-754 `DHAI` | [`ai_engine.py`](dhriv/ai_engine.py) | Funzionante |
| **4. Poliglotta, Onde EM, Sicurezza & Packaging**: parole chiave `it`/`en`/`es`/`fr`, modem BFSK/BPSK con FEC Hamming(7,4) (capace di trasmettere sia `.dhrb` sia pesi neurali `em_transmit_ai`), `WatchGuardian`, `WatchDog`, auto-apprendimento `SelfLearningEngine` e pacchetti multipiattaforma | [`polyglot.py`](dhriv/polyglot.py), [`wave_electromagnetic.py`](dhriv/wave_electromagnetic.py), [`guardians.py`](dhriv/guardians.py), [`selflearn.py`](dhriv/selflearn.py), [`packager.py`](dhriv/packager.py) | Funzionante |

## CLI

```bash
python3 -m dhriv run file.dhr|file.dhrb [--engine vm|ast] [--watchdog] [--allow-legacy] [--report]
python3 -m dhriv disasm file.dhr|file.dhrb [--bits]      # codice macchina ISA a 64 bit
python3 -m dhriv alu ADD 0b00101010 0x10 [--bits 8]      # ALU hardware simulata a porte logiche
python3 -m dhriv compile file.dhr [-o out.dhrb]
python3 -m dhriv render file.dhr --to it|en|es|fr|auto
python3 -m dhriv detect file.dhr
python3 -m dhriv guard seal|verify
python3 -m dhriv watch file.dhr [--cycles N]
python3 -m dhriv package file.dhr --target macos|windows|linux|universal
python3 -m dhriv transmit file.dhrb --wav out.wav        # binario -> onda
python3 -m dhriv receive out.wav -o back.dhrb            # onda -> binario
python3 -m dhriv repl
```

Specifica completa del linguaggio: [docs/LANGUAGE_SPEC.md](docs/LANGUAGE_SPEC.md).

## Licenza

Proprietaria, tutti i diritti riservati a Francesco Simeoni. Vedi [LICENSE.md](LICENSE.md).
