# Roadmap e limiti onesti — Dhriv

Dhriv `0.2.0-synapse` evolve `0.1.0-genesis` introducendo il livello binario a porte logiche/ALU, il compilatore Bytecode a 64 bit con `DhrivVM`, l'OOP Tri-Assiale 2.0 e il motore nativo di Intelligenza Artificiale (Tensori, Autograd, Self-Attention, Rete Neurale Tri-Assiale e Memoria Neuro-Simbolica). Questa pagina separa con chiarezza ciò che è già realizzato da ciò che resta nei traguardi successivi.

## Traguardi completati in `0.2.0-synapse`
- [x] **Fondamenta binarie e ALU**: porte logiche derivate da `NAND`, `half_adder`/`full_adder`, `BinaryALU` ripple-carry con flag `Z/C/N/V`, letterali `0b`/`0x`/`0o` e operatori bitwise `& | ^ ~ << >>`.
- [x] **Bytecode ISA a 64 bit + `DhrivVM` + `.dhrb` v2**: constant folding, istruzioni binarie a 64 bit con campo `AXIS` a 2 bit, disassemblatore `--bits`, retrocompatibilità `.dhrb` v1.
- [x] **OOP Tri-Assiale 2.0**: entità invocabili in pipe `|>` (`forward`), dispatch assiale esplicito (`axis_call`), broadcast a sciame (`axis_broadcast`), lambda `fn(x) => ...`.
- [x] **Motore AI nativo zero-dipendenze**: `Tensor` con Autograd reverse-mode verificato con differenze finite, `nn_linear`, `nn_attention` (Transformer), `TriAxialNetwork` (V/H/D), `SGD`/`Adam`, `MindMemory` e serializzazione pesi binari `DHAI` trasmissibile via onde EM.

## Limiti attuali (onesti)
- **Velocità numerica su larga scala**: `DhrivVM` e `Tensor` girano in puro Python senza dipendenze C/CUDA; ideali per architetture cognitive leggere, prototipazione, edge/ricerca e reti di piccola-media taglia, non ancora per addestrare LLM da miliardi di parametri.
- **Sistema di tipi e moduli**: tipizzazione dinamica; manca ancora un type-checker statico graduale e un sistema multi-file `import`/`module`.
- **Self-hosting e backend nativo**: il compilatore bytecode è scritto in Python e genera ISA `DHBC`, non ancora codice macchina x86-64/ARM64/LLVM nativo autonomo senza Python.
- **Rete e radio fisica**: la ricerca V/H/D e il modem EM operano in memoria o su file `.wav`/`.iq`; non c'è ancora il trasporto TCP/QUIC distribuito né il binding diretto a driver SDR hardware.

## Prossimi passi proposti (in ordine di valore)
1. **Moduli multi-file (`import` / `export`) e Sistema di Tipi Graduale** (inclusi tipi di forma per i Tensori `Tensor[M, N]`).
2. **Acceleratore C / SIMD / GPU opzionale** per `Tensor.matmul` e `DhrivVM` (mantenendo il fallback puro Python zero-dipendenze).
3. **Backend nativo** (transpilazione C / LLVM IR / WebAssembly) per generare eseguibili binari nativi standalone senza interprete Python.
4. **Self-hosting**: scrivere il lexer, il parser e l'assembler di Dhriv direttamente in Dhriv (`.dhr`).
5. **Rete Cognitiva V/H/D Distribuita**: collegare `axis_broadcast`, `TriAxialLargeScaleSearch` e `MindMemory` su nodi reali in rete mesh (integrazione con `virtual-hosting`).
6. **Hardware SDR reale**: collegamento diretto dei frame `DHRW` e `DHAI` a dispositivi RTL-SDR / HackRF / audio live.
