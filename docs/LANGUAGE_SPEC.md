# Specifica del linguaggio Dhriv `0.2.0-synapse`

## 1. Lessico e Basi Numeriche (dal Binario alle Frequenze EM)
- Estensioni: `.dhr` (sorgente), `.dhrb` (binario compilato v2 con Bytecode ISA a 64 bit + AST firmato, vedi §8).
- Commenti: `// riga`, `# riga`, `/* blocco */`. Terminatori `;` facoltativi.
- Numeri:
  - Binari: `0b1010_0110`, `0B11110000`
  - Ottali: `0o755`
  - Esadecimali: `0xFF_A0`
  - Decimali e scientifici: `42`, `3.14`, `1_000`, `2.4e9`.
- Stringhe `"..."` o `'...'` con escape (`\n`, `\t`, `\r`, `\\`, `\"`, `\'`).

## 2. Parole chiave poliglotte
Ogni parola chiave ha una forma canonica e alias. Il lexer le normalizza tutte verso lo stesso token.

| Canonico (en) | it | es | fr | Legacy accettati |
|---|---|---|---|---|
| `entity` | `entita`, `classe` | `entidad`, `clase` | `entite` | `class`, `struct` |
| `vertical` | `verticale`, `estende` | `vertical` | `verticale` | `extends` |
| `horizontal` | `orizzontale`, `con` | `horizontal` | `horizontale` | `with`, `trait` |
| `diagonal` | `diagonale`, `attraversa` | `diagonal` | `diagonale` | `bridges`, `transversal` |
| `fn` | `funzione` | `funcion` | `fonction` | `def`, `function`, `func` |
| `let` | `sia` | `sea` | `soit` | `var`, `const` |
| `wave` | `onda` | `onda` | `onde` | — |
| `new` / `return` | `nuovo` / `ritorna` | `nuevo` / `retornar` | `nouveau` / `retourner` | — |
| `if` / `else` | `se` / `altrimenti` | `si` / `sino` | `si` / `sinon` | — |
| `while` / `for` / `in` | `mentre` / `per` / `dentro` | `mientras` / `para` / `dentro` | `tantque` / `pour` / `dans` | — |
| `true` / `false` / `null` | `vero` / `falso` / `nullo` | `verdadero` / `falso` / `nulo` | `vrai` / `faux` / `nul` | `True`/`False`/`None`/`nil` |

`self` ha alias `this`, `questo`, `se_stesso`, `este`, `soi`. `print` ha alias `stampa`. `tensor` ha alias `tensore`.
`python3 -m dhriv render file.dhr --to <lingua>` riscrive solo le parole chiave preservando stringhe, commenti e identificatori.

## 3. Entità e i tre assi (OOP Tri-Assiale 2.0)
```text
entity Nome [vertical A, B] [horizontal C] [diagonal D] { campi e metodi }
```
- **Verticale (`vertical`)** — ereditarietà profonda nello stack.
- **Orizzontale (`horizontal`)** — simbiosi tra pari (trait/mixin/peer).
- **Diagonale (`diagonal`)** — ponte trasversale che attraversa insieme livelli e domini.

**Ordine di risoluzione:** campi propri → `Self` → `Verticale` → `Orizzontale` → `Diagonale` → link dinamici `<->` → link dinamici `</>`.
- **Entità invocabili**: se un'entità definisce `fn forward(x)` (o `call` / `esegui`), può essere usata direttamente come stadio di una pipeline verticale `valore |> istanza`.
- **Dispatch assiale esplicito e sciame**:
  - `axis_call(obj, "VERTICAL"|"HORIZONTAL"|"DIAGONAL", "metodo", [args])`
  - `axis_broadcast(obj, "HORIZONTAL"|"DIAGONAL"|"ALL", "metodo", [args])`
  - `coords_of(obj)` restituisce la posizione geometrica `(h, v, d_h, d_v)` nel reticolo.

## 4. Operatori Tri-Assiali Unificati (Oggetti, Onde, Tensori e Reti Neurali)
| Operatore | Asse | Su Entità / Funzioni | Su Onde EM | Su Tensori / AI |
|---|---|---|---|---|
| `x \|> f` | Verticale | `f(x)` o `x \|> entita.forward` | Pipe in filtri/analisi | Forward pass neurale (`X \|> layer \|> relu`) |
| `a <-> b` | Orizzontale | Link simbiotico bidirezionale | Sovrapposizione / interferenza | Fusione tensoriale simmetrica `(a + b) / 2` |
| `a </> b` | Diagonale | Ponte diagonale diretto `a → b` | — | Proiezione matriciale differenziabile (`matmul`) |

## 5. Istruzioni, Funzioni Lambda e Operatori Bitwise
- Funzioni e metodi supportano sia blocchi `{ ... }` sia forma compatta `fn f(x) => x * 2`, incluse lambda anonime `fn(x, y) => x + y`.
- Operatori aritmetici: `+ - * / % **` (validi anche su `Tensor` con broadcast e Autograd).
- Operatori bitwise interi: `&`, `|`, `^`, `~`, `<<`, `>>`.
- Confronti e logici: `== != < <= > >=`, `and`/`&&`, `or`/`||`, `not`/`!`.

## 6. Fondamenta Binarie & ISA a 64 bit
- Porte logiche universali derivate da `NAND`: `bit_gate("NAND"|"AND"|"OR"|"XOR"|"NOR"|"XNOR"|"NOT", a, b)`, `half_adder(a, b)`, `full_adder(a, b, cin)`.
- ALU hardware simulata a `N` bit: `alu(op, a, b, width)` con flag processore `Z, C, N, V`, `bits_of`, `bitstr`, `from_bits`, `hex_of`.
- **Parola-istruzione a 64 bit (`InstructionWord`)**:
  `[63..56 OPCODE (8b) | 55..54 AXIS (2b: 00=SELF, 01=V, 10=H, 11=D) | 53..48 FLAGS (6b) | 47..32 ARG_A (16b) | 31..0 ARG_B (32b)]`.
  Ispezionabile con `python3 -m dhriv disasm file.dhr --bits`.

## 7. Motore Nativo di Intelligenza Artificiale (`ai_engine.py`)
- **Tensori & Autograd**: `tensor(data, requires_grad)`, `matmul`, `backward(loss)`, `zero_grad`, `item`, `relu`, `sigmoid`, `tanh`, `gelu`, `softmax`, `t_sum`, `t_mean`. Attributi `.shape`, `.rows`, `.cols`, `.values`, `.grad`, `.T`.
- **Funzioni di costo**: `mse_loss(pred, target)`, `bce_loss(pred, target)`, `cross_entropy(logits, targets)`.
- **Architetture neurali**:
  - `nn_linear(in_dim, out_dim, [seed])`
  - `nn_attention(d_model, [seed])` — Scaled Dot-Product Self-Attention (Transformer).
  - `nn_triaxial(in_dim, hidden_dim, out_dim, [depth], [act], [seed])` — **Rete Neurale Tri-Assiale** nativa con stack Verticale (`tanh`), gate laterale Orizzontale (`sigmoid`) e ponte residuo proiettivo Diagonale (`X_0 </> W_D`).
- **Ottimizzatori**: `nn_sgd(params, lr, momentum)`, `nn_adam(params, lr)`, `opt_step(opt)`, `opt_zero_grad(opt)`.
- **Memoria Neuro-Simbolica & Codec Radio AI**: `ai_memory()`, `mem_store`, `mem_recall`, `cosine_sim`, `ai_pack_hex`, `ai_unpack_hex`, `em_transmit_ai` (trasmette pesi IEEE-754 `DHAI` su onde modulate con FEC Hamming + CRC32).

## 8. Contenitore binario `.dhrb` v2 e Sicurezza
Header a 48 byte (`"DHRB"` · versione `2` · flag `0x0001` · lunghezza · `SHA3-256` · `CRC32`) + payload compresso contenente sia il modulo bytecode binario `DHBC` a 64 bit sia l'AST canonico. Piena retrocompatibilità in lettura con `.dhrb` v1; nessun `eval`/`pickle`.
