# Specifica del linguaggio Dhriv `0.1.0-genesis`

## 1. Lessico
- Estensioni: `.dhr` (sorgente), `.dhrb` (binario compilato, vedi §7).
- Commenti: `// riga`, `# riga`, `/* blocco */`. Terminatori `;` facoltativi.
- Il linguaggio è insensibile ai ritorni a capo: se una riga inizia con `(`, `[` o `-`, termina la precedente con `;`.
- Numeri: `42`, `3.14`, `1_000`, notazione scientifica `2.4e9`. Stringhe `"..."` o `'...'` con escape.

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

`self` ha alias `this`, `questo`, `este`, `soi`. `print` ha alias `stampa`.
`python3 -m dhriv render file.dhr --to <lingua>` riscrive solo le parole chiave: stringhe, commenti e identificatori non cambiano.

## 3. Entità e i tre assi
```
entity Nome [vertical A, B] [horizontal C] [diagonal D] { campi e metodi }
```
Le clausole sono combinabili in qualsiasi ordine. Le entità collegate devono essere già dichiarate.

- **Verticale** — ereditarietà in profondità (stack). Si risale `vertical_parents`.
- **Orizzontale** — composizione tra pari (trait/mixin/peer). Scansione in ampiezza.
- **Diagonale** — ponte trasversale che attraversa insieme livelli e domini: dal bersaglio si esplora la sua catena diagonale, verticale e orizzontale.

**Ordine di risoluzione di un membro su un'istanza:** campi propri → metodi della classe (`Self`) → Verticale → Orizzontale → Diagonale → link dinamici `<->` → link dinamici `</>`.
Se manca, l'errore cita gli assi esplorati. Il costruttore è il metodo `init`. I campi di default sono copiati per istanza (nessuno stato mutabile condiviso).

## 4. Operatori tri-assiali
| Operatore | Asse | Significato |
|---|---|---|
| `x \|> f` / `x \|> f(a)` | Verticale | Pipe: `x` scende in `f` (come primo argomento) |
| `a <-> b` | Orizzontale | Su due istanze: risonanza simmetrica. Su due onde: sovrapposizione (interferenza) |
| `a </> b` | Diagonale | Ponte diagonale diretto `a → b` tra istanze |

Precedenza: sono gli operatori a precedenza più bassa (dopo `or`).

## 5. Istruzioni ed espressioni
`let`, `wave`, assegnazione (variabili, `obj.campo`, `lista[i]`), `if/else if/else`, `while`, `for x in lista|testo|mappa`, `return`, `fn`.
Operatori: `+ - * / % **`, `== != < <= > >=`, `and/&&`, `or/||`, `not/!`. Liste `[..]`, mappe `{k: v}`. Divisione tra interi esatta restituisce intero.

## 6. Libreria nativa
- Base: `print stampa len str num int float range push pop keys abs min max floor sqrt sin cos round upper lower join split type_of clock pi c_light`.
- Onde: `wave_new(freq, amp, phase, medium)`, `spectrum`, `wavelength`, `band`, `photon_ev`, `fspl(freq, m)`, `em_transmit(testo, snr_db, schema)`. Attributi onda: `frequency_hz wavelength_meters band_name photon_energy_joules ...`.
- Ricerca larga scala: `scale_register(id, h, v, [capacità])`, `search_h(v, cap)`, `search_v(h, cap)`, `search_d(h, v, cap)`.
- Legacy: `legacy_run(comando)` — solo con `--allow-legacy`.

## 7. Formato binario `.dhrb`
`"DHRB"` · versione u16 · flag u16 · lunghezza u32 · SHA3-256 (32 B) · CRC32 u32 · payload = zlib(JSON canonico `{meta, ast}`).
Il caricamento ricostruisce solo tipi AST registrati e rifiuta file corrotti o manomessi. **È un contenitore dell'AST, non bytecode nativo** (vedi roadmap).

## 8. Modem elettromagnetico
Frame `DHRW` + lunghezza + CRC32 + payload, codificato Hamming(7,4) (corregge 1 bit ogni 7), modulato BFSK (1500/3000 Hz) o BPSK. Esportabile in `.wav` PCM 16-bit e `.iq` float32 interleaved. Il test del repository trasmette un `.dhrb` attraverso rumore AWGN, lo riceve identico e lo esegue.
Il canale è **simulato o su file**: non c'è ancora l'accesso a una radio reale.

## 9. Sicurezza
Budget di passi e profondità, timeout del WatchDog, `legacy_run` disattivato di default, `.dhrb` verificati con SHA3/CRC prima del caricamento, nessun `eval`/`pickle`.
