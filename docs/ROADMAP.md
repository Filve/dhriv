# Roadmap e limiti onesti — Dhriv

Dhriv `0.1.0-genesis` è un nucleo reale e testato. Gli obiettivi originari (linguaggio "perfetto", compatibile con ogni lingua e ogni software, auto-compilante e auto-apprendente) sono un percorso lungo: questa pagina separa ciò che esiste da ciò che manca.

## Limiti attuali
- **Esecuzione**: interprete ad albero in Python. Corretto ma lento rispetto a un linguaggio compilato.
- **`.dhrb`**: contenitore firmato dell'AST, non bytecode né codice macchina.
- **Retrocompatibilità**: Dhriv *invoca* software esistente (`legacy_run`) e accetta parole chiave legacy. Non traduce ancora programmi C/Python/JS esistenti in Dhriv.
- **Auto-apprendimento**: limitato alla memoizzazione di funzioni pure, con soglia e persistenza su disco. Non è apprendimento generale né modifica del proprio codice.
- **Auto-compilazione**: ricompilazione incrementale per hash. Il compilatore non è ancora scritto in Dhriv (self-hosting).
- **Onde / EM**: fisica, modem, FEC, `.wav`/`.iq` reali, ma **nessun hardware radio collegato**. Il canale è simulato o su file.
- **Pacchetti OS**: archivi portabili con launcher; richiedono Python sul sistema di destinazione. Non sono installer nativi.
- **Ricerca su larga scala**: indici in memoria in un solo processo, non ancora distribuiti su rete.

## Prossimi passi proposti (in ordine di valore)
1. **Bytecode + VM a registri** con formato `.dhrb` v2 e ottimizzazioni (constant folding, inline cache nel bytecode).
2. **Sistema di tipi** graduale (annotazioni opzionali, inferenza) e moduli/import.
3. **Self-hosting**: riscrivere lexer/parser di Dhriv in Dhriv.
4. **Transpiler d'ingresso** da sottoinsiemi di Python/JavaScript/C verso Dhriv, e FFI vera (ctypes/N-API/C ABI).
5. **Backend nativo** (LLVM o WebAssembly) per eseguibili autonomi.
6. **Ricerca V/H/D distribuita**: nodi su rete, integrazione con il Cybernetwork di `virtual-hosting` (mesh, DHT, Sel & Aer).
7. **Radio reale**: driver SDR (RTL-SDR/HackRF via `.iq`/GNU Radio), audio live, seriale.
8. **Installer nativi**: `.dmg`/`.pkg` macOS, `.msi` Windows, `.deb`/AppImage Linux, con firma.
9. **Apprendimento esteso**: profilazione per specializzare il dispatch tri-assiale, suggerimenti di refactoring, apprendimento delle preferenze di lingua per progetto.
