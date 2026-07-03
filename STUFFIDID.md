## 2026-07-03
- Dannede og kørte runners til intminet
- Klargører til kopiere til lokal opsætning

## 2026-06-02
- Summarizer færdig (OBS: bruger gpt-5.4 nu i scripts, men 5.4-mini som default)
- Brugerflade til at søge records eller chunks (se src/search - standalone html der kan læse en eller flere jsonl ind)
- Næste skridt: runners for maplabourint, intminet
- Næste skridt: Samlet wrapper?
    - PDF dir
    - Hvis embeddings findes; skip
    - Schema-input
    - Vælg hvilke types review skal baseres på
    - Iterer over types
    - Hvis relevant_chunks findes; skip
    - Hvis records findes; skip
    - Opdatér tekst
- Næste skridt: Mulighed for at skrive tekst fra bunden? (evt. blot en prompt justering?)


## 2026-05-29
- Embedder færdig
- Retriever færdig
- Record færdig
- Hele pipeline kørt for Sinks and Sluices
- Al pdf data ligger for SS, MapLabourInt, Intmitnet
- Næste skridt: Summarizer af records