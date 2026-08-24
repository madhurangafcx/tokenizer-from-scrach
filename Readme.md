# tokenizer

A byte-level BPE tokenizer built from scratch, wrapped in a FastAPI endpoint that
streams each step over Server-Sent Events so the process is watchable rather than
instantaneous.

The goal is understanding, not speed. Every module carries detailed comments
explaining *why* it works the way it does — the code is meant to be read.

## Status

| Stage | Module | State |
| --- | --- | --- |
| Unicode normalization | `normalization.py` | working |
| Pre-tokenization | `pretokenizer.py` | working |
| UTF-8 byte encoding | `byte_encoder.py` | working |
| Vocabulary + persistence | `vocabulary.py` | working |
| Decoding | `decoder.py` | working |
| Facade / orchestration | `tokenizer.py` | working |
| Merge training | `bpe_trainer.py` | working |
| Merge application | `bpe_encoder.py` | working |

Every stage is implemented. Constructed with no merges, the tokenizer runs at the
byte level — every UTF-8 byte is one token, vocabulary exactly 256. Train a
vocabulary and the same pipeline compresses about **2.4× on text resembling its
training corpus**, while unseen text degrades gracefully back toward bytes instead
of failing.

## Quick start

```bash
cd text_pipeline
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/docs> for an interactive form.

Run commands from `text_pipeline/` — `import app.main` resolves against `sys.path`,
whose first entry is the working directory, so only that directory puts `app/` where
Python can find it. To launch from the repo root instead, pass
`--app-dir text_pipeline` to uvicorn.

### Without a server

```bash
cd text_pipeline
.venv/bin/python -c "
from app.tokenizer import Tokenizer
t = Tokenizer()
r = t.encode('Hi é!')
print(r.pre_tokens)     # ['Hi', ' ', 'é', '!']
print(r.token_ids)      # [72, 105, 32, 195, 169, 33]
print(t.decode(r.token_ids))
"
```

Note `é` costing two tokens — `195, 169` are its two UTF-8 bytes. That is the cost
BPE exists to remove.

## Training a vocabulary

```python
from app.bpe_trainer import train
from app.tokenizer import Tokenizer

corpus = [line for line in open("corpus.txt")]   # any iterable of strings
result = train(corpus, vocab_size=8000)

tokenizer = Tokenizer(vocabulary=result.vocabulary, merges=result.merges)
tokenizer.save("artifacts/")                     # vocab.json + merges.txt
```

`train` returns a `TrainResult` holding a `Vocabulary` and an **ordered** merge list.
Reload later with `Tokenizer.load("artifacts/")`.

What the first merges look like on a small English corpus:

```text
rank  0    'h' + 'e'    -> 'he'
rank  1    't' + 'he'   -> 'the'
rank  2  'the' + 'r'    -> 'ther'
rank  3 'ther' + 'e'    -> 'there'
rank  4    'w' + 'a'    -> 'wa'
rank  5   'wa' + 's'    -> 'was'
```

Each merge is built from earlier ones — `he` → `the` → `ther` → `there`. That
staircase is how you tell training is working. Which pair wins first depends
entirely on the corpus: here `('h','e')` beat `('t','h')` because `he` occurs more.

Vocabulary size is a budget, not a promise:

```text
vocab_size=264   8 merges   'the theory of the weather there' -> 17 tokens
vocab_size=300  44 merges                                     -> 11 tokens
vocab_size=400  62 merges                                     -> 11 tokens
```

At 400 it stopped at 62 merges — the corpus ran out of pairs occurring more than
once. Training halts there rather than merging noise, so `len(merges)` can be
smaller than `vocab_size - 256`. On a real corpus you would not hit this.

`normalization_form` and `pretokenizer` **must match** whatever the `Tokenizer`
later uses. A mismatch raises no error; the encoder simply produces pieces the
trainer never saw, the merges stop applying, and token counts quietly balloon.

## The pipeline

```text
text
  │
  ├─▶ normalization.normalize ────────▶ one canonical Unicode spelling
  │
  ├─▶ PreTokenizer.split ─────────────▶ pieces BPE may never merge across
  │
  └─▶ for each piece:
        byte_encoder.encode ──────────▶ byte ids 0..255
              │
              ├── no merges ─────────▶ Piece(text, byte_ids)
              │
              └── merges ────────────▶ BPEEncoder.encode_piece
                                        └─▶ Piece(text, merged_ids)
                          │
                          ▼
                    EncodeResult ─▶ main.py event_generator ─▶ SSE

decode:  ids ─▶ decoder.decode(vocabulary) ─▶ text
```

Two boundaries hold this together:

**The tokenizer knows nothing about HTTP.** Everything under `app/` except
`main.py` and `stages.py` is plain synchronous Python — no `async`, no FastAPI. All
tokenization finishes *before* the first SSE event is sent; the stream just walks
the finished result slowly. So the whole tokenizer is drivable from a script or a
test with no server running.

**BPE plugs into exactly one branch.** `Tokenizer._bpe` is `None` when there are no
merges, and `encode` passes byte ids straight through. Once `bpe_trainer` produces
merges, that branch wakes up and no other module changes.

## API

`POST /process` with `{"text": "..."}` returns an SSE stream of four event types.

```bash
curl -N -X POST http://127.0.0.1:8000/process \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hi é!"}'
```

Real output, abridged:

```text
event: start
data: {"stage": "Segmenting", "input_text": "Hi é!", "normalized_text": "Hi é!",
       "pre_tokens": ["Hi", " ", "é", "!"], "pre_token_count": 4,
       "total_byte_tokens": 6, "vocab_size": 256}

event: pre_token
data: {"stage": "Analyzing", "pre_token_index": 1, "pre_token": "Hi",
       "pre_token_token_count": 2, "bpe_boundary": true}

event: progress
data: {"stage": "Transforming", "pre_token": "Hi", "pre_token_index": 1,
       "pre_token_position": 1, "pre_token_total_tokens": 2, "token_id": 72,
       "token_position": 1, "tokens_processed": 1, "total_tokens": 6,
       "remaining_tokens": 5, "progress_percent": 16.67, "bpe_boundary": false}

event: complete
data: {"stage": "tokenization_complete", "pre_token_count": 4,
       "total_byte_tokens": 6, "vocab_size": 256}
```

| Event | When | Meaning |
| --- | --- | --- |
| `start` | once, first | The full picture: normalized text, all pre-tokens, totals |
| `pre_token` | per piece | Entering a BPE boundary (`bpe_boundary: true`) |
| `progress` | per token | Which token, where it sits, overall progress |
| `complete` | once, last | Final counts — distinguishes a finished stream from a dropped connection |

The `stage` labels are decorative. Tokenization is already finished before the first
event; a label reading `"Normalizing"` does not mean normalization is happening.

Field names are a frontend contract — add fields rather than renaming them.

## Key concepts

Each is explained in full in the module that implements it.

**Why bytes, not characters** (`byte_encoder.py`) — Unicode has ~150,000 codepoints;
seeding a vocabulary with all of them is wasteful, and anything omitted needs an
`<UNK>` token that destroys information. There are exactly 256 byte values, and
every possible string is some sequence of them. So: 256 seed tokens and **no `<UNK>`
is ever needed.**

**Why tokens look like text but aren't** (`byte_encoder.py`) — token strings are
stored in `vocab.json` and `merges.txt`, but 68 of the 256 byte values have no
printable form (NUL, newline, tab, space). Each gets a printable stand-in, which is
where the `Ġ` in real GPT-2 vocabularies comes from: it is byte `0x20` in a costume.
Consequence — `Vocabulary.bytes_for(id)` is the only correct way to get a token's
real bytes. Never `.encode("utf-8")` a token string.

**Why pre-tokenization comes first** (`pretokenizer.py`) — BPE merges whatever pair
is most frequent, so on raw text it happily learns `"dog."` and `", the"`.
Pre-tokenization draws walls, and merging runs inside each piece independently. The
pattern you choose decides what the model is *able* to learn; it is a modelling
decision, not preprocessing.

**Why merge order is load-bearing** (`bpe_encoder.py`) — merges are a sequence, not
a set. `"the"` can only be built once `"th"` exists. Rank is list position, so
sorting or deduplicating a merges list silently changes tokenization.

**What decoding can't recover** (`decoder.py`) — byte encoding and merging are
lossless, normalization is not (`ﬁ` became `fi`). So
`decode(encode(text)) == normalize(text)`, and equals `text` only if the input was
already normalized. `EncodeResult` keeps both forms for this reason.

## On-disk format

`Tokenizer.save(directory)` writes the two files real BPE tokenizers ship:

```text
vocab.json    {"token": id, ...}      what tokens exist
merges.txt    "left right" per line   in what order they were built
```

They are separate because they answer different questions, and because `merges.txt`
must stay line-ordered — line number *is* merge rank.

Not persisted: the normalization form and the pre-tokenizer pattern. Whoever loads a
vocabulary must supply matching ones via `Tokenizer.load(dir, normalization_form=...,
pretokenizer=...)`. A mismatch produces no error — just merges that quietly stop
applying, and token counts that balloon.

## Notes

- Normalization defaults to **NFKC**. An earlier version chained all four forms
  (`NFC → NFD → NFKC → NFKD`), which nets out to NFKD; pass `form="NFKD"` to
  reproduce that.
- `STEP_DELAY_SECONDS` in `main.py` is `0.01`, so a request costs roughly
  `(1 + pieces + tokens) × 0.01` seconds. A 2,000-byte input takes ~20s. Lower it
  for anything but demonstration.
- The default pre-tokenizer splits whitespace into its own piece, so **no merge can
  ever span a space** — unlike GPT-2, which glues a leading space onto the following
  word. `GPT2_PATTERN` is provided for comparison.
- `merge_pair` lives in `bpe_encoder.py` and is imported by `bpe_trainer.py`. Both
  must apply merges identically, so the loop exists once rather than twice.
- Training is O(rounds × distinct pieces) — every round recounts all pairs. Fine to
  a few thousand merges; the standard fix is incremental pair counts that only
  update the words containing the merged pair.
- No test suite yet; `pytest` is not in `.venv`.
