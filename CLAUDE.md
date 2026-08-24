# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A BPE tokenizer built from scratch for learning, wrapped in a FastAPI endpoint that
streams the tokenization step-by-step over SSE so the process is watchable.
`bpe_trainer.py` and `bpe_encoder.py` are deliberate stubs — the merge-learning
algorithm is the next thing to build. Everything else works today at the byte level.

## Commands

Always run from `text_pipeline/`. `app` is a regular package, but `import app.main` still
resolves against `sys.path`, whose first entry is the working directory — only
`text_pipeline/` puts `app/` somewhere Python can see it. From the repo root you get
`ModuleNotFoundError: No module named 'app'`; from inside `app/`, the relative imports in
`main.py` fail with `attempted relative import with no known parent package`. To launch
from elsewhere, pass `--app-dir text_pipeline` to uvicorn.

```bash
cd text_pipeline

# Run the API (http://127.0.0.1:8000/docs for the interactive form)
.venv/bin/uvicorn app.main:app --reload

# Exercise the pipeline without a server
.venv/bin/python -c "
from app.tokenizer import Tokenizer
t = Tokenizer()
r = t.encode('Hello, wörld! 🎉')
print(r.pre_tokens, r.total_tokens, t.vocab_size)
print(repr(t.decode(r.token_ids)))
"

# The invariant worth checking after any change to byte_encoder or vocabulary:
# decode(encode(x)) must equal the NORMALIZED text, not the original input.
```

Dependencies are pinned in `text_pipeline/requirements.txt`. There is no test suite and no
`pyproject.toml`; `pytest` is not installed in `.venv`. If you add tests, install pytest
into `.venv` first and put them in `text_pipeline/tests/`.

**The git repo root is `text_pipeline/`, not the project root.** `git` commands must run
from there, and `.gitignore` lives there for the same reason. Consequence worth knowing:
`Readme.md` and this file sit *above* the repo root, so they are not version-controlled.
The branch is `main` with no commits yet.

## Architecture

### The library/API boundary

`app/tokenizer.py` is the facade. `main.py` imports `Tokenizer` from it and nothing
else from the tokenizer side. **No module in the tokenizer chain knows about async,
HTTP, or SSE** — they are plain sync functions and classes returning dataclasses. All
tokenization completes synchronously *before* streaming starts; the SSE generator only
walks an already-computed `EncodeResult` and inserts `STEP_DELAY_SECONDS` for visual
effect. Keep it that way: putting `await` into a tokenizer module makes it unusable
from a plain script or a test.

`app/stages.py` is cosmetic. The 45 stage labels are decoration for the stream and have
no bearing on tokenization.

### Pipeline

```text
text ─▶ normalization.normalize ─▶ PreTokenizer.split ─▶ [per piece] byte_encoder.encode
                                                                  │
                                                  merges? ─── no ─┴─▶ Piece(text, byte_ids)
                                                     │
                                                    yes ─▶ BPEEncoder.encode_piece
```

`Tokenizer.__init__` sets `self._bpe = None` when `merges` is empty, and `encode()` then
passes byte ids straight through. Once `bpe_trainer.train` returns real merges, that one
branch wakes up and no other module changes. This is the seam the whole layout exists to
protect — implement the trainer and encoder against it rather than reshaping the
pipeline.

### Invariants that will bite you

**Token strings are printable stand-ins, not raw bytes.** `byte_encoder.bytes_to_unicode`
maps each of the 256 bytes to a printable character (space `0x20` → `Ġ`) so vocab entries
can be written into JSON. Consequences that are easy to violate:

- `Vocabulary.bytes_for(id)` is the only correct way to get real bytes from a token id.
  Never `.encode("utf-8")` a token string.
- `merges.txt` lines split on a literal space precisely because no token can contain one.
- `decoder.decode` joins **all** bytes and decodes once with `errors="replace"`. Decoding
  per-token would mangle every multi-byte character, since a token can hold a partial one.

**`Vocabulary.add_token` allocates from `self._next_id`, never from `len()`.** A gap in
the id space (hand-edited `vocab.json`, a partial load) makes `len()` name an id that is
already taken, and `_register` would replace that token while `token_to_id` kept pointing
at it. `_register` now raises `ValueError` on a duplicate id rather than overwriting, so
keep using it as the single write path into both dicts.

**`Vocabulary.byte_baseline()` assigns id == byte value.** That is why
`byte_encoder.encode()` output is already valid token ids before any merge exists. Merged
tokens start at 256.

**`merges` order is the merge ranking.** `BPEEncoder` builds `self.ranks` from list
position. Never sort or dedupe a merges list — `load_merges` preserves file order for
this reason.

**Normalization defaults to NFKC.** An earlier version of `main.py` chained
NFC→NFD→NFKC→NFKD, whose net effect was NFKD. Accented text now produces fewer byte
tokens than it used to (`é` stays one codepoint). Pass `normalization_form="NFKD"` to
reproduce the old output.

**The default pre-tokenizer pattern is not GPT-2's.** `DEFAULT_PATTERN` emits whitespace
as its own piece; `GPT2_PATTERN` (defined but unused) attaches a leading space to the
following word. This changes what BPE can learn — with the default, no merge can ever
span a space.

### SSE contract

`/process` emits four event types — `start`, `pre_token`, `progress`, `complete` — whose
field names any frontend depends on. `bpe_boundary` is `true` on `pre_token` and `false`
on `progress`, marking where BPE may not merge across. Preserve these names when adding
data; add fields rather than renaming.

## Other agent configs

An OpenAI Codex config (`~/.codex/config.toml`) and a Gemini CLI directory (`~/.gemini/`)
exist on this machine. To import MCP servers, slash commands, subagents, skills, or
instructions from them, reply `/import` to see what is importable, then
`/import --yes=<digest>` using the digest that scan prints. If `/import` is unavailable
here, run `claude import` from a terminal.
