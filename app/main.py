"""The HTTP layer: one endpoint that narrates tokenization over SSE.

WHAT THIS FILE IS AND IS NOT
----------------------------
It is routing plus streaming. Every piece of tokenization logic lives in
app/tokenizer.py and the modules behind it; this file imports one name,
`Tokenizer`, and otherwise only reshapes an EncodeResult into events.

Keep it that way. If you find yourself computing something about tokens here, it
belongs in the tokenizer package, where it can be tested without a server.

THE STREAMING IS THEATRE, AND THAT IS THE POINT
-----------------------------------------------
Tokenization is far too fast to watch -- it finishes in microseconds. So the
work completes *synchronously, up front*, and the generator then walks the
finished result slowly, pausing STEP_DELAY_SECONDS between events, so a human
can see pieces and tokens appear one at a time.

Two consequences worth understanding:

  * The `stage` labels are decorative. By the time any event is sent, all the
    real work is already done -- see stages.py.
  * An error inside encode() surfaces as an ordinary 500 *before* the stream
    opens, rather than mid-stream. That is the easier failure mode: a
    half-delivered SSE stream cannot retroactively become an error response.
    It is also what you will see once merges exist while BPEEncoder is still a
    stub -- the NotImplementedError lands on the TOKENIZER.encode(...) line.

WHY SSE RATHER THAN WEBSOCKETS
------------------------------
The traffic is entirely one-directional: the server narrates, the client
listens. Server-Sent Events are plain HTTP, need no handshake, and reconnect on
their own. A WebSocket would add a bidirectional channel nothing here uses.
"""

import asyncio
import json

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .stages import choose_stage
from .tokenizer import Tokenizer

app = FastAPI()

# One tokenizer for the whole process. No merges yet, so every UTF-8 byte
# is one token — identical behavior to the old inline code.
#
# Built at import time rather than per request: constructing the 256-token
# baseline and compiling the pre-tokenizer pattern is wasted work to repeat.
# Safe to share because Tokenizer is stateless across calls -- encode() reads
# its configuration and returns a fresh EncodeResult, mutating nothing. Once a
# trained vocabulary exists on disk, this becomes Tokenizer.load(<dir>).
TOKENIZER = Tokenizer()

# Pure presentation. Every event waits this long, so a request takes roughly
# (1 + pieces + tokens) * this -- a 2,000-byte input needs ~20 seconds. Lower it
# for real use; the delay exists only to make the pipeline watchable.
STEP_DELAY_SECONDS = 0.01  # visual demonstration only


class UserInput(BaseModel):
    """Request body: {"text": "..."}. Pydantic rejects anything else with a 422."""

    text: str


@app.post("/process")
async def process_text(user_input: UserInput):
    """Tokenize `text` and stream the result step by step.

    Emits four event types. Their field names are a contract with the frontend
    -- add fields rather than renaming them:

        start       once, up front: normalized text, all pre-tokens, totals
        pre_token   once per piece: a BPE boundary is being entered
        progress    once per token: which token, where, and overall progress
        complete    once, at the end: final counts
    """
    # All tokenization happens here, synchronously, before streaming starts.
    result = TOKENIZER.encode(user_input.text)

    async def event_generator():
        # sse_starlette turns each yielded dict into an SSE frame. Because this
        # is a generator, a client disconnecting simply stops the iteration --
        # there is nothing to clean up.
        previous_stage = None

        def next_stage() -> str:
            """Advance the cosmetic label, remembering it so it is not repeated."""
            nonlocal previous_stage
            previous_stage = choose_stage(previous_stage)
            return previous_stage

        # A frontend can render the complete picture from this one event; the
        # rest of the stream merely animates it.
        yield {
            "event": "start",
            "data": json.dumps({
                "stage": next_stage(),
                "input_text": result.original_text,
                # Sent alongside input_text because normalization can change
                # the text, and that difference is worth seeing.
                "normalized_text": result.normalized_text,
                "pre_tokens": result.pre_tokens,
                "pre_token_count": result.pre_token_count,
                "total_byte_tokens": result.total_tokens,
                "vocab_size": TOKENIZER.vocab_size,
            }),
        }
        await asyncio.sleep(STEP_DELAY_SECONDS)

        # Bound to locals: total_tokens is a property that re-sums every piece
        # on each access, and it is read once per token below.
        total = result.total_tokens
        position = 0

        for piece_index, piece in enumerate(result.pieces, start=1):
            # Announces a wall. bpe_boundary=True marks where merging stops --
            # the structural fact this whole visualization exists to show.
            yield {
                "event": "pre_token",
                "data": json.dumps({
                    "stage": next_stage(),
                    "pre_token_index": piece_index,
                    "pre_token": piece.text,
                    "pre_token_token_count": piece.token_count,
                    "bpe_boundary": True,
                }),
            }
            await asyncio.sleep(STEP_DELAY_SECONDS)

            for local_index, token_id in enumerate(piece.token_ids, start=1):
                position += 1

                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "stage": next_stage(),

                        # Where this token sits inside its own piece.
                        "pre_token": piece.text,
                        "pre_token_index": piece_index,
                        "pre_token_position": local_index,
                        "pre_token_total_tokens": piece.token_count,

                        # The token itself. Untrained, this id IS the byte
                        # value; after training it can stand for many bytes.
                        "token_id": token_id,
                        "token_position": position,

                        # Progress across the whole input. No division guard is
                        # needed: total is 0 only when there are no pieces, and
                        # then this loop never runs.
                        "tokens_processed": position,
                        "total_tokens": total,
                        "remaining_tokens": total - position,
                        "progress_percent": round(position / total * 100, 2),

                        # False here, True on pre_token: this is an ordinary
                        # token inside a piece, not a boundary between pieces.
                        "bpe_boundary": False,
                    }),
                }
                await asyncio.sleep(STEP_DELAY_SECONDS)

        # A terminal event, so a client can tell a finished stream from a
        # dropped connection. Note that "stage" is a fixed string here rather
        # than a random label.
        yield {
            "event": "complete",
            "data": json.dumps({
                "stage": "tokenization_complete",
                "pre_token_count": result.pre_token_count,
                "total_byte_tokens": total,
                "vocab_size": TOKENIZER.vocab_size,
            }),
        }

    return EventSourceResponse(event_generator())
