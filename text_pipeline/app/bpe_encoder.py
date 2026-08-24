"""BPE encoding: applying learned merges to one piece. NOT YET IMPLEMENTED.

TRAINING VS ENCODING -- THE ASYMMETRY
-------------------------------------
Training and encoding both "do BPE", but they answer different questions:

    bpe_trainer   which pairs are most frequent across the whole corpus?
                  -> runs once, sees everything, WRITES the rule list
    bpe_encoder   which rules apply to this one piece, in what order?
                  -> runs constantly, sees one piece, READS the rule list

Encoding never counts anything. Frequency was already decided at training time
and frozen into the merge order. This is what makes tokenization deterministic:
the same piece always produces the same ids, regardless of what surrounds it.

RANK IS THE WHOLE MECHANISM
---------------------------
Given merges [('t','h'), ('th','e'), ('e','r')], encoding "there" must apply
them in rank order, not in whatever order they happen to appear in the text:

    t h e r e
    ('t','h') is rank 0, ('e','r') is rank 2 -> apply rank 0 first
    th e r e
    ('th','e') is rank 1 -> apply it
    the r e
    ('e','r') no longer occurs; nothing left to merge
    -> ['the', 'r', 'e']

Had ('e','r') been applied first, 'th' + 'er' would never have combined into
'the' and the result would differ. Always pick the lowest-ranked pair present,
never the leftmost.
"""

from collections.abc import Iterable

from .vocabulary import Vocabulary


class BPEEncoder:
    """Applies a frozen merge list to individual pieces.

    Constructed once per Tokenizer and reused, because building `ranks` over a
    50,000-entry merge list on every call would dominate the cost of encoding.
    """

    def __init__(self, vocabulary: Vocabulary, merges: Iterable[tuple[str, str]]) -> None:
        self.vocabulary = vocabulary
        self.merges = list(merges)

        # Rank == position in merges. Lowest rank wins.
        #
        # This dict is the entire reason merge *order* is load-bearing: it
        # converts "the order the trainer discovered these pairs" into an
        # O(1) priority lookup. Sorting or deduplicating `merges` anywhere
        # upstream renumbers every rank and silently changes tokenization.
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    def encode_piece(self, byte_ids: list[int]) -> list[int]:
        """Collapse `byte_ids` into the fewest token ids the merges allow.

        Takes byte ids rather than text because pre-tokenization and byte
        encoding have already run -- see Tokenizer.encode. Returns ids again,
        so it is a drop-in replacement for the no-merge passthrough.

        Algorithm to implement:
          1. Map byte ids to their vocabulary symbols.
          2. Of all adjacent pairs, find the one with the lowest rank in
             self.ranks. If no pair is ranked, stop -- this piece is already
             as merged as the rules allow.
          3. Merge every occurrence of that pair, scanning left to right.
             Left to right matters for overlaps: merging ('a','a') in "aaa"
             gives 'aa' + 'a', not 'a' + 'aa'.
          4. Repeat from 2 until one symbol remains or no pair is ranked.
          5. Map the surviving symbols back to ids.

        A single-symbol piece needs no work: there is no adjacent pair, so
        step 2 stops immediately and the byte id passes through unchanged.

        Never merge across pieces -- the pre-tokenizer's boundaries are the
        whole reason it runs first.
        """
        raise NotImplementedError(
            "bpe_encoder.encode_piece needs merges from bpe_trainer"
        )
