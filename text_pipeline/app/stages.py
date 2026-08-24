"""Cosmetic stage labels for the SSE stream.

Nothing here affects tokenization. These are the verbs that flicker past in the
UI while the pipeline is narrated, and they are picked at random -- a label
saying "Normalizing" does not mean normalization is happening at that moment.
Tokenization has in fact already finished before the first event is sent; see
main.py.

Kept out of main.py so that file reads as routing plus streaming rather than
being half filled by a word list.
"""

import random

# dict.fromkeys() removes duplicates while preserving insertion order, which a
# set would not. Order is only cosmetic here, but keeping it stable makes the
# list easy to scan and edit.
PROCESSING_STAGES = list(dict.fromkeys([
    "Reading",
    "Scanning",
    "Parsing",
    "Analyzing",
    "Processing",
    "Normalizing",
    "Encoding",
    "Decoding",
    "Mapping",
    "Converting",
    "Transforming",
    "Validating",
    "Checking",
    "Calculating",
    "Computing",
    "Indexing",
    "Tokenizing",
    "Segmenting",
    "Filtering",
    "Sorting",
    "Matching",
    "Resolving",
    "Detecting",
    "Extracting",
    "Generating",
    "Building",
    "Preparing",
    "Loading",
    "Buffering",
    "Optimizing",
    "Synchronizing",
    "Updating",
    "Inspecting",
    "Evaluating",
    "Verifying",
    "Classifying",
    "Structuring",
    "Organizing",
    "Aggregating",
    "Compiling",
    "Finalizing",
    "Assembling",
    "Refining",
    "Completing",
    "Executing",
    "Dispatching",
    "Returning",
]))


def choose_stage(previous_stage: str | None) -> str:
    """Pick a label, never the one just used.

    Excluding the previous label is the point: consecutive identical labels
    read as a frozen UI, so the stream looks stalled even while it is working.
    Pass None for the first call, when there is no previous label to avoid.
    """
    candidates = [s for s in PROCESSING_STAGES if s != previous_stage]
    return random.choice(candidates)
