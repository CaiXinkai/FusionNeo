from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from Bio.Align import PairwiseAligner

from .common import clean_sequence


@dataclass(frozen=True)
class BreakpointCall:
    head_end: int
    tail_start: int
    breakpoint: int
    overlap: int
    gap: int
    head_identity: float
    tail_identity: float
    confidence: str
    method: str = "head_tail_local_alignment"

    def to_dict(self) -> dict:
        return asdict(self)


def _align_query_interval(reference: str, query: str) -> tuple[int, int, float]:
    aligner = PairwiseAligner(mode="local")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -1.0
    alignment = aligner.align(reference, query)[0]
    ref_blocks, query_blocks = alignment.aligned
    if len(query_blocks) == 0:
        raise ValueError("No local alignment found")
    q_start = int(query_blocks[0][0])
    q_end = int(query_blocks[-1][1])
    matches = 0
    aligned = 0
    for (r0, r1), (q0, q1) in zip(ref_blocks, query_blocks):
        width = min(int(r1 - r0), int(q1 - q0))
        matches += sum(
            reference[int(r0) + i] == query[int(q0) + i] for i in range(width)
        )
        aligned += width
    return q_start, q_end, matches / max(aligned, 1)


def infer_breakpoint(fusion: str, head: str, tail: str) -> BreakpointCall:
    fusion = clean_sequence(fusion)
    head = clean_sequence(head)
    tail = clean_sequence(tail)
    if not fusion or not head or not tail:
        raise ValueError("Fusion, head, and tail sequences must be non-empty")

    head_blocks = SequenceMatcher(None, head, fusion, autojunk=False).get_matching_blocks()
    tail_blocks = SequenceMatcher(None, tail, fusion, autojunk=False).get_matching_blocks()
    head_anchored = [block for block in head_blocks if block.b == 0 and block.size >= 8]
    tail_anchored = [
        block for block in tail_blocks if block.b + block.size == len(fusion) and block.size >= 8
    ]
    use_anchored = False
    if head_anchored and tail_anchored:
        head_block = max(head_anchored, key=lambda block: block.size)
        tail_block = max(tail_anchored, key=lambda block: block.size)
        candidate_head_end = head_block.size
        candidate_tail_start = tail_block.b
        use_anchored = abs(candidate_head_end - candidate_tail_start) <= 15
    if use_anchored:
        head_end = candidate_head_end
        tail_start = candidate_tail_start
        head_identity = 1.0
        tail_identity = 1.0
        method = "anchored_exact_match"
    else:
        _, head_end, head_identity = _align_query_interval(head, fusion)
        tail_start, _, tail_identity = _align_query_interval(tail, fusion)
        method = "head_tail_local_alignment"
    overlap = max(0, head_end - tail_start)
    gap = max(0, tail_start - head_end)
    breakpoint = round((head_end + tail_start) / 2)

    min_identity = min(head_identity, tail_identity)
    if min_identity >= 0.95 and overlap <= 3 and gap <= 3:
        confidence = "high"
    elif min_identity >= 0.80 and overlap <= 15 and gap <= 15:
        confidence = "medium"
    else:
        confidence = "low"
    return BreakpointCall(
        head_end=head_end,
        tail_start=tail_start,
        breakpoint=breakpoint,
        overlap=overlap,
        gap=gap,
        head_identity=head_identity,
        tail_identity=tail_identity,
        confidence=confidence,
        method=method,
    )
