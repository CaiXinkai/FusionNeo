from __future__ import annotations

from .common import clean_sequence


def junction_context(sequence: str, breakpoint: int, flank: int = 50) -> tuple[str, int]:
    sequence = clean_sequence(sequence)
    start = max(0, breakpoint - flank)
    end = min(len(sequence), breakpoint + flank)
    return sequence[start:end], breakpoint - start


def peptide_context(
    sequence: str, peptide_start: int, peptide_end: int, flank: int = 50
) -> tuple[str, int, int]:
    sequence = clean_sequence(sequence)
    context_start = max(0, peptide_start - flank)
    context_end = min(len(sequence), peptide_end + flank)
    return (
        sequence[context_start:context_end],
        peptide_start - context_start,
        peptide_end - context_start,
    )


def enumerate_junction_peptides(
    sequence: str,
    breakpoint: int,
    lengths: tuple[int, ...] | list[int] = (8, 9, 10, 11, 12, 13, 14),
    min_side_residues: int = 1,
) -> list[dict]:
    sequence = clean_sequence(sequence)
    if not 0 < breakpoint < len(sequence):
        return []
    peptides: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for length in lengths:
        first_start = breakpoint - length + min_side_residues
        last_start = breakpoint - min_side_residues
        for start in range(first_start, last_start + 1):
            end = start + length
            if start < 0 or end > len(sequence):
                continue
            left = breakpoint - start
            right = end - breakpoint
            if left < min_side_residues or right < min_side_residues:
                continue
            peptide = sequence[start:end]
            key = (peptide, start, end)
            if key in seen:
                continue
            seen.add(key)
            peptides.append(
                {
                    "peptide": peptide,
                    "peptide_length": length,
                    "peptide_start": start,
                    "peptide_end": end,
                    "junction_offset": left,
                    "left_residues": left,
                    "right_residues": right,
                }
            )
    return peptides


def enumerate_nonjunction_background(
    sequence: str,
    breakpoint: int,
    lengths: tuple[int, ...] | list[int],
    max_per_length: int = 20,
) -> list[dict]:
    sequence = clean_sequence(sequence)
    rows: list[dict] = []
    for length in lengths:
        candidates = []
        for start in range(0, len(sequence) - length + 1):
            end = start + length
            if not (start < breakpoint < end):
                candidates.append((start, end))
        if len(candidates) > max_per_length:
            step = len(candidates) / max_per_length
            candidates = [candidates[int(i * step)] for i in range(max_per_length)]
        for start, end in candidates:
            rows.append(
                {
                    "peptide": sequence[start:end],
                    "peptide_length": length,
                    "peptide_start": start,
                    "peptide_end": end,
                    "junction_offset": -1,
                    "left_residues": 0,
                    "right_residues": 0,
                }
            )
    return rows


def enumerate_protein_background(
    sequence: str,
    lengths: tuple[int, ...] | list[int],
    max_per_length: int = 20,
) -> list[dict]:
    sequence = clean_sequence(sequence)
    rows: list[dict] = []
    for length in lengths:
        candidates = list(range(0, len(sequence) - length + 1))
        if len(candidates) > max_per_length:
            step = len(candidates) / max_per_length
            candidates = [candidates[int(i * step)] for i in range(max_per_length)]
        for start in candidates:
            end = start + length
            rows.append(
                {
                    "peptide": sequence[start:end],
                    "peptide_length": length,
                    "peptide_start": start,
                    "peptide_end": end,
                    "junction_offset": -1,
                    "left_residues": 0,
                    "right_residues": 0,
                }
            )
    return rows
