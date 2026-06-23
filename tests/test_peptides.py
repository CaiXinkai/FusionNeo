from fusionneo.peptides import enumerate_junction_peptides


def test_all_peptides_cross_junction():
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    breakpoint = 10
    rows = enumerate_junction_peptides(sequence, breakpoint, lengths=[8, 9])
    assert rows
    assert all(row["peptide_start"] < breakpoint < row["peptide_end"] for row in rows)
    assert all(row["left_residues"] >= 1 and row["right_residues"] >= 1 for row in rows)


def test_expected_number_for_one_length():
    rows = enumerate_junction_peptides("A" * 30, 15, lengths=[9])
    assert len(rows) == 8

