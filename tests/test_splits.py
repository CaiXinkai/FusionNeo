from fusionneo.splits import stable_group_split


def test_same_group_never_crosses_splits():
    assert stable_group_split("EML4::ALK", 42) == stable_group_split("EML4::ALK", 42)


def test_split_is_known():
    assert stable_group_split("EWSR1::FLI1", 42) in {"train", "validation", "test"}

