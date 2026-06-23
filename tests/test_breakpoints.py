from fusionneo.breakpoints import infer_breakpoint


def test_infer_clean_breakpoint():
    head = "MKTAYIAKQRQISFVKSHFSRQ"
    tail = "GILGYTEHQVVSSDFNSDTHS"
    fusion = head[:12] + tail[7:]
    call = infer_breakpoint(fusion, head, tail)
    assert abs(call.breakpoint - 12) <= 1
    assert call.confidence == "high"
