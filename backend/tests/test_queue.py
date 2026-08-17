from app.store_docs import _autoclaim_messages, _envelope


def test_autoclaim_messages_tuple():
    claimed = ["0-1", [("1-0", {"job": '{"doc_id":"a"}'})], []]
    msgs = _autoclaim_messages(claimed)
    env = _envelope(msgs[0])
    assert env.msg_id == "1-0"
    assert env.job["doc_id"] == "a"


def test_autoclaim_messages_empty():
    assert _autoclaim_messages(None) == []
    assert _autoclaim_messages(("0-0", [])) == []
