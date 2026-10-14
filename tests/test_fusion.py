import pytest

from mtg_search.fusion import rrf


def test_basic_rrf():
    a = [("x", 9.0), ("y", 5.0), ("z", 1.0)]
    b = [("y", 0.9), ("x", 0.8)]
    out = dict(rrf([a, b]))
    assert out["x"] == pytest.approx(1 / 61 + 1 / 62)
    assert out["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["z"] == pytest.approx(1 / 63)
    assert [i for i, _ in rrf([a, b])][-1] == "z"


def test_weights_and_limit():
    a = [("x", 1.0), ("y", 0.5)]
    b = [("y", 1.0), ("x", 0.5)]
    fused = rrf([a, b], weights=[1.0, 2.0])
    assert fused[0][0] == "y"
    assert len(rrf([a, b], limit=1)) == 1
    assert rrf([a, b], weights=[0.0, 1.0])[0][0] == "y"
    with pytest.raises(ValueError):
        rrf([a, b], weights=[1.0])


def test_duplicates_and_empty():
    assert rrf([]) == []
    out = dict(rrf([[("x", 1), ("x", 0.5), ("y", 0.1)]], k=0))
    assert out == {"x": 1.0, "y": 0.5}
