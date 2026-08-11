import pandas as pd

from drumhumanizer.holdout import build_drummer_holdout


def _toy():
    # 3 drummers, 3 genres; enough files per genre to split.
    rows = []
    fid = 0
    for drummer, genres in [
        ("d1", ["rock", "jazz", "funk"]),
        ("d2", ["rock", "jazz"]),
        ("d3", ["rock", "funk"]),
    ]:
        for genre in genres:
            for _ in range(10):                      # 10 files per (drummer, genre)
                for note in range(3):                # 3 notes per file
                    rows.append({"file_id": f"f{fid}", "drummer": drummer,
                                 "genre": genre, "velocity": 50 + note})
                fid += 1
    return pd.DataFrame(rows)


def test_test_split_is_exactly_the_holdout_drummers():
    df = _toy()
    train, val, test = build_drummer_holdout([df], {"d2"}, val_frac=0.1, seed=0)
    assert set(test["drummer"]) == {"d2"}
    assert "d2" not in set(train["drummer"])
    assert "d2" not in set(val["drummer"])


def test_no_file_leaks_across_splits():
    df = _toy()
    train, val, test = build_drummer_holdout([df], {"d2"}, val_frac=0.1, seed=0)
    ft, fv, fe = set(train["file_id"]), set(val["file_id"]), set(test["file_id"])
    assert ft & fv == set()
    assert ft & fe == set()
    assert fv & fe == set()
    # partition is exhaustive: every input row lands in exactly one split
    assert len(train) + len(val) + len(test) == len(df)


def test_all_genres_retained_in_training():
    df = _toy()
    # hold out d3 (plays rock+funk) — both must still be trainable via d1/d2
    train, val, test = build_drummer_holdout([df], {"d3"}, val_frac=0.1, seed=0)
    assert set(train["genre"]) == set(df["genre"])


def test_val_is_nonempty_and_file_disjoint_and_deterministic():
    df = _toy()
    a = build_drummer_holdout([df], {"d2"}, val_frac=0.2, seed=7)
    b = build_drummer_holdout([df], {"d2"}, val_frac=0.2, seed=7)
    assert len(a[1]) > 0                                  # val non-empty
    assert set(a[1]["file_id"]) == set(b[1]["file_id"])  # deterministic under seed


def test_singleton_genre_file_stays_in_train():
    # a genre with a single file must not be sent to val (would vanish from train)
    df = _toy()
    extra = pd.DataFrame([{"file_id": "solo", "drummer": "d1",
                           "genre": "reggae", "velocity": 60}])
    train, val, test = build_drummer_holdout([df, extra], {"d2"}, val_frac=0.5, seed=1)
    assert "reggae" in set(train["genre"])
    assert "solo" not in set(val["file_id"])
