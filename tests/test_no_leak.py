"""The acting player's inputs (and the search's decision) must not depend on hidden cards."""
import random

import torch

from ladis.bots import rule_bot
from ladis.engine import WhistBatch
from ladis.features import encode, encode_tokens
from ladis.model import ActorCritic
from ladis.search import SearchPolicy


def _twin_games(steps, seed):
    """Two games, identical from the acting player's seat, with different hidden hands."""
    torch.manual_seed(seed)
    A = WhistBatch(1).reset()
    for _ in range(steps):
        A.step(rule_bot(A))
    me = int(A.cur[0])
    owner = A.owner[0].clone()
    played = A.played_by[0].any(0)
    hidden = [c for c in range(52) if not played[c] and owner[c] != me]
    rng = random.Random(seed)
    swaps = 0
    for _ in range(200):  # swap same-suit hidden cards between two other players
        c, d = rng.sample(hidden, 2)
        if c // 13 == d // 13 and owner[c] != owner[d]:
            owner[c], owner[d] = owner[d].clone(), owner[c].clone()
            swaps += 1
    B = WhistBatch(1).reset(leader=A.first_leader, deal=owner.unsqueeze(0))
    for t in range(steps):
        B.step(A.hist_card[:, t])
    assert int(B.cur[0]) == me
    hidden_differs = not torch.equal(A.hands[0], B.hands[0])
    return A, B, hidden_differs


def test_actor_inputs_ignore_hidden_cards():
    differs = 0
    for seed in range(40):
        A, B, d = _twin_games(steps=random.Random(seed).randrange(0, 48), seed=seed)
        differs += d
        for hist in (False, True):
            assert torch.equal(encode(A, hist=hist), encode(B, hist=hist)), "actor input leaks hidden cards"
        ta, tb = encode_tokens(A), encode_tokens(B)
        for k in ta:
            assert torch.equal(ta[k], tb[k]), f"transformer token '{k}' leaks hidden cards"
        # sanity: the oracle (critic-only) input DOES see them, so the test has teeth
        if d:
            assert not torch.equal(encode(A, oracle=True), encode(B, oracle=True))
    assert differs > 20


def test_search_ignores_hidden_cards():
    models = (ActorCritic(64, 1).eval(), ActorCritic(arch="tf", d=32, layers=1, heads=2).eval())
    for model in models:
        for seed in range(6):
            A, B, d = _twin_games(steps=9 + 4 * seed, seed=100 + seed)
            assert d
            priors = ("uniform", "belief") if model.arch == "tf" else ("uniform",)
            for prior in priors:
                s1 = SearchPolicy(model, N=24, seed=7, prior=prior)
                s2 = SearchPolicy(model, N=24, seed=7, prior=prior)
                a1, a2 = s1(A), s2(B)
                assert torch.equal(a1, a2), f"search ({prior}) decision depends on hidden cards"
                # every card's estimated value must match too, not just the final pick
                assert torch.equal(s1.q.nan_to_num(-1), s2.q.nan_to_num(-1)), f"search ({prior}) values leak"


def test_search_survives_sampler_failures(monkeypatch):
    """If the belief-weighted sampler fails, the search must still only replay consistent layouts."""
    import ladis.search as S
    model = ActorCritic(arch="tf", d=32, layers=1, heads=2).eval()
    A, _, _ = _twin_games(steps=21, seed=5)
    calls = {"n": 0}
    real = S.sample_world_weighted

    def flaky(*a, **k):
        calls["n"] += 1
        return (None, 0.0) if calls["n"] % 3 == 0 else real(*a, **k)
    monkeypatch.setattr(S, "sample_world_weighted", flaky)
    sp = SearchPolicy(model, N=12, seed=3, prior="belief")
    a = sp(A)  # must not raise "illegal action" during the history replay
    assert A.legal_mask()[0, a[0]]
