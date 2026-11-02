import torch
from ladis.engine import WhistBatch
from ladis.bots import random_bot, rule_bot
from ladis.arena import play


def ref_tricks(hist_card, hist_seat, first_leader):
    tricks = [0, 0]
    leader = first_leader
    hands_played = {}
    for t in range(13):
        cards = hist_card[4 * t:4 * t + 4]
        seats = hist_seat[4 * t:4 * t + 4]
        assert seats[0] == leader
        assert seats == [(leader + i) % 4 for i in range(4)]
        ls = cards[0] // 13
        best, bw = None, None
        for c, s in zip(cards, seats):
            v = 100 + c % 13 if c // 13 == 0 else (c % 13 if c // 13 == ls else -1)
            if best is None or v > best:
                best, bw = v, s
        tricks[bw % 2] += 1
        leader = bw
    return tricks


def check_follow(owner, hist_card, hist_seat):
    hands = {s: set(i for i in range(52) if owner[i] == s) for s in range(4)}
    for t in range(13):
        ls = hist_card[4 * t] // 13
        for i in range(4):
            c, s = hist_card[4 * t + i], hist_seat[4 * t + i]
            assert c in hands[s]
            if i > 0 and c // 13 != ls:
                assert not any(x // 13 == ls for x in hands[s]), "revoke"
            hands[s].remove(c)


def test_random_games():
    for bot in (random_bot, rule_bot):
        env = WhistBatch(256).reset()
        fl = env.first_leader.clone()
        owner = env.owner.clone()
        tr = play(env, [bot] * 4)
        assert (tr.sum(1) == 13).all()
        for b in range(256):
            hc, hs = env.hist_card[b].tolist(), env.hist_seat[b].tolist()
            check_follow(owner[b].tolist(), hc, hs)
            assert ref_tricks(hc, hs, int(fl[b])) == tr[b].tolist()


def test_hist_features():
    from ladis.features import encode, OBS_DIM
    from ladis.engine import NR, NS, NC
    env = WhistBatch(64).reset()
    for _ in range(23):
        env.step(rule_bot(env))
    o = encode(env, hist=True)[:, OBS_DIM:OBS_DIM + NR * NS * NC].view(64, NR, NS, NC)
    p = env.cur
    for b in range(64):
        for t in range(NR):
            for i in range(4):
                c = int(env.hist_card[b, 4 * t + i])
                if c < 0:
                    continue
                r = (int(env.hist_seat[b, 4 * t + i]) - int(p[b])) % 4
                assert o[b, t, r, c] == 1
    assert o.sum() == 64 * 23
