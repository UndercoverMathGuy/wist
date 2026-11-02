"""Baseline bots (vectorised, run on the same device as the env).

* random_bot     – uniform random legal card.
* rule_bot       – "grandparent": standard Whist technique + perfect card counting
                   (knows which cards are masters, tracks voids). No signalling.
"""
from __future__ import annotations

import torch

from .engine import NC, NS, NR, WhistBatch

BIG = 10_000.0


def random_bot(env: WhistBatch, gen=None) -> torch.Tensor:
    L = env.legal_mask().float()
    return torch.multinomial(L, 1, generator=gen).squeeze(1)


def _argmin(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, score, torch.full_like(score, BIG)).argmin(1)


def _argmax(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, score, torch.full_like(score, -BIG)).argmax(1)


def rule_bot(env: WhistBatch, eps: float = 0.0, gen=None) -> torch.Tensor:
    B, dev = env.B, env.device
    ar = env.ar
    p = env.cur
    suit = env.suit.unsqueeze(0).expand(B, NC)
    rank = env.rank.unsqueeze(0).expand(B, NC).float()
    H = env.hands[ar, p]
    L = env.legal_mask()
    all_played = env.played_by.any(1)
    U = (~H) & (~all_played)  # unseen (held by the other three)
    U3 = U.reshape(B, NS, NR).int()
    higher_unseen = (torch.flip(torch.cumsum(torch.flip(U3, [2]), 2), [2]) - U3).reshape(B, NC)
    master = H & (higher_unseen == 0)
    n = env.n_in_trick
    partner = (p + 2) % NS
    lho = (p + 1) % NS
    rho = (p + 3) % NS
    opp_void = env.void[ar, lho] | env.void[ar, rho]  # (B,4)
    trumps_unseen = U[:, :NR].sum(1)
    my_len = H.reshape(B, NS, NR).sum(2)  # (B,4)
    rankf = rank

    low_any = _argmin(rankf + 0.01 * suit, L)

    # ---------------------------------------------------------------- LEAD
    my_trumps = my_len[:, 0]
    # trump lead with 5+
    tmask = H & (suit == 0)
    hi_t = _argmax(rankf, tmask)
    hi_t_class = higher_unseen[ar, hi_t]
    class_size = (tmask & (higher_unseen == hi_t_class.unsqueeze(1))).sum(1)
    trump_lead = torch.where((hi_t_class == 0) | (class_size >= 2), hi_t, _argmin(rankf, tmask))
    do_trump = (my_trumps >= 5) & (trumps_unseen > 0)
    # cash masters in side suits that opponents are not known void in (or no trumps out)
    safe_suit = (~opp_void) | (trumps_unseen == 0).unsqueeze(1)  # (B,4)
    safe_card = safe_suit.gather(1, suit)
    cash_mask = master & (suit != 0) & safe_card
    do_cash = cash_mask.any(1)
    cash = _argmax(rankf, cash_mask)
    # partner's suit
    pled = env.led_by[ar, partner].reshape(B, NS, NR).any(2)  # suits partner led
    psuit = pled & safe_suit & (my_len > 0)
    psuit[:, 0] = False
    pcard_mask = H & psuit.gather(1, suit)
    plen = my_len.gather(1, suit)
    pret_hi = _argmax(rankf, pcard_mask)
    pret_lo = _argmin(rankf, pcard_mask)
    pret_short = (pcard_mask.sum(1) <= 2)
    pret = torch.where(pret_short, pret_hi, pret_lo)
    do_pret = pcard_mask.any(1)
    # longest side suit
    side_len = my_len.clone().float()
    side_len[:, 0] = -1
    strength = (H.reshape(B, NS, NR).float() * torch.arange(NR, device=dev).float()).sum(2) / 100.0
    best_suit = (side_len + strength).argmax(1)
    has_side = (my_len[:, 1:] > 0).any(1)
    smask = H & (suit == best_suit.unsqueeze(1))
    s_hi = _argmax(rankf, smask)
    s_hi_class = higher_unseen[ar, s_hi]
    s_class_size = (smask & (higher_unseen == s_hi_class.unsqueeze(1))).sum(1)
    top_seq = (s_class_size >= 2) & (env.rank[s_hi] >= 9)
    # 4th best: sort ranks descending, take index 3
    sr = torch.where(smask, rankf, torch.full_like(rankf, -1.0))
    srt, sidx = sr.sort(1, descending=True)
    slen = smask.sum(1)
    fourth = sidx[:, 3]
    s_low = _argmin(rankf, smask)
    long_lead = torch.where(top_seq | (s_hi_class == 0), s_hi, torch.where(slen >= 4, fourth, s_low))
    long_lead = torch.where(has_side, long_lead, trump_lead)
    lead = torch.where(do_trump, trump_lead,
                       torch.where(do_cash, cash, torch.where(do_pret, pret, long_lead)))

    # ---------------------------------------------------------------- FOLLOW
    tc = env.trick_cards
    ws = env.winning_seat()
    wc = tc[ar, ws].clamp(min=0)
    ls = env.lead_suit.clamp(min=0)
    w_suit = env.suit[wc]
    w_rank = env.rank[wc]

    def value(cards_suit, cards_rank):
        return torch.where(cards_suit == 0, 100 + cards_rank,
                           torch.where(cards_suit == ls.unsqueeze(1), cards_rank, -1))

    card_val = value(suit, env.rank.unsqueeze(0).expand(B, NC))
    w_val = torch.where(w_suit == 0, 100 + w_rank, w_rank)
    beats = card_val > w_val.unsqueeze(1)
    partner_winning = (ws == partner) & (n > 0)
    # is the winning card unbeatable by the players still to act (ignoring ruffs unless known)?
    w_master = higher_unseen.reshape(B, NC)  # placeholder shape use below
    U_suit_w = U.reshape(B, NS, NR)[ar, w_suit]  # (B,13) unseen cards in winning suit
    higher_w = (U_suit_w & (torch.arange(NR, device=dev).unsqueeze(0) > w_rank.unsqueeze(1))).any(1)
    # ruff risk: LHO still to play (n==2) and void in lead suit with trumps maybe left
    lho_void_lead = env.void[ar, lho].gather(1, ls.unsqueeze(1)).squeeze(1)
    ruff_risk = (w_suit != 0) & lho_void_lead & (trumps_unseen > 0)
    partner_safe = partner_winning & ((n == 3) | (~higher_w & ~ruff_risk))

    can_follow = (L & (suit == ls.unsqueeze(1))).any(1) & (n > 0)
    beat_mask = L & beats
    can_beat = beat_mask.any(1)
    lowest_beat = _argmin(rankf + 0.01 * suit, beat_mask)
    # third hand high: lowest card of the equivalence class of my highest legal card
    hi_l = _argmax(rankf, L)
    hi_class = higher_unseen[ar, hi_l]
    third_high = _argmin(rankf, L & (higher_unseen == hi_class.unsqueeze(1)) & (suit == env.suit[hi_l].unsqueeze(1)))
    lead_card = tc[ar, env.leader].clamp(min=0)
    honour_led = env.rank[lead_card] >= 9

    f2 = torch.where(honour_led & can_beat, lowest_beat, low_any)
    third_beats = beats[ar, third_high]
    f3 = torch.where(partner_safe, low_any, torch.where(third_beats, third_high, low_any))
    f4 = torch.where(partner_winning, low_any, torch.where(can_beat, lowest_beat, low_any))
    follow = torch.where(n == 1, f2, torch.where(n == 2, f3, f4))

    # ---------------------------------------------------------------- VOID
    trump_beat = H & (suit == 0) & beats
    ruff = _argmin(rankf, trump_beat)
    can_ruff = trump_beat.any(1)
    side_cards = H & (suit != 0)
    lenc = my_len.gather(1, suit).float()
    dcost = rankf + 20.0 * master.float() + 3.0 * lenc + 100.0 * (suit == 0).float()
    # don't bare an honour: penalise discarding the low guard of K / Q
    discard = _argmin(dcost, L)
    void_play = torch.where(partner_safe | (partner_winning & (n == 3)), discard,
                            torch.where(can_ruff, ruff, discard))

    act = torch.where(n == 0, lead, torch.where(can_follow, follow, void_play))
    if eps > 0:
        rnd = random_bot(env, gen)
        coin = torch.rand(B, device=dev, generator=gen) < eps
        act = torch.where(coin, rnd, act)
    return act
