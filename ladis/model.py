"""Policy/value networks.

Every model consumes an observation *dict* produced by `model.observe(env)`:
  flat  (B,D)    hand-crafted features (own hand + public info)            [actor]
  oth   (B,156)  the other three hands                     [critic only, training]
  hand/hcard/hrel/win  token ids (see features.encode_tokens)    [transformer actor]

arch="mlp": residual MLP actor on `flat` (v1/v2).
arch="tf" : transformer actor. Tokens = [CLS] + my 13 hand cards + up to 52 played cards.
            Each played-card token = card + suit + rank + who-played-it (relative seat)
            + trick number + position in trick + who-won-that-trick + off-suit/trump flags.
            CLS token = linear projection of the hand-crafted `flat` features.
            Pre-LayerNorm encoder (attention + MLP, residual). The policy scores each of my
            hand tokens ("pointer" head); illegal cards are masked.
The critic is always a residual MLP over flat + other hands (asymmetric actor-critic):
it is used only to compute training advantages, never to choose a card.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import NC, NR, NS
from .features import ORACLE_EXTRA, PAD_CARD, encode, encode_tokens, obs_dim


class ResBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.ln = nn.LayerNorm(w)
        self.l1 = nn.Linear(w, w * 2)
        self.l2 = nn.Linear(w * 2, w)

    def forward(self, x):
        return x + self.l2(torch.relu(self.l1(self.ln(x))))


class MLP(nn.Module):
    def __init__(self, din, dout, width=512, depth=4):
        super().__init__()
        self.inp = nn.Linear(din, width)
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(depth)])
        self.ln = nn.LayerNorm(width)
        self.out = nn.Linear(width, dout)

    def forward(self, x):
        return self.out(self.ln(self.blocks(self.inp(x))))


class TransformerActor(nn.Module):
    def __init__(self, flat_dim, d=256, layers=6, heads=8, ff=4):
        super().__init__()
        self.card = nn.Embedding(NC + 1, d)          # 52 = pad
        self.suit = nn.Embedding(NS + 1, d)
        self.rank = nn.Embedding(NR + 1, d)
        self.seat = nn.Embedding(NS + 1, d)          # who played (rel) / 4 = n.a.
        self.trick = nn.Embedding(NR, d)
        self.pos = nn.Embedding(NS, d)
        self.won = nn.Embedding(NS + 1, d)           # who won that trick (rel) / 4 = unfinished
        self.flag = nn.Embedding(3, d)               # 0 followed suit, 1 discard, 2 ruff
        self.kind = nn.Embedding(3, d)               # 0 CLS, 1 my hand, 2 history
        self.cls = nn.Linear(flat_dim, d)
        layer = nn.TransformerEncoderLayer(d, heads, d * ff, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("trick_idx", torch.arange(NC) // NS, persistent=False)
        self.register_buffer("pos_idx", torch.arange(NC) % NS, persistent=False)

    def _card_emb(self, c):
        pad = c == PAD_CARD
        s = torch.where(pad, NS, c // NR)
        r = torch.where(pad, NR, c % NR)
        return self.card(c) + self.suit(s) + self.rank(r)

    # ------------------------------------------------------------------ fast path
    # Same parameters/semantics as `forward_reference`, but:
    #  * exact length truncation: pad tokens are dropped (they are masked out of attention, so
    #    outputs at real positions are unchanged). History is cut to the longest real history in
    #    the batch (rounded up to a whole trick), hand to the largest hand in the batch;
    #  * hand-written pre-LN blocks on the nn.TransformerEncoder weights: one fused QKV matmul,
    #    F.scaled_dot_product_attention (fused kernel) with a boolean mask, no MHA wrapper;
    #  * the core is torch.compile-able (`compile_core`), fusing norms/GELU/residuals/embeddings.
    def forward(self, obs):
        hand = obs["hand"]
        hcard = obs["hcard"]
        with torch.no_grad():
            nh = int((hand != PAD_CARD).sum(1).max().item()) if hand.numel() else 0
            nl = int((hcard != PAD_CARD).sum(1).max().item()) if hcard.numel() else 0
        H = max(nh, 1)
        L = min(NC, max(NS, -(-nl // NS) * NS))  # whole tricks, at least one
        core = self._core_compiled if getattr(self, "_core_compiled", None) is not None else self._core
        score = core(hand[:, :H].long(), hcard[:, :L].long(), obs["hrel"][:, :L].long(), obs["win"].long(),
                     obs["flat"])
        B = hand.shape[0]
        logits = torch.full((B, NC + 1), -1e9, device=score.device, dtype=score.dtype)
        logits.scatter_(1, hand[:, :H].long(), score)
        return logits[:, :NC]

    def compile_core(self, **kw):
        self._core_compiled = torch.compile(self._core, dynamic=True, **kw)
        return self

    def _core(self, hand, hcard, hrel, win, flat):
        B, H = hand.shape
        L = hcard.shape[1]
        T = L // NS
        hpad = hcard == PAD_CARD
        lead = hcard.view(B, T, NS)[:, :, 0]
        lead_suit = torch.where(lead == PAD_CARD, -1, lead // NR).repeat_interleave(NS, 1)
        suit = torch.where(hpad, -1, hcard // NR)
        off = (suit != lead_suit) & ~hpad
        flag = torch.where(off & (suit == 0), 2, off.long())
        won = win[:, :T].repeat_interleave(NS, 1)
        h = (self._card_emb(hcard) + self.seat(hrel) + self.trick(self.trick_idx[:L]).unsqueeze(0)
             + self.pos(self.pos_idx[:L]).unsqueeze(0) + self.won(won) + self.flag(flag) + self.kind.weight[2])
        m = self._card_emb(hand) + self.kind.weight[1]
        c = (self.cls(flat.to(self.cls.weight.dtype)) + self.kind.weight[0]).unsqueeze(1)
        x = torch.cat([c, m, h], 1)
        keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=hand.device), hand != PAD_CARD, ~hpad], 1)
        attn_mask = keep[:, None, None, :]  # True = may attend
        for layer in self.enc.layers:
            x = x + self._attn(layer, layer.norm1(x), attn_mask)
            y = layer.norm2(x)
            x = x + layer.linear2(F.gelu(layer.linear1(y)))
        x = self.ln(x)
        hm = x[:, 1:1 + H]
        return self.head(torch.cat([hm, x[:, :1].expand(-1, H, -1)], -1)).squeeze(-1)  # (B,H)

    @staticmethod
    def _attn(layer, x, attn_mask):
        mha = layer.self_attn
        B, S, D = x.shape
        nh = mha.num_heads
        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).view(B, S, 3, nh, D // nh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, nh, S, dh)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return mha.out_proj(o.transpose(1, 2).reshape(B, S, D))

    # ------------------------------------------------------------------ reference (original) path
    def forward_reference(self, obs):
        hand = obs["hand"].long()
        hcard = obs["hcard"].long()
        hrel = obs["hrel"].long()
        win = obs["win"].long()
        B = hand.shape[0]
        hpad = hcard == PAD_CARD
        lead = hcard.view(B, NR, NS)[:, :, 0]
        lead_suit = torch.where(lead == PAD_CARD, -1, lead // NR).repeat_interleave(NS, 1)
        suit = torch.where(hpad, -1, hcard // NR)
        off = (suit != lead_suit) & ~hpad
        flag = torch.where(off & (suit == 0), 2, off.long())
        won = win.repeat_interleave(NS, 1)
        h = (self._card_emb(hcard) + self.seat(hrel) + self.trick(self.trick_idx).unsqueeze(0)
             + self.pos(self.pos_idx).unsqueeze(0) + self.won(won) + self.flag(flag) + self.kind.weight[2])
        m = self._card_emb(hand) + self.kind.weight[1]
        c = (self.cls(obs["flat"].to(self.cls.weight.dtype)) + self.kind.weight[0]).unsqueeze(1)
        x = torch.cat([c, m, h], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=hand.device), hand == PAD_CARD, hpad], 1)
        x = self.ln(self.enc(x, src_key_padding_mask=pad))
        hm = x[:, 1:1 + NR]
        score = self.head(torch.cat([hm, x[:, :1].expand(-1, NR, -1)], -1)).squeeze(-1)  # (B,13)
        logits = torch.full((B, NC + 1), -1e9, device=x.device, dtype=score.dtype)
        logits.scatter_(1, hand, score)
        return logits[:, :NC]


class ActorCritic(nn.Module):
    def __init__(self, width=512, depth=4, cwidth=512, cdepth=4, hist=False, arch="mlp",
                 d=256, layers=6, heads=8):
        super().__init__()
        self.hist = hist
        self.arch = arch
        D = obs_dim(hist)
        if arch == "tf":
            self.actor = TransformerActor(D, d, layers, heads)
        else:
            self.actor = MLP(D, 52, width, depth)
            nn.init.zeros_(self.actor.out.weight)
            nn.init.zeros_(self.actor.out.bias)
        self.critic = MLP(D + ORACLE_EXTRA, 1, cwidth, cdepth)

    @torch.no_grad()
    def observe(self, env, oracle=False) -> dict:
        x = encode(env, oracle=oracle, hist=self.hist)
        D = obs_dim(self.hist)
        o = {"flat": x[:, :D].half()}
        if oracle:
            o["oth"] = x[:, D:].bool()
        if self.arch == "tf":
            o.update(encode_tokens(env))
        return o

    def logits(self, obs, legal):
        if self.arch == "tf":
            lg = self.actor(obs)
        else:
            lg = self.actor(obs["flat"].to(self.actor.inp.weight.dtype))
        return lg.masked_fill(~legal, -1e9)

    def value(self, obs):
        x = torch.cat([obs["flat"].float(), obs["oth"].float()], 1)
        return self.critic(x).squeeze(-1)


def obs_index(obs: dict, idx) -> dict:
    return {k: v[idx] for k, v in obs.items()}
