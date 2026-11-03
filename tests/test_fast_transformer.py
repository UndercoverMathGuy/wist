"""The fast transformer path must be numerically identical to the original nn.TransformerEncoder path."""
import torch

from ladis.bots import rule_bot
from ladis.engine import WhistBatch
from ladis.model import ActorCritic, obs_index


def _states(model, steps_list, B=16, seed=0):
    torch.manual_seed(seed)
    out = []
    for steps in steps_list:
        env = WhistBatch(B).reset()
        for _ in range(steps):
            env.step(rule_bot(env))
        out.append((model.observe(env, oracle=True), env.legal_mask()))
    obs = {k: torch.cat([o[k] for o, _ in out]) for k in out[0][0]}
    legal = torch.cat([l for _, l in out])
    return obs, legal


def test_fast_matches_reference_mixed_lengths():
    torch.manual_seed(0)
    m = ActorCritic(arch="tf", d=64, layers=3, heads=4).eval()
    # randomise the zero-initialised head so the test is not trivially passing
    torch.nn.init.normal_(m.actor.head[-1].weight, std=0.5)
    obs, legal = _states(m, [0, 1, 5, 17, 30, 44, 51])
    with torch.no_grad():
        ref = m.actor.forward_reference(obs).masked_fill(~legal, -1e9)
        fast = m.actor(obs).masked_fill(~legal, -1e9)
    assert torch.allclose(ref, fast, atol=1e-4, rtol=1e-4), (ref - fast).abs().max()
    # every sub-batch length (as used by length-sorted micro-batches) must agree too
    for idx in (torch.arange(0, 16), torch.arange(48, 64), torch.arange(96, 112)):
        o = obs_index(obs, idx)
        with torch.no_grad():
            a = m.actor.forward_reference(o).masked_fill(~legal[idx], -1e9)
            b = m.actor(o).masked_fill(~legal[idx], -1e9)
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)


def test_fast_gradients_match_reference():
    torch.manual_seed(1)
    m = ActorCritic(arch="tf", d=32, layers=2, heads=2)
    torch.nn.init.normal_(m.actor.head[-1].weight, std=0.5)
    obs, legal = _states(m, [3, 26, 50], B=8)
    act = torch.distributions.Categorical(logits=m.actor.forward_reference(obs).masked_fill(~legal, -1e9)).sample()
    grads = []
    for fn in (m.actor.forward_reference, m.actor.forward):
        m.zero_grad()
        lg = fn(obs).masked_fill(~legal, -1e9)
        torch.nn.functional.cross_entropy(lg, act).backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.actor.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-5, rtol=1e-4), (grads[0] - grads[1]).abs().max()


def test_graph_padding_path_matches_reference():
    """The padded/bucketed inference path (used with CUDA graphs) must give identical logits.
    On CPU we run it with the eager core standing in for the captured graph."""
    import ladis.model as M
    torch.manual_seed(2)
    m = ActorCritic(arch="tf", d=32, layers=2, heads=2).eval()
    torch.nn.init.normal_(m.actor.head[-1].weight, std=0.5)
    obs, legal = _states(m, [0, 7, 33, 50], B=10)
    with torch.no_grad():
        ref = m.actor.forward_reference(obs).masked_fill(~legal, -1e9)
        m.actor._core_graph = m.actor._core
        old = M.torch.compiler.cudagraph_mark_step_begin
        M.torch.compiler.cudagraph_mark_step_begin = lambda: None
        try:
            got = m.actor._forward_graphed(obs).masked_fill(~legal, -1e9)
        finally:
            M.torch.compiler.cudagraph_mark_step_begin = old
            m.actor._core_graph = None
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-4), (ref - got).abs().max()
