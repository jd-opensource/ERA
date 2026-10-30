"""Unit tests for GpuPhaseGate request-driven predict⊥train protocol."""

from __future__ import annotations

import queue
import threading
import time

from omegaconf import OmegaConf

from joyrlinf.core.gpu_phase_gate import (
    GpuPhaseGate,
    NullGpuPhaseGate,
    build_gpu_phase_gate,
    resolve_gpu_phase_gate,
    resolve_phase_gate_granularity,
)


def test_resolve_defaults_and_aliases():
    cfg = OmegaConf.create({"actor": {"model": {"openpi": {}}}})
    assert resolve_gpu_phase_gate(cfg) == "none"
    assert resolve_phase_gate_granularity(cfg) == "training"

    cfg = OmegaConf.create({
        "actor": {"model": {"openpi": {
            "gpu_phase_gate": "per_chunk",
            "phase_gate_granularity": "training",
        }}},
    })
    assert resolve_gpu_phase_gate(cfg) == "per_chunk"
    assert resolve_phase_gate_granularity(cfg) == "training"

    cfg = OmegaConf.create({
        "actor": {"model": {"openpi": {"gpu_phase_gate": "off"}}},
    })
    assert resolve_gpu_phase_gate(cfg) == "none"


def test_build_null_when_none_or_missing_group():
    cfg = OmegaConf.create({
        "actor": {"model": {"openpi": {"gpu_phase_gate": "none"}}},
    })
    gate = build_gpu_phase_gate(
        cfg, role="actor", group=object(), my_rank=0, peer_rank=1,
    )
    assert isinstance(gate, NullGpuPhaseGate)

    cfg = OmegaConf.create({
        "actor": {"model": {"openpi": {"gpu_phase_gate": "per_chunk"}}},
    })
    gate = build_gpu_phase_gate(
        cfg, role="rollout", group=None, my_rank=0, peer_rank=1,
    )
    assert isinstance(gate, NullGpuPhaseGate)


class _QueuePeerGate(GpuPhaseGate):
    """GpuPhaseGate with in-process queues instead of dist.send/recv."""

    def __init__(
        self, role, out_q, in_q, granularity="microbatch",
        credits_per_predict=1,
    ):
        super().__init__(
            role=role,
            group=None,
            my_rank=0 if role == "actor" else 1,
            peer_rank=1 if role == "actor" else 0,
            granularity=granularity,
            credits_per_predict=credits_per_predict,
        )
        self._out_q = out_q
        self._in_q = in_q

    def _send(self, code: int, epoch: int = 0) -> None:
        self._out_q.put((code, epoch))

    def _recv(self) -> tuple[int, int]:
        while not self._stop.is_set():
            try:
                return self._in_q.get(timeout=0.05)
            except queue.Empty:
                continue
        raise RuntimeError("stopped")


def test_microbatch_gate_predict_train_mutex():
    a2r: queue.Queue[int] = queue.Queue()
    r2a: queue.Queue[int] = queue.Queue()
    actor = _QueuePeerGate("actor", a2r, r2a, granularity="microbatch")
    rollout = _QueuePeerGate("rollout", r2a, a2r, granularity="microbatch")
    actor.start_actor_service()

    order: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def mark(ev: str) -> None:
        with lock:
            order.append(ev)

    def rollout_fn():
        barrier.wait()
        for i in range(2):
            rollout.wait_predict_ok()
            mark(f"predict_begin_{i}")
            time.sleep(0.02)
            mark(f"predict_end_{i}")
            rollout.signal_train_ok()
            mark(f"env_{i}")
            time.sleep(0.01)
        rollout.wait_predict_ok()
        mark("bootstrap_begin")
        time.sleep(0.01)
        mark("bootstrap_end")
        rollout.signal_epoch_predicts_done()

    def actor_fn():
        barrier.wait()
        actor.enter_training()
        try:
            for i in range(3):
                actor.before_microbatch()
                mark(f"train_begin_{i}")
                time.sleep(0.02)
                mark(f"train_end_{i}")
                actor.after_microbatch()
        finally:
            actor.end_training_granularity_release()
            actor.exit_training()

    t_r = threading.Thread(target=rollout_fn)
    t_a = threading.Thread(target=actor_fn)
    t_r.start()
    t_a.start()
    t_r.join(timeout=5.0)
    t_a.join(timeout=5.0)
    actor.stop_actor_service()
    assert not t_r.is_alive() and not t_a.is_alive(), order

    # Reconstruct exclusive ownership: predict/bootstrap intervals must not
    # intersect train intervals.
    predict_open = False
    train_open = False
    for ev in order:
        if ev.startswith("predict_begin") or ev == "bootstrap_begin":
            assert not train_open, order
            predict_open = True
        elif ev.startswith("predict_end") or ev == "bootstrap_end":
            predict_open = False
        elif ev.startswith("train_begin"):
            assert not predict_open, order
            train_open = True
        elif ev.startswith("train_end"):
            train_open = False

    assert any(x.startswith("predict_begin") for x in order)
    assert any(x.startswith("train_begin") for x in order)


def test_multi_credit_releases_n_microbatches_per_predict():
    """One predict must release credits_per_predict microbatches, and the
    next predict must not start until those microbatches are drained."""
    a2r: queue.Queue[int] = queue.Queue()
    r2a: queue.Queue[int] = queue.Queue()
    actor = _QueuePeerGate(
        "actor", a2r, r2a, granularity="microbatch", credits_per_predict=4,
    )
    rollout = _QueuePeerGate(
        "rollout", r2a, a2r, granularity="microbatch", credits_per_predict=4,
    )
    actor.start_actor_service()

    order: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def mark(ev: str) -> None:
        with lock:
            order.append(ev)

    def rollout_fn():
        barrier.wait()
        for i in range(2):
            rollout.wait_predict_ok()
            mark(f"predict_begin_{i}")
            time.sleep(0.02)
            mark(f"predict_end_{i}")
            rollout.signal_train_ok()
            time.sleep(0.01)  # env step
        rollout.wait_predict_ok()
        mark("bootstrap_begin")
        time.sleep(0.01)
        mark("bootstrap_end")
        rollout.signal_epoch_predicts_done()

    def actor_fn():
        barrier.wait()
        actor.enter_training()
        try:
            for i in range(8):
                actor.before_microbatch()
                mark(f"train_begin_{i}")
                time.sleep(0.005)
                mark(f"train_end_{i}")
                actor.after_microbatch()
        finally:
            actor.end_training_granularity_release()
            actor.exit_training()

    t_r = threading.Thread(target=rollout_fn)
    t_a = threading.Thread(target=actor_fn)
    t_r.start()
    t_a.start()
    t_r.join(timeout=5.0)
    t_a.join(timeout=5.0)
    actor.stop_actor_service()
    assert not t_r.is_alive() and not t_a.is_alive(), order

    # Mutual exclusion: predict/bootstrap never overlaps a train interval.
    predict_open = False
    train_open = False
    for ev in order:
        if ev.startswith("predict_begin") or ev == "bootstrap_begin":
            assert not train_open, order
            predict_open = True
        elif ev.startswith("predict_end") or ev == "bootstrap_end":
            predict_open = False
        elif ev.startswith("train_begin"):
            assert not predict_open, order
            train_open = True
        elif ev.startswith("train_end"):
            train_open = False

    # Exactly 4 microbatches between predict 0 completing and predict 1
    # starting (credits drained before the next grant).
    i_p0_end = order.index("predict_end_0")
    i_p1_begin = order.index("predict_begin_1")
    mbs_between = [
        ev for ev in order[i_p0_end:i_p1_begin]
        if ev.startswith("train_begin")
    ]
    assert mbs_between == [
        "train_begin_0", "train_begin_1", "train_begin_2", "train_begin_3",
    ], order

    # Remaining 4 microbatches come from predict 1's credits.
    i_p1_end = order.index("predict_end_1")
    i_bs_begin = order.index("bootstrap_begin")
    mbs_after = [
        ev for ev in order[i_p1_end:i_bs_begin]
        if ev.startswith("train_begin")
    ]
    assert mbs_after == [
        "train_begin_4", "train_begin_5", "train_begin_6", "train_begin_7",
    ], order


def test_training_granularity_uses_one_credit_for_whole_train_burst():
    a2r: queue.Queue[int] = queue.Queue()
    r2a: queue.Queue[int] = queue.Queue()
    actor = _QueuePeerGate("actor", a2r, r2a, granularity="training")
    rollout = _QueuePeerGate("rollout", r2a, a2r, granularity="training")
    actor.start_actor_service()

    rollout.wait_predict_ok()
    rollout.signal_train_ok()
    actor.enter_training()
    actor.before_microbatch()

    next_predict_granted = threading.Event()

    def request_next_predict():
        rollout.wait_predict_ok()
        next_predict_granted.set()

    waiter = threading.Thread(target=request_next_predict)
    waiter.start()
    time.sleep(0.03)
    assert not next_predict_granted.is_set()

    # Later microbatches in the same run_training call must not consume more
    # credits or reopen the rollout grant.
    actor.after_microbatch()
    actor.before_microbatch()
    actor.after_microbatch()
    assert not next_predict_granted.is_set()

    actor.end_training_granularity_release()
    actor.exit_training()
    waiter.join(timeout=1.0)
    actor.stop_actor_service()
    assert next_predict_granted.is_set()


def test_messages_before_enter_training_are_not_lost():
    """TRAIN_OK / EPOCH_DONE arriving while the actor is still in
    passthrough (batch recv / advantage window) must be recorded in the
    epoch ledger, so the subsequent training neither loses credits nor
    deadlocks when the whole feed epoch finished early."""
    a2r: queue.Queue[int] = queue.Queue()
    r2a: queue.Queue[int] = queue.Queue()
    actor = _QueuePeerGate("actor", a2r, r2a, granularity="microbatch")
    rollout = _QueuePeerGate("rollout", r2a, a2r, granularity="microbatch")
    actor.start_actor_service()

    # Entire feed epoch (epoch 1: 3 predicts + done) completes while the
    # actor is still outside run_training (grants are passthrough).
    for _ in range(3):
        rollout.wait_predict_ok()
        rollout.signal_train_ok()
    rollout.wait_predict_ok()
    rollout.signal_epoch_predicts_done()

    # Give the service thread time to ingest all messages.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        with actor._cv:
            if 1 in actor._done_epochs and actor._credits_by_epoch.get(1) == 3:
                break
        time.sleep(0.01)
    with actor._cv:
        assert actor._credits_by_epoch.get(1) == 3, actor._credits_by_epoch
        assert 1 in actor._done_epochs

    done = threading.Event()

    def actor_fn():
        actor.enter_training()  # default feed epoch = 1
        try:
            for _ in range(5):  # more MBs than credits: needs EPOCH_DONE too
                actor.before_microbatch()
                actor.after_microbatch()
        finally:
            actor.end_training_granularity_release()
            actor.exit_training()
        done.set()

    t = threading.Thread(target=actor_fn)
    t.start()
    t.join(timeout=3.0)
    actor.stop_actor_service()
    assert done.is_set(), "actor deadlocked despite early EPOCH_DONE"


def test_stale_epoch_entries_are_pruned():
    """set_training_step(N) makes enter_training consume epoch N+1 only;
    older epochs' leftover credits must be dropped."""
    a2r: queue.Queue[int] = queue.Queue()
    r2a: queue.Queue[int] = queue.Queue()
    actor = _QueuePeerGate("actor", a2r, r2a, granularity="microbatch")
    with actor._cv:
        actor._credits_by_epoch = {1: 4, 2: 2}
        actor._done_epochs = {1}
    actor.set_training_step(1)  # feed epoch 2
    actor.enter_training()
    with actor._cv:
        assert actor._feed_epoch == 2
        assert actor._credits_by_epoch == {2: 2}
        assert actor._done_epochs == set()
    actor.exit_training()


def test_final_step_without_rollout_does_not_deadlock():
    """Last training step has no concurrent rollout epoch: after
    mark_no_more_predicts() microbatches must run without waiting for
    predict credits (regression test for the shutdown deadlock)."""
    a2r: queue.Queue[int] = queue.Queue()
    r2a: queue.Queue[int] = queue.Queue()
    actor = _QueuePeerGate("actor", a2r, r2a, granularity="microbatch")
    actor.start_actor_service()

    done = threading.Event()

    def actor_fn():
        actor.mark_no_more_predicts()
        actor.enter_training()
        try:
            for _ in range(4):
                actor.before_microbatch()
                actor.after_microbatch()
        finally:
            actor.end_training_granularity_release()
            actor.exit_training()
        done.set()

    t = threading.Thread(target=actor_fn)
    t.start()
    t.join(timeout=3.0)
    actor.stop_actor_service()
    assert done.is_set(), "actor deadlocked on final step without rollout"


def test_resolve_credits_per_predict():
    from joyrlinf.core.gpu_phase_gate import resolve_credits_per_predict

    cfg = OmegaConf.create({
        "cluster": {
            "num_nodes": 1,
            "component_placement": {
                "actor": "8-15", "env": "0-7", "rollout": "0-7",
            },
        },
        "env": {"train": {"total_num_envs": 1024}},
        "actor": {"micro_batch_size": 32},
    })
    # 1024 envs / 8 actors / mbs 32 = 4 microbatches per predict.
    assert resolve_credits_per_predict(cfg) == 4

    cfg.actor.micro_batch_size = 128
    assert resolve_credits_per_predict(cfg) == 1

    # Degenerate config falls back to 1.
    bad = OmegaConf.create({})
    assert resolve_credits_per_predict(bad) == 1


def test_null_gate_is_noop():
    g = NullGpuPhaseGate()
    g.set_pre_yield_hook(lambda: None)
    g.start_actor_service()
    g.enter_training()
    g.before_microbatch()
    g.after_microbatch()
    g.wait_predict_ok()
    g.signal_train_ok()
    g.signal_epoch_predicts_done()
    g.exit_training()
    g.stop_actor_service()
    assert g.training_credit_wait_seconds() == 0.0


def test_actor_credit_wait_is_measured_and_reset():
    actor = GpuPhaseGate(
        role="actor",
        group=None,
        my_rank=0,
        peer_rank=1,
        granularity="microbatch",
    )
    actor.enter_training()

    def release_credit():
        time.sleep(0.03)
        with actor._cv:
            actor._credits_by_epoch[actor._feed_epoch] = 1
            actor._cv.notify_all()

    release = threading.Thread(target=release_credit)
    release.start()
    actor.before_microbatch()
    release.join(timeout=1.0)
    assert actor.training_credit_wait_seconds() >= 0.02
    actor.after_microbatch()
    actor.exit_training()

    actor.enter_training()
    assert actor.training_credit_wait_seconds() == 0.0
    actor.mark_no_more_predicts()
    actor.before_microbatch()
    assert actor.training_credit_wait_seconds() < 0.02
    actor.after_microbatch()
    actor.exit_training()


def test_pre_yield_hook_runs_only_when_credits_drained():
    """The pre-yield hook fires exactly when the actor is about to reopen
    the grant (feed-epoch credits drained), before the peer's next predict
    can be granted — never mid-burst while credits remain."""
    a2r: queue.Queue[int] = queue.Queue()
    r2a: queue.Queue[int] = queue.Queue()
    actor = _QueuePeerGate(
        "actor", a2r, r2a, granularity="microbatch", credits_per_predict=2,
    )
    rollout = _QueuePeerGate(
        "rollout", r2a, a2r, granularity="microbatch", credits_per_predict=2,
    )
    actor.start_actor_service()

    calls: list[str] = []
    actor.set_pre_yield_hook(lambda: calls.append("release"))

    try:
        actor.set_training_step(0)
        actor.enter_training()

        rollout.wait_predict_ok()
        rollout.signal_train_ok()  # feed epoch 1: +2 credits

        actor.before_microbatch()
        actor.after_microbatch()
        assert calls == [], "hook must not fire while credits remain"

        actor.before_microbatch()
        actor.after_microbatch()
        assert calls == ["release"], "hook must fire once credits drained"

        # The grant reopened after the hook: next predict is served.
        rollout.wait_predict_ok()
        rollout.signal_epoch_predicts_done()
    finally:
        actor.exit_training()
        actor.stop_actor_service()

    # exit_training also runs the hook (leftover-credit corner).
    assert calls == ["release", "release"]
