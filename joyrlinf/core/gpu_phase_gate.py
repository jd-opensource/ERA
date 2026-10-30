"""Same-GPU phase gate: predict ⊥ train; cached env may overlap train.

Colocated async MPS (EnvRollout + Actor on one physical GPU). Uses a
dedicated Gloo pair group — never the batch tensor pipe.

Request-driven protocol (avoids deadlock while Actor blocks on batch recv)::

  Rollout:  send PREDICT_REQ(e) → recv PREDICT_OK → predict
            → send TRAIN_OK(e) | EPOCH_DONE(e) → chunk_step (or finish epoch)
  Actor:    recv PREDICT_REQ → (wait until feed-epoch credits drained & idle)
            → send PREDICT_OK → recv TRAIN_OK(e) | EPOCH_DONE(e)
            → ledger[e] += credits_per_predict (or mark e done)

Every message carries the rollout epoch index ``e``. The actor keeps a
per-epoch credit ledger: training step N consumes only credits of its feed
epoch N+1, and messages arriving while the actor is still outside
``run_training`` (batch recv / advantage window) are recorded instead of
dropped. An EPOCH_DONE that lands before ``enter_training`` therefore still
releases that step's microbatches — without this, a feed epoch finishing
inside the passthrough window would deadlock the whole job.

Each finished predict yields ``credits_per_predict`` micro-batch credits
(auto: envs_per_gpu / actor.micro_batch_size — one predict produces
envs_per_gpu samples, i.e. that many micro-batches worth of train data).
The next predict is granted only once the Actor has consumed all feed-epoch
credits and left its compute window, preserving predict ⊥ train.

While Actor is outside ``run_training``, grants are immediate (passthrough)
so step-1 collect can run while the main thread waits on ``recv_batch``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Literal, Optional

import torch
import torch.distributed as dist

from joyrlinf.core.launcher import parse_placement

logger = logging.getLogger(__name__)

MSG_PREDICT_REQ = 1
MSG_PREDICT_OK = 2
MSG_TRAIN_OK = 3
MSG_EPOCH_DONE = 4


def resolve_gpu_phase_gate(cfg) -> str:
    openpi = None
    if hasattr(cfg, "get"):
        actor = cfg.get("actor")
        if actor is not None and hasattr(actor, "get"):
            model = actor.get("model")
            if model is not None and hasattr(model, "get"):
                openpi = model.get("openpi")
    mode = "none"
    if openpi is not None and hasattr(openpi, "get"):
        mode = str(openpi.get("gpu_phase_gate", "none")).strip().lower()
    if mode in ("off", "false", "0"):
        mode = "none"
    if mode not in ("none", "per_chunk"):
        raise ValueError(
            "openpi.gpu_phase_gate must be 'none' or 'per_chunk', "
            f"got {mode!r}"
        )
    return mode


def resolve_phase_gate_granularity(cfg) -> str:
    openpi = None
    if hasattr(cfg, "get"):
        actor = cfg.get("actor")
        if actor is not None and hasattr(actor, "get"):
            model = actor.get("model")
            if model is not None and hasattr(model, "get"):
                openpi = model.get("openpi")
    # One training burst is the efficient default for embodied workloads:
    # actor compute runs while the paired process is inside the much longer
    # environment step, without repeatedly handing the GPU back between
    # microbatches.  ``microbatch`` remains available for compute-heavy actors.
    gran = "training"
    if openpi is not None and hasattr(openpi, "get"):
        gran = str(
            openpi.get("phase_gate_granularity", "training")
        ).strip().lower()
    if gran not in ("microbatch", "training"):
        raise ValueError(
            "openpi.phase_gate_granularity must be 'microbatch' or "
            f"'training', got {gran!r}"
        )
    return gran


class GpuPhaseGate:
    """Cross-process GPU compute mutex between paired EnvRollout and Actor."""

    def __init__(
        self,
        *,
        role: Literal["rollout", "actor"],
        group,
        my_rank: int,
        peer_rank: int,
        granularity: str = "microbatch",
        credits_per_predict: int = 1,
    ) -> None:
        self.role = role
        self.group = group
        self.my_rank = int(my_rank)
        self.peer_rank = int(peer_rank)
        self.granularity = granularity
        self.credits_per_predict = max(1, int(credits_per_predict))
        self._stop = threading.Event()
        self._mode = "passthrough"  # passthrough | training
        self._lock = threading.Lock()
        self._computing = False  # actor mid micro-batch forward/backward
        self._can_grant = threading.Event()
        self._can_grant.set()  # idle → grants allowed
        # Per-epoch credit ledger (actor side); guarded by _cv. Training step
        # N consumes credits/done of its feed epoch N+1 only.
        self._cv = threading.Condition()
        self._credits_by_epoch: dict[int, int] = {}
        self._done_epochs: set[int] = set()
        self._feed_epoch = 0
        # Feed epoch of the next enter_training; runner overrides via
        # set_training_step, otherwise auto-increments (unit tests).
        self._next_feed_epoch = 1
        # Set on the final training step: the paired rollout collects no
        # further epoch, so no credits will ever arrive again.
        self._no_more_predicts = False
        self._training_gate_armed = False
        self._training_credit_wait_s = 0.0
        # Rollout-side current epoch index; runner overrides via begin_epoch.
        self._rollout_epoch = 1
        self._service: Optional[threading.Thread] = None
        # Actor-side: runs on the training thread right before the grant
        # reopens for the paired rollout's next predict (e.g. empty_cache).
        self._pre_yield_hook = None

    def set_pre_yield_hook(self, hook) -> None:
        """Actor-side: register ``hook()`` to run before yielding the GPU.

        Invoked on the training thread at every yield point — feed-epoch
        credits drained after a microbatch, end of a training-granularity
        window, and ``exit_training`` — strictly before ``_can_grant`` is
        set. The hook must return quickly (it delays the grant); offload
        slow work such as ``empty_cache`` to a background thread.
        """
        self._pre_yield_hook = hook

    def _run_pre_yield_hook(self) -> None:
        if self._pre_yield_hook is None:
            return
        try:
            self._pre_yield_hook()
        except Exception:
            logger.exception("GpuPhaseGate pre-yield hook failed")

    def _send(self, code: int, epoch: int = 0) -> None:
        tensor = torch.tensor([code, epoch], dtype=torch.long)
        dist.send(tensor, dst=self.peer_rank, group=self.group)

    def _recv(self) -> tuple[int, int]:
        tensor = torch.zeros(2, dtype=torch.long)
        dist.recv(tensor, src=self.peer_rank, group=self.group)
        return int(tensor[0].item()), int(tensor[1].item())

    # ------------------------------------------------------------------
    # Actor service
    # ------------------------------------------------------------------

    def start_actor_service(self) -> None:
        if self.role != "actor":
            raise RuntimeError("start_actor_service is actor-only")
        if self._service is not None:
            return
        self._stop.clear()
        self._service = threading.Thread(
            target=self._actor_service_loop,
            name=f"gpu-phase-gate-{self.my_rank}",
            daemon=True,
        )
        self._service.start()
        logger.info(
            "GpuPhaseGate actor service started rank=%d peer=%d gran=%s "
            "credits_per_predict=%d",
            self.my_rank, self.peer_rank, self.granularity,
            self.credits_per_predict,
        )

    def stop_actor_service(self) -> None:
        if self._service is None:
            return
        self._stop.set()
        self._can_grant.set()
        with self._cv:
            self._cv.notify_all()
        self._service.join(timeout=5.0)
        self._service = None

    def _actor_service_loop(self) -> None:
        while not self._stop.is_set():
            try:
                code, _ = self._recv()
            except Exception:
                if self._stop.is_set():
                    break
                logger.exception("GpuPhaseGate recv failed")
                break
            if code != MSG_PREDICT_REQ:
                logger.warning(
                    "GpuPhaseGate expected PREDICT_REQ, got %s", code,
                )
                continue

            # Wait until main is not inside a micro-batch compute window.
            while not self._stop.is_set():
                if self._can_grant.wait(timeout=0.05):
                    break
            if self._stop.is_set():
                break

            try:
                self._send(MSG_PREDICT_OK)
                done_code, done_epoch = self._recv()
            except Exception:
                if self._stop.is_set():
                    break
                logger.exception("GpuPhaseGate grant/ack failed")
                break

            # Record unconditionally: messages landing while the actor is
            # still outside run_training (batch recv / advantage window)
            # belong to the epoch that will feed the upcoming training step
            # and must not be dropped.
            if done_code == MSG_EPOCH_DONE:
                with self._cv:
                    self._done_epochs.add(done_epoch)
                    self._cv.notify_all()
                continue
            if done_code != MSG_TRAIN_OK:
                logger.warning(
                    "GpuPhaseGate expected TRAIN_OK/EPOCH_DONE, got %s",
                    done_code,
                )
                continue
            with self._cv:
                self._credits_by_epoch[done_epoch] = (
                    self._credits_by_epoch.get(done_epoch, 0)
                    + self.credits_per_predict
                )
                self._cv.notify_all()

    def set_training_step(self, step: int) -> None:
        """Actor-side: declare the runner step for the next enter_training.

        Training step N is fed by rollout epoch N + 1.
        """
        if self.role != "actor":
            return
        with self._cv:
            self._next_feed_epoch = int(step) + 1

    def enter_training(self) -> None:
        if self.role != "actor":
            return
        with self._lock:
            self._training_credit_wait_s = 0.0
        with self._lock:
            self._mode = "training"
            self._training_gate_armed = False
            self._computing = False
        with self._cv:
            self._feed_epoch = self._next_feed_epoch
            self._next_feed_epoch = self._feed_epoch + 1
            # Drop stale ledger entries from epochs that fed earlier steps.
            self._credits_by_epoch = {
                e: c for e, c in self._credits_by_epoch.items()
                if e >= self._feed_epoch
            }
            self._done_epochs = {
                e for e in self._done_epochs if e >= self._feed_epoch
            }
        # Allow the first overlapping predict immediately.
        self._can_grant.set()

    def mark_no_more_predicts(self) -> None:
        """Actor-side: disarm credit gating for the final training step.

        The actor's step-N training is credited by rollout epoch N+1's
        predicts. The last step has no such epoch, so without this the first
        ``before_microbatch`` would wait forever and deadlock shutdown.
        """
        if self.role != "actor":
            return
        with self._cv:
            self._no_more_predicts = True
            self._cv.notify_all()
        logger.info(
            "GpuPhaseGate rank=%d: no more predicts — credit gating disarmed",
            self.my_rank,
        )

    def exit_training(self) -> None:
        if self.role != "actor":
            return
        with self._lock:
            self._mode = "passthrough"
            self._computing = False
            self._training_gate_armed = False
        with self._cv:
            self._cv.notify_all()
        # Covers leftover credits (more credits than microbatches): the loop
        # never reopened the grant, so release before passthrough grants.
        self._run_pre_yield_hook()
        self._can_grant.set()

    def _feed_done_locked(self) -> bool:
        return self._no_more_predicts or self._feed_epoch in self._done_epochs

    def _feed_credits_locked(self) -> int:
        return self._credits_by_epoch.get(self._feed_epoch, 0)

    def _wait_credit(self, *, consume: bool) -> None:
        """Block until a feed-epoch credit is available (or done/stopping)."""
        started = time.monotonic()
        with self._cv:
            while (
                self._feed_credits_locked() <= 0
                and not self._feed_done_locked()
                and not self._stop.is_set()
            ):
                self._cv.wait(timeout=0.05)
            if consume and self._feed_credits_locked() > 0:
                self._credits_by_epoch[self._feed_epoch] -= 1
        elapsed = time.monotonic() - started
        with self._lock:
            self._training_credit_wait_s += elapsed

    def training_credit_wait_seconds(self) -> float:
        """Return exact actor time blocked waiting for predict credits."""
        if self.role != "actor":
            return 0.0
        with self._lock:
            return float(self._training_credit_wait_s)

    def before_microbatch(self) -> None:
        if self.role != "actor":
            return
        with self._lock:
            mode = self._mode
        if mode != "training":
            return

        if self.granularity == "training":
            if not self._training_gate_armed:
                # Wait until at least one predict finished (or epoch done).
                self._wait_credit(consume=True)
                self._training_gate_armed = True
            # Hold grant closed for the rest of run_training.
            self._can_grant.clear()
            with self._lock:
                self._computing = True
            return

        # microbatch: consume one predict-done credit unless epoch finished.
        # While waiting with zero credits the grant stays open so the paired
        # rollout can start the predict that will produce the next credits.
        self._wait_credit(consume=True)
        self._can_grant.clear()
        with self._lock:
            self._computing = True

    def after_microbatch(self) -> None:
        if self.role != "actor":
            return
        with self._lock:
            self._computing = False
            mode = self._mode
        if mode != "training":
            self._can_grant.set()
            return
        if self.granularity == "training":
            # Keep grants closed until end_training_granularity_release.
            return
        # microbatch: open the grant only once all feed-epoch credits are
        # drained so the next predict never overlaps credited microbatches.
        with self._cv:
            drained = self._feed_credits_locked() <= 0
        if drained:
            self._run_pre_yield_hook()
            self._can_grant.set()

    def end_training_granularity_release(self) -> None:
        if self.role != "actor" or self.granularity != "training":
            return
        with self._lock:
            self._computing = False
        self._run_pre_yield_hook()
        self._can_grant.set()

    # ------------------------------------------------------------------
    # Rollout API
    # ------------------------------------------------------------------

    def begin_epoch(self, epoch: int) -> None:
        """Rollout-side: declare the epoch index tagged onto gate messages."""
        if self.role != "rollout":
            return
        self._rollout_epoch = int(epoch)

    def wait_predict_ok(self) -> None:
        if self.role != "rollout":
            return
        self._send(MSG_PREDICT_REQ, self._rollout_epoch)
        code, _ = self._recv()
        if code != MSG_PREDICT_OK:
            raise RuntimeError(
                f"GpuPhaseGate rollout expected PREDICT_OK, got {code}"
            )

    def signal_train_ok(self) -> None:
        if self.role != "rollout":
            return
        self._send(MSG_TRAIN_OK, self._rollout_epoch)

    def signal_epoch_predicts_done(self) -> None:
        if self.role != "rollout":
            return
        self._send(MSG_EPOCH_DONE, self._rollout_epoch)
        # Auto-advance when the runner does not call begin_epoch explicitly.
        self._rollout_epoch += 1


class NullGpuPhaseGate:
    role = "none"
    granularity = "none"

    def set_pre_yield_hook(self, hook) -> None:
        return

    def start_actor_service(self) -> None:
        return

    def stop_actor_service(self) -> None:
        return

    def enter_training(self) -> None:
        return

    def set_training_step(self, step: int) -> None:
        return

    def begin_epoch(self, epoch: int) -> None:
        return

    def mark_no_more_predicts(self) -> None:
        return

    def exit_training(self) -> None:
        return

    def training_credit_wait_seconds(self) -> float:
        return 0.0

    def before_microbatch(self) -> None:
        return

    def after_microbatch(self) -> None:
        return

    def end_training_granularity_release(self) -> None:
        return

    def wait_predict_ok(self) -> None:
        return

    def signal_train_ok(self) -> None:
        return

    def signal_epoch_predicts_done(self) -> None:
        return


def resolve_credits_per_predict(cfg) -> int:
    """Micro-batch credits per finished predict.

    One predict serves every env on the paired rollout GPU, producing
    ``envs_per_actor = total_num_envs / num_actors`` train samples, i.e.
    ``envs_per_actor / actor.micro_batch_size`` micro-batches worth of data.
    Floor (min 1) so the actor never outpaces sample production.
    """
    try:
        total_envs = int(cfg.env.train.total_num_envs)
        num_actors = len(parse_placement(cfg.cluster).actor_ranks)
        mbs = int(cfg.actor.micro_batch_size)
    except Exception:
        logger.exception(
            "GpuPhaseGate could not derive credits_per_predict; using 1"
        )
        return 1
    return max(1, total_envs // max(1, num_actors) // max(1, mbs))


def build_gpu_phase_gate(
    cfg,
    *,
    role: Literal["rollout", "actor"],
    group,
    my_rank: int,
    peer_rank: int,
) -> GpuPhaseGate | NullGpuPhaseGate:
    mode = resolve_gpu_phase_gate(cfg)
    if mode == "none" or group is None:
        return NullGpuPhaseGate()
    gran = resolve_phase_gate_granularity(cfg)
    credits = resolve_credits_per_predict(cfg)
    logger.info(
        "GpuPhaseGate build role=%s gran=%s credits_per_predict=%d",
        role, gran, credits,
    )
    return GpuPhaseGate(
        role=role,
        group=group,
        my_rank=my_rank,
        peer_rank=peer_rank,
        granularity=gran,
        credits_per_predict=credits,
    )
