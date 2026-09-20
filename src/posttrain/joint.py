"""Orchestration policy for the joint DMD+RL loop (Stage 6, DMDR route).

The GPU training loop itself is assembled during the framework bring-up; this
module holds the *policy* pieces that can be specified and tested now:

- TwoTimescale: DMD2's update-ratio schedule between the fake-score critic
  and the generator (paper default: 5 fake updates per generator update).
- PromotionGate: the cold-start -> joint transition. The gate is reward
  reliability, NOT distillation convergence — waiting for convergence would
  collapse the design into the sequential pipeline DMDR refuted.
- LambdaController: rule-based adjustment of the RL weight lambda_rl. The
  hacking signature is test-reward plateau + reference-KID regression; the
  response is to lower lambda, never to raise it automatically.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TwoTimescale:
    """Fake-score/generator update ratio (DMD2 recommends 5:1)."""

    fake_per_generator: int = 5

    def __post_init__(self):
        if self.fake_per_generator < 1:
            raise ValueError("fake_per_generator must be >= 1")

    def generator_updates(self, fake_steps: int) -> int:
        """Generator updates owed after `fake_steps` critic updates."""
        return fake_steps // self.fake_per_generator


@dataclass
class PromotionGate:
    """Decide when cold-start distillation may enter the joint RL phase.

    Reward models only give meaningful rankings once few-step rollouts are
    coherent. The gate promotes after `patience` consecutive evaluations in
    which at least `coherence_threshold` of the probed rollouts were judged
    coherent (by the VLM judge's parse+quality check on a fixed probe set).
    """

    coherence_threshold: float = 0.8
    patience: int = 3
    _streak: int = 0
    _promoted: bool = False
    _history: List[float] = field(default_factory=list)

    def observe(self, coherent_fraction: float) -> bool:
        """Record one probe; return True exactly once, on promotion."""
        if not 0.0 <= coherent_fraction <= 1.0:
            raise ValueError("coherent_fraction must be in [0, 1]")
        self._history.append(coherent_fraction)
        if self._promoted:
            return False
        if coherent_fraction >= self.coherence_threshold:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.patience:
            self._promoted = True
            return True
        return False

    @property
    def promoted(self) -> bool:
        """One-way latch: promotion outlives later failing probes."""
        return self._promoted


@dataclass
class LambdaController:
    """Rule-based controller for the RL weight lambda_rl.

    lambda only ever decreases automatically: reward hacking is asymmetric
    damage, and raising lambda on a plateau risks pushing into it. A manual
    increase remains an operator decision outside this rule.
    """

    value: float = 1.0
    factor: float = 0.5
    floor: float = 0.05
    window: int = 4
    plateau_tol: float = 1e-3
    kid_regress_tol: float = 0.05
    _rewards: List[float] = field(default_factory=list)
    _kids: List[float] = field(default_factory=list)

    def observe(self, test_reward: float, kid: float) -> float:
        """Record one evaluation; return the (possibly lowered) lambda."""
        self._rewards.append(test_reward)
        self._kids.append(kid)
        if len(self._rewards) < self.window + 1:
            return self.value
        recent = self._rewards[-(self.window + 1):]
        plateau = max(recent) - min(recent) <= self.plateau_tol
        kid_regressed = self._kids[-1] > self._kids[-2] + self.kid_regress_tol
        if plateau and kid_regressed:
            self.value = max(self.floor, self.value * self.factor)
        return self.value
