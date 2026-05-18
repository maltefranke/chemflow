"""Lightning datamodule variant that uses the unified scaffold wrapper.

Subclass of :class:`LightningDataModule` that swaps the default
:class:`FlowMatchingDatasetWrapper` for
:class:`FlowMatchingDatasetWrapperScaffold` with a configurable ``mode``:

- ``"molecule_optimization"`` (default) — full-molecule source from the
  same-scaffold pool. The model learns substituent-level edits while the
  shared scaffold is preserved.
- ``"scaffold_decoration"`` — bare-scaffold source (just the scaffold heavy
  atoms of the target). The model learns to grow the full molecule by
  inserting decoration atoms.

The scaffold cache lives at ``scaffold_groups_dir/scaffold_groups_<split>.pt``
and is built lazily on first run.
"""

from __future__ import annotations

import hydra

from chemflow.dataset.lightning_datamodule import LightningDataModule
from chemflow.dataset.scaffold_wrappers import FlowMatchingDatasetWrapperScaffold


_VALID_MODES = ("molecule_optimization", "scaffold_decoration")


class ScaffoldLightningDataModule(LightningDataModule):
    """Datamodule that builds the unified scaffold wrapper around base datasets.

    Args:
        scaffold_groups_dir: Where to read/write the per-split scaffold cache.
        wrapper_kind: ``"molecule_optimization"`` or ``"scaffold_decoration"``;
            forwarded as ``mode`` to :class:`FlowMatchingDatasetWrapperScaffold`.
        assignment_method: Only used by ``molecule_optimization`` —
            ``"mcs_constrained"`` (default) or ``"substituent"`` for
            branch-aware OT on the decoration atoms.
        compute_qed_delta: Only meaningful for ``molecule_optimization``;
            silently disabled in ``scaffold_decoration`` (bare scaffolds have
            no QED).
    """

    def __init__(
        self,
        *args,
        scaffold_groups_dir: str,
        wrapper_kind: str = "molecule_optimization",
        assignment_method: str = "mcs_constrained",
        compute_qed_delta: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if wrapper_kind not in _VALID_MODES:
            raise ValueError(
                f"wrapper_kind must be one of {list(_VALID_MODES)}, "
                f"got {wrapper_kind!r}."
            )
        self._wrapper_kind = wrapper_kind
        self._scaffold_groups_dir = scaffold_groups_dir
        self._assignment_method = assignment_method
        self._compute_qed_delta = compute_qed_delta

    def _wrapper_kwargs(self, stage: str) -> dict:
        return dict(
            mode=self._wrapper_kind,
            distributions=self.distributions,
            interpolator=self.interpolator,
            n_atoms_strategy=self.n_atoms_strategy,
            time_dist=self.time_dist,
            stage=stage,
            rotate=self.rotate,
            n_augmentations=self.n_augmentations if stage == "train" else 1,
            scaffold_groups_dir=self._scaffold_groups_dir,
            assignment_method=self._assignment_method,
            compute_qed_delta=self._compute_qed_delta,
        )

    def setup(self, stage=None):
        if self.vocab is None:
            raise ValueError(
                "vocab and distributions must be set before calling setup()."
            )

        train_cfg = self.datasets.train.copy()
        base_train = hydra.utils.instantiate(
            train_cfg,
            vocab=self.vocab,
            distributions=self.distributions,
            split="train",
        )
        self.train_dataset = FlowMatchingDatasetWrapperScaffold(
            base_dataset=base_train, **self._wrapper_kwargs("train")
        )

        self.val_datasets = []
        for dataset_cfg in self.datasets.val:
            val_cfg = dataset_cfg.copy()
            base_val = hydra.utils.instantiate(
                val_cfg,
                vocab=self.vocab,
                distributions=self.distributions,
                split="val",
            )
            self.val_datasets.append(
                FlowMatchingDatasetWrapperScaffold(
                    base_dataset=base_val, **self._wrapper_kwargs("val")
                )
            )

        self.test_datasets = []
        for dataset_cfg in self.datasets.test:
            test_cfg = dataset_cfg.copy()
            base_test = hydra.utils.instantiate(
                test_cfg,
                vocab=self.vocab,
                distributions=self.distributions,
                split="test",
            )
            self.test_datasets.append(
                FlowMatchingDatasetWrapperScaffold(
                    base_dataset=base_test, **self._wrapper_kwargs("test")
                )
            )
