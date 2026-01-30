import hydra
import omegaconf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig
from omegaconf import OmegaConf

from chemflow.flow_matching.interpolation import Interpolator
from chemflow.dataset.molecule_data import MoleculeBatch
import torch.nn.functional as F
from chemflow.utils import remove_token_from_distribution

from chemflow.flow_matching.schedules import FastPowerSchedule

# resolvers for more complex config expressions
OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)

torch.set_float32_matmul_precision("medium")

pl.seed_everything(42)


def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # Instantiate preprocessing to compute distributions from training dataset
    hydra.utils.log.info("Instantiating preprocessing...")
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)

    # Extract vocab and distributions from preprocessing
    vocab = preprocessing.vocab
    distributions = preprocessing.distributions

    if cfg.data.cat_strategy != "mask":
        # remove <MASK> token from the atom_type_distribution and edge_type_distribution
        atom_tokens, atom_type_distribution = remove_token_from_distribution(
            vocab.atom_tokens, distributions.atom_type_distribution, "<MASK>"
        )
        edge_tokens, edge_type_distribution = remove_token_from_distribution(
            vocab.edge_tokens, distributions.edge_type_distribution, "<MASK>"
        )
        vocab.atom_tokens = atom_tokens
        distributions.atom_type_distribution = atom_type_distribution
        vocab.edge_tokens = edge_tokens
        distributions.edge_type_distribution = edge_type_distribution

    cfg.data.vocab = vocab
    print(distributions)

    hydra.utils.log.info(
        f"Preprocessing complete.\n"
        f"Found {len(vocab.atom_tokens)} atom tokens: {vocab.atom_tokens}\n"
        f"Found {len(vocab.edge_tokens)} edge tokens: {vocab.edge_tokens}\n"
        f"Found {len(vocab.charge_tokens)} charge tokens: {vocab.charge_tokens}"
    )
    hydra.utils.log.info("Distributions computed from training dataset.")

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule,
        _recursive_=False,
        vocab=vocab,
        distributions=distributions,
    )
    # Call setup to create datasets with tokens and distributions
    datamodule.setup()

    # Get one sample and one target from the test dataloader
    samples_batched, targets_batched = next(iter(datamodule.val_dataloader()[0]))

    # Convert to data lists to extract individual molecules
    samples_list = samples_batched.to_data_list()[:32]
    targets_list = targets_batched.to_data_list()[:32]

    results = []
    targets_single = []

    for sample_single, target_single in zip(samples_list, targets_list):
        # Duplicate the same sample 100 times
        num_duplicates = 100
        samples_duplicated = [sample_single.clone() for _ in range(num_duplicates)]
        targets_duplicated = [target_single.clone() for _ in range(num_duplicates)]

        # Create MoleculeBatch from duplicated lists
        samples_batched_dup = MoleculeBatch.from_data_list(samples_duplicated)
        targets_batched_dup = MoleculeBatch.from_data_list(targets_duplicated)

        # Create interpolator (use the one from datamodule or create a new one)
        interpolator = Interpolator(
            vocab=vocab,
            distributions=distributions,
            ins_noise_scale=0.25,
            ins_schedule=FastPowerSchedule(beta=2.5),
            del_schedule=FastPowerSchedule(beta=2.5),
        )

        # Instantiate integrator to get time distribution (same as used in training/sampling)
        integrator = hydra.utils.instantiate(
            cfg.model.integrator,
            vocab=vocab,
            distributions=distributions,
        )

        # Get time points using the same method as the integrator
        # The integrator's get_time_steps() returns step sizes, but we need time points
        # So we'll compute time points the same way the integrator does internally
        # Use num_duplicates as the number of steps to get exactly that many time points
        device = integrator.device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if integrator.time_strategy == "linear":
            time_points = torch.linspace(0, 1, num_duplicates + 1, device=device)
        elif integrator.time_strategy == "log":
            # Same logic as in integrator.get_time_steps()
            start_log = torch.log10(torch.tensor(0.01, device=device))
            end_log = torch.log10(torch.tensor(1.0, device=device))
            time_points = 1 - torch.logspace(
                start_log, end_log, num_duplicates + 1, device=device
            )
            time_points = torch.flip(time_points, dims=[0])
        else:
            raise ValueError(f"Invalid time strategy: {integrator.time_strategy}")

        # Use the time points for interpolation (excluding the last one which is 1.0)
        # This gives us num_duplicates time points from 0 to just before 1.0
        t = time_points[:-1]

        # Interpolate all 100 samples at their respective times
        # This mirrors the integration process where each sample is integrated at different time steps
        mol_t, mol_1, ins_targets = interpolator.interpolate_different_size(
            samples_batched_dup,
            targets_batched_dup,
            t,
        )

        # Convert MoleculeBatch to trajectory format (list of MoleculeData objects)
        # This matches the format of sample() when return_traj=True
        # The trajectory contains 100 time steps: one MoleculeData for each time 0.00, 0.01, ..., 0.99
        trajectory = mol_t.to_data_list()

        results.append(trajectory)
        targets_single.append(target_single)

    # Save trajectory
    torch.save(results, "results.pt")

    # Save ground truth molecule in a separate file
    torch.save(targets_single, "ground_truth.pt")


@hydra.main(
    config_path="../configs",
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
