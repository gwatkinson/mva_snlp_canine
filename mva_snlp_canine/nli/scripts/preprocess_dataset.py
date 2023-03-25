import click

from mva_snlp_canine.nli.dataset import process_dataset
from mva_snlp_canine.utils import load_config_nli


@click.command()
@click.argument("config_file_path", type=str)
def main(config_file_path):
    cfg = load_config_nli(config_file_path)

    process_dataset(
        num_train_samples=cfg.NUM_TRAIN_SAMPLES,
        num_val_samples=cfg.NUM_VAL_SAMPLES,
        num_test_samples=cfg.NUM_TEST_SAMPLES,
        train_language_subset=cfg.TRAIN_LANGUAGES_SUBSET,
        train_probs=cfg.TRAIN_PROBS,
        test_language_subset=cfg.TEST_LANGUAGES_SUBSET,
        test_probs=cfg.TEST_PROBS,
        save_path=cfg.DIR_PATH_PREPROCESSED_DATASET,
        hub_path=cfg.HUB_PATH_PREPROCESSED_DATASET,
        seed=cfg.SEED,
        n_jobs=cfg.N_JOBS,
        no_pbar=cfg.NO_PBAR,
        save_local=cfg.SAVE_LOCAL,
        push_to_hub=cfg.PUSH_TO_HUB,
        token=cfg.TOKEN,
    )


if __name__ == "__main__":
    main()
