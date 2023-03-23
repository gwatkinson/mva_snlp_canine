import click

from mva_snlp_canine.nli.dataset import process_dataset
from mva_snlp_canine.nli.defaults import (
    N_JOBS,
    NO_PBAR,
    NUM_TEST_SAMPLES,
    NUM_TRAIN_SAMPLES,
    NUM_VAL_SAMPLES,
    PREPROCESSING_HUB_PATH,
    PREPROCESSING_PATH,
    SEED,
    TEST_LANGUAGES_SUBSET,
    TEST_PROBS,
    TOKEN,
    TRAIN_LANGUAGES_SUBSET,
    TRAIN_PROBS,
)


@click.command()
@click.argument("experiment_name", type=str)
@click.option(
    "--num_train_samples",
    "-nt",
    default=NUM_TRAIN_SAMPLES,
    type=int,
    help="Number of training samples.",
    show_default=True,
)
@click.option(
    "--num_val_samples",
    "-nv",
    default=NUM_VAL_SAMPLES,
    type=int,
    help="Number of validation samples.",
    show_default=True,
)
@click.option(
    "--num_test_samples",
    "-ns",
    default=NUM_TEST_SAMPLES,
    type=int,
    help="Number of test samples.",
    show_default=True,
)
@click.option(
    "--train_language_subset",
    "-tls",
    default=TRAIN_LANGUAGES_SUBSET,
    type=list,
    help="Languages to keep in the train and validation set.",
    show_default=True,
)
@click.option(
    "--train_probs",
    "-tlp",
    default=TRAIN_PROBS,
    type=list,
    help="Probabilities of the languages to keep in the train and validation set.",
    show_default=True,
)
@click.option(
    "--test_language_subset",
    "-sls",
    default=TEST_LANGUAGES_SUBSET,
    type=list,
    help="Languages to keep in the test set.",
    show_default=True,
)
@click.option(
    "--test_probs",
    "-slp",
    default=TEST_PROBS,
    type=list,
    help="Probabilities of the languages to keep in the test set.",
    show_default=True,
)
@click.option(
    "--save_path",
    "-o",
    default=PREPROCESSING_PATH,
    type=str,
    help="Path to save the dataset.",
    show_default=True,
)
@click.option(
    "--hub_path",
    "-h",
    default=PREPROCESSING_HUB_PATH,
    type=str,
    help="Path to the dataset on the HuggingFace Hub.",
    show_default=True,
)
@click.option(
    "--seed",
    default=SEED,
    type=int,
    help="Seed for the random number generator.",
    show_default=True,
)
@click.option(
    "--n_jobs",
    "-j",
    default=N_JOBS,
    type=int,
    help="Number of jobs to run in parallel.",
    show_default=True,
)
@click.option(
    "--no_pbar",
    default=NO_PBAR,
    type=bool,
    help="Disable the progress bar.",
    show_default=True,
)
@click.option(
    "--token",
    default=TOKEN,
    type=str,
    help="Path to cached token.",
    show_default=False,
)
def main(
    experiment_name,
    num_train_samples,
    num_val_samples,
    num_test_samples,
    train_language_subset,
    train_probs,
    test_language_subset,
    test_probs,
    save_path,
    hub_path,
    seed,
    n_jobs,
    no_pbar,
    token,
):
    experiment_path = save_path.format(experiment_name=experiment_name)
    experiment_hub_path = hub_path.format(experiment_name=experiment_name)

    process_dataset(
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        num_test_samples=num_test_samples,
        train_language_subset=train_language_subset,
        train_probs=train_probs,
        test_language_subset=test_language_subset,
        test_probs=test_probs,
        save_path=experiment_path,
        hub_path=experiment_hub_path,
        seed=seed,
        n_jobs=n_jobs,
        no_pbar=no_pbar,
        token=token,
    )


if __name__ == "__main__":
    main()
