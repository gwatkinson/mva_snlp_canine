import click
import torch
import torch._dynamo

from mva_snlp_canine.nli.dataset import process_dataset
from mva_snlp_canine.nli.training import finetune_model
from mva_snlp_canine.utils import load_config_nli

torch._dynamo.config.verbose = True


@click.command()
@click.argument("config_file_path", type=str)
def main(config_file_path):
    cfg = load_config_nli(config_file_path)

    # Process dataset
    dataset = process_dataset(
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

    # Training
    for model_name_or_path, postfix in zip(cfg.MODEL_LIST, cfg.MODEL_POSTFIX):
        print(f"\n==== Training for {model_name_or_path} =====\n")

        experiment_output_dir = cfg.DIR_TEMPLATE_TRAINING.format(postfix=postfix)
        experiment_hub_path = cfg.HUB_TEMPLATE_TRAINING.format(postfix=postfix)

        finetune_model(
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_is_tokenized=False,
            num_labels=cfg.NUM_LABELS,
            training_kwargs=cfg.TRAINING_KWARGS,
            output_dir=experiment_output_dir,
            save_local=cfg.SAVE_LOCAL,
            hub_path=experiment_hub_path,
            push_to_hub=cfg.PUSH_TO_HUB,
            token=cfg.TOKEN,
            dataset_name=cfg.HUB_PATH_PREPROCESSED_DATASET,
        )


if __name__ == "__main__":
    main()
