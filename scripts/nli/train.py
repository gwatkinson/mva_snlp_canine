import click

from mva_snlp_canine.nli.training import finetune_model
from mva_snlp_canine.utils import load_config, load_dataset_from_config


@click.command()
@click.argument("config_file_path", type=str)
def main(config_file_path):
    cfg = load_config(config_file_path)
    dataset_is_tokenized = cfg.DATASET_IS_TOKENISED

    model_list = cfg.MODEL_LIST
    model_postfix = cfg.MODEL_POSTFIX

    for model_name_or_path, postfix in zip(model_list, model_postfix):
        print(f"Training for {model_name_or_path}...")

        if cfg.DATASET_IS_TOKENISED:
            dataset = load_dataset_from_config(cfg, "tokenized", postfix=postfix)
        else:
            dataset = load_dataset_from_config(cfg, "preprocessed")

        experiment_output_dir = cfg.DIR_TEMPLATE_TRAINING.format(postfix=postfix)
        experiment_hub_path = cfg.HUB_TEMPLATE_TRAINING.format(postfix=postfix)

        finetune_model(
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_is_tokenized=dataset_is_tokenized,
            num_labels=cfg.NUM_LABELS,
            training_kwargs=cfg.TRAINING_KWARGS,
            output_dir=experiment_output_dir,
            save_local=cfg.SAVE_LOCAL,
            hub_path=experiment_hub_path,
            push_to_hub=cfg.PUSH_TO_HUB,
            token=cfg.TOKEN,
        )


if __name__ == "__main__":
    main()
