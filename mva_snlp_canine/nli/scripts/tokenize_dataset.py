import click

from mva_snlp_canine.nli.dataset import tokenize_dataset
from mva_snlp_canine.utils import load_config_nli, load_dataset_from_config


@click.command()
@click.argument("config_file_path", type=str)
def main(config_file_path):
    cfg = load_config_nli(config_file_path)

    dataset = load_dataset_from_config(cfg, "tokenized")

    model_list = cfg.MODEL_LIST
    model_postfix = cfg.MODEL_POSTFIX
    for model_name_or_path, postfix in zip(model_list, model_postfix):
        print(f"Tokenizing {model_name_or_path}...")

        model_token_save_path = cfg.DIR_PATH_TOKENIZED_DATASET.format(postfix=postfix)
        model_token_hub_save_path = cfg.HUB_PATH_TOKENIZER_DATASET.format(
            postfix=postfix
        )

        tokenize_dataset(
            dataset=dataset,
            model_name_or_path=model_name_or_path,
            save_path=model_token_save_path,
            hub_path=model_token_hub_save_path,
            n_jobs=cfg.N_JOBS,
            no_pbar=cfg.NO_PBAR,
            save_local=cfg.SAVE_LOCAL,
            push_to_hub=cfg.PUSH_TO_HUB,
            token=cfg.TOKEN,
        )


if __name__ == "__main__":
    main()
