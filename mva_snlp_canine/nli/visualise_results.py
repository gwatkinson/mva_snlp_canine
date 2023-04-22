import click
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

sns.set_theme(
    style="white", context="paper", palette="deep", font="sans-serif", font_scale=1.5
)

language_subset = [
    "en",
    "ar",
    "fr",
    "es",
    "de",
    "el",
    "bg",
    "ru",
    "tr",
    "zh",
    "th",
    "vi",
    "hi",
    "ur",
    "sw",
]

language_to_abbr = {
    "English": "en",
    "Arabic": "ar",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Greek": "el",
    "Bulgarian": "bg",
    "Russian": "ru",
    "Turkish": "tr",
    "Chinese": "zh",
    "Thai": "th",
    "Vietnamese": "vi",
    "Hindi": "hi",
    "Urdu": "ur",
    "Swahili": "sw",
}
abbr_to_language = {v: k for k, v in language_to_abbr.items()}
abbr_to_model = {"bert": "BERT", "canine_s": "Canine-S", "canine_c": "Canine-C"}
model_to_abbr = {v: k for k, v in abbr_to_model.items()}


def load_metric_df(exp_dir, attacked=False):
    path = (
        f"{exp_dir}/results/metrics.csv"
        if not attacked
        else f"{exp_dir}/results/attacked_metrics.csv"
    )
    metric_df = pd.read_csv(path)
    metric_df["language"] = metric_df["language"].map(abbr_to_language)
    metric_df["model"] = metric_df["model"].map(abbr_to_model)
    metric_df.sort_values(by=["language", "model"], inplace=True)
    return metric_df


def visualize_test_metrics(metric_df, num, languages, attacked=False):
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.barplot(metric_df, y="accuracy", x="language", hue="model", ax=ax)
    ax.set(xlabel="Language", ylabel="Accuracy", title=f"({num}k {languages})")
    ax.legend(loc="upper left", title="Model", bbox_to_anchor=(1, 1), ncol=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    title = (
        "Accuracy by Language and Model"
        if not attacked
        else "Accuracy on Attacked Dataset"
    )
    fig.suptitle(title)
    fig.tight_layout()

    return fig


def boxplot_test_metrics(metric_df, num, languages, attacked=False):
    fig = px.box(
        metric_df,
        x="model",
        y="f1",
        points="all",
        hover_data=[
            "language",
            "loss",
            "accuracy",
            "precision",
            "recall",
            "runtime",
            "samples_per_second",
        ],
    )

    title = (
        f"F1 barplot by language ({num}k {languages})"
        if not attacked
        else f"F1 on attacked dataset ({num}k {languages})"
    )
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="F1",
        height=600,
        width=800,
    )
    fig.update_traces(
        hovertemplate="<b>Language</b>: %{customdata[0]}<br>"
        + "<b>F1</b>: %{y:.2f}<br>"
        + "<b>Loss</b>: %{customdata[1]:.2f}<br>"
        + "<b>Accuracy</b>: %{customdata[2]:.2f}<br>"
        + "<b>Precision</b>: %{customdata[3]:.2f}<br>"
        + "<b>Recall</b>: %{customdata[4]:.2f}<br>"
        + "<b>Runtime</b>: %{customdata[5]:.2f}<br>"
        + "<b>Samples per second</b>: %{customdata[6]:.0f}<br>"
        + "<extra></extra>",
    )
    return fig


def visualise_exp(exp_name, num=None, languages=None, attacked=False):
    exp_dir = f"nli_results/{exp_name}"
    num = num or exp_name.split("_")[0][:-1]
    languages = languages or ", ".join(exp_name.split("_")[1:])

    metric_df = load_metric_df(exp_dir, attacked=attacked)

    bar_path = (
        f"{exp_dir}/results/barplot_metrics.pdf"
        if not attacked
        else f"{exp_dir}/results/barplot_metrics_attacked.pdf"
    )
    bar_fig = visualize_test_metrics(metric_df, num, languages, attacked)
    bar_fig.savefig(bar_path)
    plt.close(bar_fig)
    print(f"--- Barplot saved to ./{bar_path} ---")

    box_path = (
        f"{exp_dir}/results/boxplot_metrics.pdf"
        if not attacked
        else f"{exp_dir}/results/boxplot_metrics_attacked.pdf"
    )
    box_path_html = (
        f"{exp_dir}/results/boxplot_metrics.html"
        if not attacked
        else f"{exp_dir}/results/boxplot_metrics_attacked.html"
    )
    box_fig = boxplot_test_metrics(metric_df, num, languages, attacked)
    box_fig.write_image(box_path)
    box_fig.write_html(box_path_html, include_plotlyjs="cdn")
    print(f"--- Boxplot saved to ./{box_path} ---")


# click command to run the script on an experiment
@click.command()
@click.argument("exp_name")
@click.option("--num", default=None, help="Number of samples used in the training set")
@click.option("--languages", default=None, help="Languages used in the training set")
@click.option(
    "--attacked",
    default=False,
    is_flag=True,
    help="Whether to visualise attacked metrics",
)
def main(exp_name, num, languages, attacked):
    visualise_exp(exp_name, num, languages, attacked)


if __name__ == "__main__":
    main()
