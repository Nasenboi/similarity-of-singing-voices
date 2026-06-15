import matplotlib.pyplot as plt


def plot_model_train_results(
    test_loss,
    train_loss,
    test_accuracy=None,
    train_accuracy=None,
    save_path: str = None,
    model_name: str = "Model",
):
    """Plot the training progress of a model

    Args:
        test_loss (iterable): The test loss
        train_loss (iterable): The train loss
        test_accuracy (iterable, optional): The test accuracy. Defaults to None.
        train_accuracy (iterable, optional): The train accuracy. Defaults to None.
        save_path (str, optional): An optional path to save the plot to. Defaults to None.
        model_name (str, optional): The models name. Defaults to "Model".
    """
    fig, ax1 = plt.subplots()
    ax1.plot(test_loss, label="Test Loss", color="red")
    ax1.plot(train_loss, label="Train Loss", color="orange", linestyle=":")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    if test_accuracy is not None or train_accuracy is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", color="black")
        ax2.tick_params(axis="y", labelcolor="black")

        if test_accuracy is not None:
            ax2.plot(test_accuracy, label="Test Accuracy", color="green")
        if train_accuracy is not None:
            ax2.plot(train_accuracy, label="Train Accuracy", color="blue", linestyle=":")

    plt.title(f"{model_name} Training Progress")
    lines1, labels1 = ax1.get_legend_handles_labels()
    if test_accuracy is not None or train_accuracy is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_scores(
    x,
    y,
    title: str = "Accuracy Scores",
    xlabel: str = "Accuracy (%)",
    ylabel: str = "Features",
    save_path: str = None,
    random_chance: float = None,
    human_baseline: float = None,
):
    """Plots any scores as a bar plot

    Args:
        x (iterable): the scores
        y (iterable): the labels
        title (str, optional): Plot title. Defaults to "Accuracy Scores".
        xlabel (str, optional): X axis label. Defaults to "Accuracy (%)".
        ylabel (str, optional): Y axis label_. Defaults to "Features".
        save_path (str, optional): Optional path to save the model to. Defaults to None.
        random_chance (float, optional): Optional random chance line. Defaults to None.
    """
    x, y = list(x), list(y)
    plt.barh(y=y, width=x)
    for i, v in enumerate(x):
        plt.text(0.01, i, f"{v:.3f}", va="center", ha="left")
    if random_chance is not None:
        plt.axvline(
            x=random_chance,
            linestyle=":",
            color="red",
            alpha=1.0,
            label=f"Random chance = {random_chance:.3f}",
        )

    if human_baseline is not None:
        plt.axvline(
            x=human_baseline,
            linestyle=":",
            color="green",
            alpha=1.0,
            label=f"Human Baseline = {human_baseline:.3f}",
        )

    if random_chance is not None or human_baseline is not None:
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()
