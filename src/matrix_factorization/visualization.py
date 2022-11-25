import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_learning_curve(training_losses, validation_losses, validation_scores, path=None):

    """
    Visualize learning curves of the models

    Parameters
    ----------
    training_losses: array-like of shape (n_epochs or n_steps)
        Array of training losses computed after every epoch or step

    validation_losses: array-like of shape (n_epochs or n_steps)
        Array of validation losses computed after every epoch or step

    validation_scores: array-like of shape (n_epochs or n_steps)
        Array of validation scores computed after every epoch or step

    path: str or None
        Path of the output file (if path is None, plot is displayed with selected backend)
    """

    fig, axes = plt.subplots(figsize=(32, 18), nrows=2, dpi=100)
    sns.lineplot(
        x=np.arange(1, len(training_losses) + 1),
        y=training_losses,
        ax=axes[0],
        label='train_loss'
    )
    if validation_losses is not None:
        sns.lineplot(
            x=np.arange(1, len(validation_losses) + 1),
            y=validation_losses,
            ax=axes[0],
            label='val_loss'
        )
    for metric, scores in validation_scores.items():
        sns.lineplot(
            x=np.arange(1, len(scores) + 1),
            y=scores,
            ax=axes[1],
            label=metric
        )

    for i in range(2):
        axes[i].set_xlabel('Epochs/Steps', size=15, labelpad=12.5)
        axes[i].set_ylabel('Loss/Metrics', size=15, labelpad=12.5)
        axes[i].tick_params(axis='x', labelsize=12.5, pad=10)
        axes[i].tick_params(axis='y', labelsize=12.5, pad=10)
        axes[i].legend(prop={'size': 18})

    axes[0].set_title('Training and Validation Losses', size=20, pad=15)
    axes[1].set_title('Validation Scores', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
