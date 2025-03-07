import sys
import traceback
import warnings

import matplotlib.pylab as plt
import numpy as np
import pandas as pd


class Vector(list):
    def __getattr__(self, key):
        return Vector([getattr(instance, key) for instance in self])
    
    def __call__(self, *args, **kwargs):
        return [instance(*args, **kwargs) for instance in self]

    def __getstate__(self):
        # Return the state that needs to be pickled
        return self.__dict__.copy(), list(self)

    def __setstate__(self, state):
        # Restore the state from the pickled state
        self.__dict__, list_contents = state
        self.extend(list_contents)


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


# warnings.showwarning = warn_with_traceback


def plot_feature_importance(features, importance):
    """
    Plot feature importances of the RF model.
    """
    import seaborn as sns

    importance = pd.DataFrame(dict(features=features, importance=importance))
    importance = importance.sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, np.ceil(len(importance) / 3)))
    sns.barplot(
        data=importance,
        x="importance",
        y="features",
        ax=ax,
    )
    ax.set_title(label="Feature Importances", fontsize=14)
    ax.set_xlabel(xlabel="Mean decrease in impurity (MDI)", fontsize=12)
    ax.set_ylabel(ylabel="Features", fontsize=12)
    return fig
