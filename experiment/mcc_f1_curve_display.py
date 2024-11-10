class MCCF1CurveDisplay:
    """MCC-F1 Curve visualization with threshold values."""

    def __init__(self, *, f1, mcc, thresholds,
                 mcc_f1=None, estimator_name=None, pos_label=None):
        self.estimator_name = estimator_name
        self.f1 = f1
        self.mcc = mcc
        self.thresholds = thresholds
        self.mcc_f1 = mcc_f1
        self.pos_label = pos_label

    def plot(self, ax=None, *, name=None, show_best_threshold=False, n_thresholds=0, **kwargs):
        """Plot visualization with threshold values

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is created.

        name : str, default=None
            Name of ROC Curve for labeling. If `None`, use the name of the estimator.

        n_thresholds : int, default=5
            Number of threshold values to display on the curve.

        Returns
        -------
        display : MCCF1CurveDisplay
            Object that stores computed values.
        """
        import numpy as np
        name = self.estimator_name if name is None else name

        line_kwargs = {}
        if self.mcc_f1 is not None and name is not None:
            line_kwargs["label"] = f"{name} (MCC-F1 = {self.mcc_f1:.2f})"
        elif self.mcc_f1 is not None:
            line_kwargs["label"] = f"MCC-F1 = {self.mcc_f1:.2f}"
        elif name is not None:
            if show_best_threshold:
                line_kwargs["label"] = f"{name} (Best $\\tau$ = {self.thresholds[np.argmax(self.mcc)]:.3f}, normMCC = {np.max(self.mcc):.3f})"
            else:
                line_kwargs["label"] = name

        line_kwargs.update(**kwargs)

        import matplotlib.pyplot as plt
        from matplotlib.figure import figaspect
        import numpy as np

        if ax is None:
            fig, ax = plt.subplots(figsize=figaspect(1.))

        # Plot the MCC-F1 curve
        self.line_, = ax.plot(self.f1, self.mcc, **line_kwargs)

        # Add threshold values
        if n_thresholds > 0:
            # Get indices for evenly spaced points along the curve
            n_points = len(self.thresholds)
            indices = np.linspace(0, n_points - 1, n_thresholds, dtype=int)

            # Plot threshold points and values
            ax.scatter(self.f1[indices], self.mcc[indices],
                       color='orange', zorder=2, s=20)

            for idx in indices:
                # Add annotation with threshold value
                ax.annotate(f'$\\tau$={self.thresholds[idx]:.2f}',
                            (self.f1[idx], self.mcc[idx]),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        info_pos_label = (f" (Positive label: {self.pos_label})"
                          if self.pos_label is not None else "")

        xlabel = "F1-Score" + info_pos_label
        ylabel = "MCC" + info_pos_label
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))

        if "label" in line_kwargs:
            ax.legend(loc="lower right")

        self.ax_ = ax
        self.figure_ = ax.figure
        return self
