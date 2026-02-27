__version__ = "0.1.0"

from .cleaning import (
    suggest_fill_strategy,
    handle_missing_values,
)
from .data_viz import (
    plot_correlation_heatmap, 
    plot_histograms, 
    set_theme, 
    get_theme_colors, 
    apply_plot_style, 
    list_themes,
    plot_countplots, 
    plot_boxplots,
    plot_scatter,
    _example_usage,
    quick_eda
)
from .model_eval import (
    evaluate_model,
    evaluate_classification,
    evaluate_regression,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_residuals,
    auto_cross_validate,
    plot_correlation
)
from .preprocessing import (
    detect_outliers,
    auto_encode,
    auto_fix_dtypes,
    remove_outliers,
    cap_outliers,
    scale_data,
)

__all__ = [
    "data_cleaning",
    "visualization",
    "evaluate_model",
    "evaluate_classification",
    "evaluate_regression",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_residuals",
    "auto_cross_validate",
    "suggest_fill_strategy",
    "handle_missing_values",
    "plot_correlation",
    "plot_correlation_heatmap", 
    "plot_histograms", 
    "set_theme", 
    "get_theme_colors", 
    "apply_plot_style", 
    "list_themes",
    "plot_countplots", 
    "plot_boxplots",
    "plot_scatter",
    "_example_usage",
    "quick_eda",
    "detect_outliers",
    "auto_encode",
    "auto_fix_dtypes",
    "remove_outliers",
    "cap_outliers",
    "scale_data"
]
