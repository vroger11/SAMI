import os

from os.path import exists, join
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, save, output_file
from bokeh.io import export_svgs


TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save"
TOOLTIPS = [
    ("index", "$index"),
    ("filepath", "@filepath"),
    ("sex", "@sex"),
    ("intel", "@intel"),
    ("severity", "@severity"),
    ("mean_score", "@mean_score"),
    ("std_score", "@std_score"),
    ("min_score", "@min_score"),
    ("max_score", "@max_score"),
    ("median_score", "@median_score"),
]

def plot_results(model_name, source, seg_size, feature_preparation_method, score_version="mean", y_axis='severity', folder_out="results"):
    """Save scatter plot of a given computed score and severity or intelligibility score.

    Parameters:
    -----------
    model_name: str
        Name of the model used.

    source: dict
        Dictionnary containing the data that will be used as a bokeh ColumnDataSource

    seg_size: int
        Number of samples used per segments.

    feature_preparation_method: str or None
        Method used to prepare the features outputed by the model.

    score_version: str, optional
        Can be either "std", "mean", "median", "min" or "max".

    y_axis: str, optional
        Can be either "severity" or "intel".

    folder_out: str, optional
        Folder path where the results will be put.
    """
    title = f" {model_name} {score_version} score using a frame size of {seg_size} samples"

    p = figure(tools=TOOLS, toolbar_location="above", plot_width=1000, tooltips=TOOLTIPS, title=title)
    p.xaxis.axis_label = f"{feature_preparation_method} {model_name} {score_version} score"
    p.yaxis.axis_label = "Severity" if y_axis == 'severity' else "Intelligibility"

    score_version += "_"

    source = ColumnDataSource(source)
    p.circle(f'{score_version}score', y_axis, size=10, color="group", source=source)

    folder_out = join(folder_out, feature_preparation_method)
    folder_out = join(folder_out, y_axis)
    if not exists(folder_out):
        os.makedirs(folder_out)

    output_file(join(folder_out, f"{score_version}{model_name}_{y_axis}_result.html"))
    save(p)
    p.output_backend = "svg"
    export_svgs(p, filename=join(folder_out, f"{score_version}{model_name}_{y_axis}_result.svg"))
