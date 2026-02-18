import ast
import math
import os
import subprocess

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATASET_COLORS = {
    dataset: f"rgba({r}, {g}, {b}, 0.8)"
    for dataset, (r, g, b) in zip(
        [
            "oracle",
            "oasst2",
            "gemini",
            "gpt4o_mini",
            "claude-3-haiku",
            "llama",
            "ministral",
            "mistral-small",
        ],
        [
            (3, 44, 110),
            (171, 99, 250),
            (78, 170, 153),
            (8, 8, 8),
            (204, 120, 92),
            (73, 136, 196),
            (255, 141, 51),
            (255, 68, 51),
        ],
    )
}


def save_to_pdf(input_file: str) -> None:
    """
    Converts a Plotly SVG file to PDF
    using Typst's svg2pdf (https://github.com/typst/svg2pdf)
    and moves it to the plots/pdfs directory.

    Converting using svg2pdf gives better results than using
    Plotly's built-in PDF export (which uses Kaleido).
    """
    try:
        input_file = os.path.basename(input_file)
        # run svg2pdf in the plots/ directory for the input file
        subprocess.run(["svg2pdf", input_file], check=True, cwd="plots")

        # svg2pdf does not seem to accept an output argument,
        # and instead creates a PDF with the same name as the SVG.
        # So manually move the generated PDF to plots/pdfs/
        os.makedirs("plots/pdfs", exist_ok=True)
        src = os.path.join("plots", f"{input_file[:-4]}.pdf")
        dst = os.path.join("plots", "pdfs", f"{input_file[:-4]}.pdf")
        try:
            os.rename(src, dst)
        # if this fails because dst exists, remove dst and try again
        except Exception:
            os.remove(dst)
            os.rename(src, dst)
    except Exception as e:
        print(f"Error converting SVG to PDF: {e}")


def is_human_dataset(dataset_name: str) -> bool:
    """Check if a dataset is human-generated."""
    human_datasets = {"oasst2", "oracle"}
    return dataset_name.lower() in human_datasets


def emojize(dataset_name: str, symbol_only: bool = False) -> str:
    """Add emoji prefix based on whether dataset is human or LLM-generated."""
    if is_human_dataset(dataset_name):
        return f"🧑 {dataset_name}" if not symbol_only else "🧑"
    else:
        return f"🤖 {dataset_name}" if not symbol_only else "🤖"


def load_ci_map(
    csv_path: str,
    ci_type: str | None = None,
) -> dict[str, dict[str, tuple[float | None, float | None]]]:
    """Load confidence interval map from a CSV file."""
    if ci_type not in {"markers", "bias_rate"}:
        raise ValueError(f"Invalid ci_type: {ci_type}")

    df = pd.read_csv(csv_path)

    def parse_ci(ci_str):
        """
        Parse a confidence interval string into a tuple of floats,
        assuming the string is valid.
        """
        if pd.isna(ci_str) or ci_str == "-":
            return None, None
        lo, hi = ast.literal_eval(ci_str)
        return float(lo), float(hi)

    map = {}
    # GFL markers plot
    if ci_type == "markers":
        # Iterate through rows to build map with both experiments
        # and language marker type (neutral and inclusive)
        for _, row in df.iterrows():
            map[row["dataset"]] = {
                "incl_e1": parse_ci(row.get("ci_incl_e1")),
                "neut_e1": parse_ci(row.get("ci_neut_e1")),
                "incl_e2": parse_ci(row.get("ci_incl_e2")),
                "neut_e2": parse_ci(row.get("ci_neut_e2")),
            }
    # Bias rate plot
    elif ci_type == "bias_rate":
        # Iterate through rows to build map with both experiments
        # and conditions (overall bias and bias considering human nouns only)
        for _, row in df.iterrows():
            map[row["dataset"]] = {
                "e1": parse_ci(row.get("ci_e1")),
                "e2": parse_ci(row.get("ci_e2")),
                "hn_e1": parse_ci(row.get("ci_hn_e1")),
                "hn_e2": parse_ci(row.get("ci_hn_e2")),
            }
    return map


def ci_to_error(
    bar_values: list[float | None],
    datasets: list[str],
    ci_key: str,
    ci_map: dict[str, dict[str, tuple[float | None, float | None]]],
) -> dict:
    """
    Convert confidence intervals to Plotly error bar format.
    """
    err_plus = []
    err_minus = []

    for val, ds in zip(bar_values, datasets):
        if val is None or ds not in ci_map:
            err_plus.append(0)
            err_minus.append(0)
            continue

        lo, hi = ci_map[ds][ci_key]
        if lo is None or hi is None:
            err_plus.append(0)
            err_minus.append(0)
        else:
            err_minus.append(val - lo)
            err_plus.append(hi - val)

    # return in Plotly error bar format
    # https://plotly.com/python/error-bars/
    return dict(
        type="data",
        symmetric=False,
        array=err_plus,
        arrayminus=err_minus,
        thickness=1.8,
        width=6,
    )


def visualize_m_scores(
    m_score_results_array,
    output_file: str = "",
    m_score_results_e2_map: dict | None = None,
    ci: bool = False,
    pdf: bool = False,
    legacy: bool = False,
) -> None:
    """Shelved function for visualizing m_scores."""
    n_plots = len(m_score_results_array)
    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    if legacy:
        bin_start = 0.0
        bin_end = 1.000001  # tiny epsilon to ensure values == 1.0 fall in the last bin
        bin_size = 0.1
        tickvals = [0.0, 0.25, 0.50, 0.75, 1.0]
    else:
        # non-legacy supports [-1,1]
        # bin_start = -1.25
        # bin_end = 1.25
        # bin_size = 0.35
        # tickvals = [-1, -0.5, 0, 0.5, 1]

        bin_start = -2.25
        bin_end = 2.25
        bin_size = 0.35
        tickvals = [-2, -1, 0, 1, 2]

    # sort by overall m_score E1 ascending
    m_score_results_array = sorted(
        m_score_results_array,
        key=lambda r: r["overall_m_score"],
        reverse=False,
    )

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            (
                f"<b>🦰 {res['dataset']}</b>"
                if res["dataset"] in {"oracle", "oasst2"}
                else f"<b>🤖 {res['dataset']}</b>"
            )
            for res in m_score_results_array
        ],
        vertical_spacing=0.15,
    )

    template = "plotly_white"
    font_size = 18

    # load confidence interval data
    ci_map = None
    if ci:
        ci_map = load_ci_map("analyses/mscores.csv")

    for idx, m_score_results in enumerate(m_score_results_array, 1):
        row = math.ceil(idx / n_cols)
        col = idx - (row - 1) * n_cols
        dataset = m_score_results["dataset"]
        scores_df = pd.DataFrame.from_dict(m_score_results["detailed_scores"])
        scores = scores_df["m_score"]
        has_e2_data = m_score_results_e2_map and dataset in m_score_results_e2_map

        ci_lwr_e1 = ci_upr_e1 = None
        ci_lwr_e2 = ci_upr_e2 = None

        if ci_map and dataset in ci_map:
            ci_lwr_e1, ci_upr_e1 = ci_map[dataset]["e1"]
            ci_lwr_e2, ci_upr_e2 = ci_map[dataset]["e2"]

        print(f"{dataset} - Min: {scores.min():.4f}, Max: {scores.max():.4f}")
        print(f"{dataset} - Total values: {len(scores)}")

        # min_oor = -1 if not legacy else 0
        # out_of_range = scores[(scores < min_oor) | (scores > 1)]
        # assert (
        #     not out_of_range.any()
        # ), f"{dataset}: {len(out_of_range)} values outside [{min_oor},1]: {out_of_range.tolist()}"

        eps = 1e-12
        bin_edges = np.arange(bin_start, bin_end + eps, bin_size)
        # arithmetic mean of consecutive edges used for bar x positions
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # debug histogram for E1 scores
        hist_scores, _ = np.histogram(scores, bins=bin_edges)
        print(f"{dataset} - Histogram counts: {hist_scores}")
        assert hist_scores.sum() == len(
            scores
        ), f"[E1] Histogram counts do not sum to total number of scores; {hist_scores.sum()} != {len(scores)}"

        if has_e2_data and m_score_results_e2_map:
            mgonly_results = m_score_results_e2_map[dataset]
            mgonly_scores = pd.DataFrame.from_dict(mgonly_results["detailed_scores"])[
                "m_score"
            ]

            # debug histogram for E2 scores
            hist_mg, _ = np.histogram(mgonly_scores, bins=bin_edges)
            print(f"{dataset} - (e2) Histogram counts: {hist_mg}")
            assert hist_mg.sum() == len(
                mgonly_scores
            ), f"[E2] Histogram counts do not sum to total number of MG-only scores; {hist_mg.sum()} != {len(mgonly_scores)}"

        # manual binning for legacy scores, E1
        counts_e1, _ = np.histogram(scores, bins=bin_edges)
        percent_e1 = counts_e1 / counts_e1.sum() * 100

        fig.add_trace(
            # plot using go.Bar so we can group bars for E1 and E2 side-by-side
            # grouped bars were not appearing correctly with go.Histogram
            # for E2 with legacy score
            # … which is why we have to do all this binning manually
            go.Bar(
                x=bin_centers,
                y=percent_e1,
                name=f"{dataset} (Non-MG)",
                showlegend=False,
                opacity=0.9,
                # offsetgroup set to fix bin overlap and extra gap on the right
                # happening when the first trace was a model with both E1 and E2 data
                # before: https://i.edoyen.com/ShareX/2025/12/HjgaQS.png
                # after: https://i.edoyen.com/ShareX/2025/12/YadAzH.png
                offsetgroup=f"{dataset}-e1",
                marker=dict(color=DATASET_COLORS.get(dataset, "gray")),
            ),
            row=row,
            col=col,
        )

        # vline for overall m_score (E1)
        overall_m_score = m_score_results["overall_m_score"]
        fig.add_vline(
            x=overall_m_score,
            line_dash="dash",
            line_color="#3366cc",
            opacity=0.7,
            row=row,  # type: ignore
            col=col,  # type: ignore
        )

        # add confidence interval shaded region (E1, non-legacy only)
        if ci and ci_map is not None and not legacy:
            fig.add_vrect(
                x0=ci_lwr_e1,
                x1=ci_upr_e1,
                fillcolor="#3366cc",
                opacity=0.18,
                layer="below",
                line_width=0,
                row=row,  # type: ignore
                col=col,  # type: ignore
            )

        # compute ranks for E1 overall m_scores
        nonmg_rank_dict = {
            res["dataset"]: i + 1
            for i, res in enumerate(
                sorted(
                    m_score_results_array,
                    key=lambda r: r["overall_m_score"],
                    reverse=False,
                )
            )
        }

        if has_e2_data and m_score_results_e2_map:
            mgonly_results = m_score_results_e2_map[dataset]
            mgonly_scores = pd.DataFrame.from_dict(mgonly_results["detailed_scores"])[
                "m_score"
            ]

            # manual binning for legacy scores, E2
            counts_e2, _ = np.histogram(mgonly_scores, bins=bin_edges)
            percent_e2 = counts_e2 / counts_e2.sum() * 100
            group_id = f"{dataset}-e2"

            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=percent_e2,
                    name=f"{dataset} (MG)",
                    showlegend=False,
                    opacity=0.9,
                    # width=bin_size * 0.45,
                    offsetgroup=group_id,
                    # alignmentgroup=group_id,
                    marker=dict(
                        color=DATASET_COLORS.get(dataset, "gray"),
                        pattern=dict(
                            shape="\\",
                            fillmode="replace",
                            size=10,
                            solidity=0.3,
                            bgcolor=DATASET_COLORS.get(dataset, "gray"),
                            fgcolor="white",
                        ),
                    ),
                ),
                row=row,
                col=col,
            )

            # vline for overall m_score (E2)
            fig.add_vline(
                x=mgonly_results["overall_m_score"],
                line_dash="dot",
                line_color="rgb(217,95,2)",
                opacity=0.85,
                row=row,  # type: ignore
                col=col,  # type: ignore
            )

            # add confidence interval shaded region (E2, non-legacy only)
            if ci and ci_map is not None and not legacy:
                fig.add_vrect(
                    x0=ci_lwr_e2,
                    x1=ci_upr_e2,
                    fillcolor="rgb(217,95,2)",
                    opacity=0.18,
                    layer="below",
                    line_width=0,
                    row=row,  # type: ignore
                    col=col,  # type: ignore
                )

        # compute ranks for E2 overall m_scores
        mg_rank_dict = {}
        if m_score_results_e2_map:
            mg_rank_dict = {
                ds: i + 1
                for i, ds in enumerate(
                    sorted(
                        m_score_results_e2_map.keys(),
                        key=lambda ds: m_score_results_e2_map[ds]["overall_m_score"],
                        reverse=False,
                    )
                )
            }

        rank_badge = f"(<b>#{nonmg_rank_dict[dataset]}</b>/8)"
        overall_m_score_annot = (
            f"Non-MG Overall:<br><b>{overall_m_score:.3f}</b> {rank_badge}"
        )

        # prepare annotation boxes with background
        # first, add E1 overall score
        annots = [
            (overall_m_score_annot, "#3366cc", "rgba(51, 102, 204, 0.15)"),
        ]

        if has_e2_data and m_score_results_e2_map:
            mgonly_results = m_score_results_e2_map[dataset]
            mg_rank_badge = f"(<b>#{mg_rank_dict[dataset]}</b>/6)"
            # then, add E2 overall score
            annots.append(
                (
                    f"   MG Overall:<br>   <b>{mgonly_results['overall_m_score']:.3f}</b> {mg_rank_badge}",
                    "rgb(217,95,2)",
                    "rgba(217, 95, 2, 0.15)",
                )
            )

        # add background rectangles and text annotations
        # for overall scores
        y_top = 82
        box_height = 16
        box_spacing = 1.2
        x0_frac = 0.11
        x1_frac = 0.53
        x_text_frac = (x0_frac + x1_frac) / 5

        for i, (text, text_color, bg_color) in enumerate(annots):
            y_box_top = y_top - i * (box_height + box_spacing)
            y_box_bottom = y_box_top - box_height
            y_text = y_box_top - box_height / 2

            # add background rectangle
            fig.add_shape(
                type="rect",
                xref="x domain",
                yref=f"y{idx}",
                x0=x0_frac,
                x1=x1_frac,
                y0=y_box_bottom,
                y1=y_box_top,
                fillcolor=bg_color,
                line=dict(color=text_color, width=1.5),
                layer="above",  # z-index higher than background
                row=row,
                col=col,
            )

            # add text annotation
            fig.add_annotation(
                text=text,
                xref="x domain",
                yref=f"y{idx}",
                x=x_text_frac,
                y=y_text,
                showarrow=False,
                align="center",
                font=dict(
                    family="Arial",
                    size=17,
                    color=text_color,
                ),
                row=row,
                col=col,
            )

    # GN generic legend entry (solid fill)
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name="Non-MG Instructions (Exp. 1)",
            marker=dict(color="gray", opacity=1),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # MG generic legend entry (pattern fill)
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name="MG Instructions (Exp. 2)",
            marker=dict(
                color="gray",
                opacity=1,
                pattern=dict(
                    shape="\\",
                    fillmode="replace",
                    size=10,
                    solidity=0.3,
                    fgcolor="#fff",
                    bgcolor="gray",
                ),
            ),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # legend-only dummy traces
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#3366cc", dash="dash"),
            name="Non-MG Overall MScore",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="rgb(217,95,2)", dash="dot"),
            name="MG Overall MScore",
        ),
        row=1,
        col=1,
    )

    if not legacy:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color="#3366cc", opacity=0.18),
                name="Non-MG 95% CI (shaded)",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color="rgb(217,95,2)", opacity=0.18),
                name="MG 95% CI (shaded)",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        template=template,
        font=dict(size=font_size, family="Arial"),
        height=420 * n_rows,
        width=420 * n_cols,
        showlegend=True,
        barmode="group",
        legend=dict(
            x=1.0,
            y=1.1 * (n_cols / 0.5),
            xanchor="right",
            yanchor="top",
            orientation="h",
        ),
        margin=dict(t=100, l=50, r=50, b=50),
        paper_bgcolor="white",
        plot_bgcolor="#f2f2f2",
        bargap=0.1,
        bargroupgap=0.05,
    )

    # axis labels
    title_text = "MScore" if not legacy else "MScore (Legacy)"
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_xaxes(title_text=title_text, row=r, col=c)
            if r in [1, 2, 3] and c == 1:
                fig.update_yaxes(title_text="Percentage (%)", row=r, col=c)

    fig.update_xaxes(
        tickmode="array",
        ticksuffix=" ",
        gridcolor="lightgray",
        tickvals=tickvals,
        range=[bin_start, bin_end],
        constrain="domain",
    )
    fig.update_yaxes(
        tickmode="array",
        ticksuffix=" ",
        gridcolor="lightgray",
    )

    if legacy:
        fig.update_xaxes(
            range=[0, 1.0],
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        )

    if output_file != "":
        output_file = output_file[:-4] + "_legacy.svg" if legacy else output_file
        fig.write_image(output_file, scale=3, format="svg")
        if pdf:
            save_to_pdf(output_file)


def visualize_classes(
    m_score_results_array,
    datasets,
    output_file: str = "",
    e2: bool = False,
    pdf: bool = False,
) -> None:
    """
    Visualize top 5 MG human noun classes across datasets.
    Creates one plot per experiment.
    """
    df = pd.read_pickle("dfs/masc_gen_df.pkl")

    symbol_map = {
        "Human": "circle",
        "LLM": "square",
    }

    human_datasets = ["oracle", "oasst2"]
    llm_datasets = [
        "gemini",
        "gpt4o_mini",
        "claude-3-haiku",
        "llama",
        "ministral",
        "mistral-small",
    ]

    class_data = {}

    for dataset, results in zip(datasets, m_score_results_array):
        masc_gen_nouns = []
        for result in results:
            masc_gen_logs = result.get("real_masc_gen_logs", [])
            masc_gen_nouns.extend([log["masc_gen"] for log in masc_gen_logs])

        matched_data = df[df["noun"].isin(masc_gen_nouns)][["merged_classes"]]
        matched_data = matched_data.replace("", float("NaN")).dropna()

        all_classes = matched_data["merged_classes"].str.split(", ").explode()

        class_counts = all_classes.value_counts().head(5)

        class_data[dataset] = class_counts

    # top 5 classes across all datasets
    top_classes = (
        pd.concat(class_data.values())
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)  # type: ignore
        .head(5)
        .index
    )

    class_to_index = {cls: i for i, cls in enumerate(top_classes)}

    # compute offsets so that each dataset is on its own "column"
    human_offsets = {
        dataset: -0.4 + i * 0.1 for i, dataset in enumerate(human_datasets)
    }

    if e2:
        # give more space to LLMs since they are the only datasets plotted in E2
        llm_offsets = {
            dataset: -0.4 + i * 0.15 for i, dataset in enumerate(llm_datasets)
        }
    else:
        llm_offsets = {
            dataset: -0.15 + i * 0.115 for i, dataset in enumerate(llm_datasets)
        }

    dataset_offsets = {**human_offsets, **llm_offsets}

    scatter_data = []
    for dataset, class_counts in class_data.items():
        for cls in top_classes:
            group = "Human" if is_human_dataset(dataset) else "LLM"

            scatter_data.append(
                {
                    "Class": cls,
                    "Jittered Class": class_to_index[cls] + dataset_offsets[dataset],
                    "Frequency": class_counts.get(
                        cls, 0
                    ),  # use 0 if class is not in the dataset
                    "Dataset": f"{dataset} ({emojize(dataset, symbol_only=True)})",
                    "Group": group,
                }
            )

    fig = px.scatter(
        pd.DataFrame(scatter_data),
        x="Jittered Class",
        y="Frequency",
        color="Dataset",
        symbol_map=symbol_map,
        template="plotly_white",
        opacity=1,
    )

    for dataset in human_datasets:
        fig.for_each_trace(
            lambda trace: (
                trace.update(
                    marker_symbol="circle", marker_color=DATASET_COLORS[dataset]
                )
                if dataset in trace.name
                else None
            )
        )

    for dataset in llm_datasets:
        fig.for_each_trace(
            lambda trace: (
                trace.update(
                    marker_symbol="square", marker_color=DATASET_COLORS[dataset]
                )
                if dataset in trace.name
                else None
            )
        )

    title_exp_label = "E2" if e2 else "E1"

    fig.update_layout(
        title=f"Top 5 MG Human Noun Classes Across Responses ({title_exp_label})",
        title_x=0.5,
        xaxis_title="Class",
        yaxis_title="Frequency (unique nouns)",
        legend_title="Dataset / Model",
        height=600,
        width=1000,
        legend=dict(
            font=dict(size=13, family="Arial"),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.05,
        ),
    )

    fig.update_xaxes(
        tickvals=list(class_to_index.values()),
        ticktext=list(class_to_index.keys()),
        title="Class",
    )

    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    class_indices = list(class_to_index.values())
    for i in range(len(class_indices) - 1):
        mid_point = (
            class_indices[i] + class_indices[i + 1]
        ) / 2  # Midpoint between categories
        fig.add_vline(x=mid_point, line_width=1.5, line_dash="dash", line_color="black")

    if output_file != "":
        fig.write_image(output_file, scale=1.8, format="svg")
        if pdf:
            save_to_pdf(output_file)


def visualize_bias_rate(
    m_score_results,
    general_results,
    datasets,
    output_file: str = "",
    e2_data_map: dict | None = None,
    ci: bool = True,
    pdf: bool = False,
) -> None:
    """
    Visualize bias rates (MG use rates) across datasets.
    """
    template = "plotly_white"
    font_size = 20

    masc_gen_percentages = []
    bias_rates = []
    masc_gen_percentages_neutral = []
    bias_rates_neutral = []

    ci_map = {}
    if ci:
        ci_map = load_ci_map("analyses/bias_rates.csv", ci_type="bias_rate")

    for m_score_result, general_result in zip(m_score_results, general_results):
        dataset_name = m_score_result["dataset"]

        bias_rates.append(m_score_result["bias_rate"] * 100)

        total_texts = len(general_result)
        masc_gen_count = sum(
            1 for result in general_result if result.get("real_masc_gen_logs")
        )
        masc_gen_percentages.append(
            (masc_gen_count / total_texts) * 100 if total_texts > 0 else 0
        )

        # Exp. 2
        if e2_data_map is not None and dataset_name in e2_data_map:
            m_score_result_neut, general_result_neut = e2_data_map[dataset_name]

            bias_rates_neutral.append(m_score_result_neut["bias_rate"] * 100)

            total_texts_neut = len(general_result_neut)
            masc_gen_count_neut = sum(
                1 for result in general_result_neut if result.get("real_masc_gen_logs")
            )
            masc_gen_percentages_neutral.append(
                (masc_gen_count_neut / total_texts_neut) * 100
                if total_texts_neut > 0
                else 0
            )
        else:
            # if dataset is not part of Exp. 2, append None
            # to have equal length lists
            bias_rates_neutral.append(None)
            masc_gen_percentages_neutral.append(None)

    if e2_data_map is not None:
        dataset_bias_pairs = [
            (i, ds, bias_rates_neutral[i]) for i, ds in enumerate(datasets)
        ]
        # sort by bias_rates_neutral (index 2) descending, with None values at the end
        dataset_bias_pairs.sort(
            key=lambda x: (x[2] is None, -(x[2] if x[2] is not None else 0))
        )
        sorted_indices = [i for i, _, _ in dataset_bias_pairs]

        datasets = [datasets[i] for i in sorted_indices]
        bias_rates = [bias_rates[i] for i in sorted_indices]
        masc_gen_percentages = [masc_gen_percentages[i] for i in sorted_indices]
        bias_rates_neutral = [bias_rates_neutral[i] for i in sorted_indices]
        masc_gen_percentages_neutral = [
            masc_gen_percentages_neutral[i] for i in sorted_indices
        ]

    datasets_with_emoji = [emojize(ds) for ds in datasets]

    e2_excluded = {"oasst2", "oracle"}
    e2_indices = [i for i, ds in enumerate(datasets) if ds not in e2_excluded]

    e2_datasets = [datasets_with_emoji[i] for i in e2_indices]
    e2_bias = [bias_rates_neutral[i] for i in e2_indices]
    e2_masc = [masc_gen_percentages_neutral[i] for i in e2_indices]
    e2_raw_names = [datasets[i] for i in e2_indices]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "<b>Non-MG Instructions (Exp. 1)</b>",
            "<b>MG Instructions (Exp. 2)</b>",
        ),
        horizontal_spacing=0.2,
    )

    # Exp. 1
    fig.add_trace(
        go.Bar(
            y=datasets_with_emoji,
            x=masc_gen_percentages,
            orientation="h",
            name="MG ≥ 1 overall",
            marker=dict(
                color="#756bb1",
                pattern=dict(shape="/", fgcolor="#4C4C4C"),
            ),
            text=[f"{x:.2f}%" for x in masc_gen_percentages],
            textposition="inside",
            insidetextanchor="start",
            error_x=(
                ci_to_error(masc_gen_percentages, datasets, "hn_e1", ci_map)
                if ci
                else None
            ),
            legendgroup="E1",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            y=datasets_with_emoji,
            x=bias_rates,
            orientation="h",
            # add spacing so that both legends match subplots position
            name="MG ≥ 1 with human nouns                                   ",
            marker=dict(color="#bcbddc"),
            text=[f"{x:.2f}%" for x in bias_rates],
            textposition="inside",
            insidetextanchor="start",
            error_x=ci_to_error(bias_rates, datasets, "e1", ci_map) if ci else None,
            legendgroup="E1",
        ),
        row=1,
        col=1,
    )

    # Exp. 2
    if e2_data_map is not None:
        fig.add_trace(
            go.Bar(
                y=e2_datasets,
                x=e2_masc,
                orientation="h",
                name="MG ≥ 1 overall",
                marker=dict(
                    color="#d6616b",
                    pattern=dict(shape="/", fgcolor="#4C4C4C"),
                ),
                text=[f"{x:.2f}%" for x in e2_masc],
                textposition="inside",
                insidetextanchor="start",
                textfont=dict(color="white"),
                error_x=(
                    ci_to_error(e2_masc, e2_raw_names, "hn_e2", ci_map) if ci else None
                ),
                legendgroup="E2",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                y=e2_datasets,
                x=e2_bias,
                orientation="h",
                name="MG ≥ 1 with human nouns",
                marker=dict(color="#e6939a"),
                text=[f"{x:.2f}%" for x in e2_bias],
                textposition="inside",
                insidetextanchor="start",
                error_x=(
                    ci_to_error(e2_bias, e2_raw_names, "e2", ci_map) if ci else None
                ),
                legendgroup="E2",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Masculine Generics (MG) Use Rate In Responses</b>",
            x=0.5,
            y=0.98,
            font=dict(size=font_size),
        ),
        height=600,
        width=1200,
        template=template,
        font=dict(size=font_size - 1),
        margin=dict(pad=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        uniformtext_minsize=16,
        uniformtext_mode="show",
    )

    # lower subplot titles
    fig.for_each_annotation(lambda a: a.update(y=a.y - 0.015))

    fig.update_xaxes(
        title="Percentage (%)",
        range=[0, 100],
        ticksuffix="%",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title="Percentage (%)",
        range=[0, 100],
        ticksuffix="%",
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title="Dataset / Model",
        tickfont=dict(size=font_size),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title=None,
        tickfont=dict(size=font_size),
        row=1,
        col=2,
    )

    if output_file:
        fig.write_image(output_file, scale=1.8, format="svg")
    if pdf:
        save_to_pdf(output_file)


def visualize_mg_count(
    dfs,  # from mscore/get_mg_count()
    total_df: pd.DataFrame,
    model_specific: bool = False,
    rangee: list[int] | None = None,
    z_score: float = 1.7,
    output_file: str = "",
    e2: bool = False,
    pdf: bool = False,
    save_df: bool = True,
    filter_epicenes: bool = False,
):
    """
    Visualize MG counts across datasets with optional outlier detection.
    """
    template = "plotly_white"
    font_size = 14

    all_data = []

    nouns = total_df["noun"].tolist()
    ranks = total_df["rank"].tolist()

    # get individual noun count in each df in dfs
    # Note that some nouns may be epicene nouns; this is normal!
    # This is because they were used as MG (e.g. with a masculine determiner)
    for df in dfs:
        df_nouns = df["noun"].tolist()
        df_counts = df["count"].tolist()
        # given that each DF is dataset-specific,
        # we can get the dataset name from any row
        dataset = df["dataset"].tolist()[0]

        for noun, count in zip(df_nouns, df_counts):
            all_data.append(
                {
                    "Noun": "<b>"
                    + str(ranks[nouns.index(noun)])
                    + '</b>. "'
                    + noun
                    + '"<br> ('
                    + str(total_df[total_df["noun"] == noun]["count"].tolist()[0])
                    + " total)",
                    "Count": count,
                    "Rank": ranks[nouns.index(noun)],
                    "Dataset": dataset,
                    "RawNoun": noun,
                }
            )

    df = pd.DataFrame(all_data)
    if save_df:
        df_copy = df.copy()
        df_copy.pop("Noun")
        df_copy = df_copy[["RawNoun", "Dataset", "Count", "Rank"]]
        df_copy = (
            df_copy.groupby("RawNoun")
            .agg({"Count": "sum"})
            .reset_index()
            .sort_values(by="Count", ascending=False)
        )
        csv_name = "mg_count_detailed_e2.csv" if e2 else "mg_count_detailed_e1.csv"
        csv_name = (
            csv_name.replace(".csv", "_no_epicenes.csv")
            if filter_epicenes
            else csv_name
        )
        df_copy.to_csv(f"analyses/{csv_name}", index=False)

    if rangee:
        df = df[(df["Rank"] >= rangee[0]) & (df["Rank"] <= rangee[1])]

    if model_specific:
        # get average counts
        noun_avg_counts = df.groupby("Noun")["Count"].mean().reset_index()
        noun_avg_counts.rename(columns={"Count": "AvgCount"}, inplace=True)

        # get std for each noun count
        noun_std_counts = df.groupby("Noun")["Count"].std().reset_index()
        noun_std_counts.rename(columns={"Count": "StdCount"}, inplace=True)

        # add avg/std to main DF
        df = df.merge(noun_avg_counts, on="Noun")
        df = df.merge(noun_std_counts, on="Noun")

        # get Z-score to identify outliers
        # Z-score = (count - average) / std
        df["Z_Score"] = (df["Count"] - df["AvgCount"]) / df["StdCount"].replace(
            0, 1
        )  # avoid division by zero

        # print(df.sort_values(by="Z_Score", ascending=False).head(15))

        # outliers = z_score > 1.7
        df["IsOutlier"] = df["Z_Score"] > z_score

        # keep nouns with at least one outlier
        outlier_nouns = df.loc[df["IsOutlier"], "Noun"].unique()
        df = df[df["Noun"].isin(outlier_nouns)]

        plot_title = (
            f"Outlier MG Count (Z-Score > {z_score}) (E2)"
            if e2
            else f"Outlier MG Count (Z-Score > {z_score}) (E1)"
        )

        zscore_dict = {}
        for _, row in df.iterrows():
            noun = row["RawNoun"]
            dataset = row["Dataset"]
            zscore = row["Z_Score"]
            if noun not in zscore_dict:
                zscore_dict[noun] = {}
            zscore_dict[noun][dataset] = zscore

        # include z-scores for each dataset
        # custom x-axis tick labels with z-scores
        noun_zscores = {}
        for noun in df["RawNoun"].unique():
            datasets = df[df["RawNoun"] == noun]["Dataset"].unique()
            dataset_with_highest_zscore = max(
                datasets,
                key=lambda d: df[(df["RawNoun"] == noun) & (df["Dataset"] == d)][
                    "Z_Score"
                ].values[0],
            )

            # create zinfo only for dataset with highest score
            z_info = "".join(
                [
                    f"[{d}: z={df[(df['RawNoun'] == noun) & (df['Dataset'] == d)]['Z_Score'].values[0]:.2f}]"
                    for d in datasets
                    if d == dataset_with_highest_zscore
                ]
            )

            # z_info = "<br>".join([f"{d}: z={df[(df['RawNoun']==noun) & (df['Dataset']==d)]['Z_Score'].values[0]:.2f}"
            orig_label = df[df["RawNoun"] == noun]["Noun"].iloc[0]
            # combine original label with z-scores
            noun_zscores[noun] = (
                f"{orig_label}<br><span style='font-size:{font_size - 2}px'>{z_info}</span>"
            )

        # mapping for x-axis
        df["NounWithZScore"] = df["RawNoun"].map(noun_zscores)
    else:
        plot_title = "MG Count"

    nouns_sorted = df["Noun"].unique()

    fig = px.bar(
        df,
        x="NounWithZScore" if model_specific else "Noun",
        y="Count",
        color="Dataset",
        barmode="group",
        text="Count",
        title=plot_title,
        labels=(
            {"Count": "Count", "NounWithZScore": "Noun"}
            if model_specific
            else {"Count": "Count", "Noun": "Noun"}
        ),
        template=template,
        color_discrete_map=DATASET_COLORS,
    )

    fig.update_traces(
        textposition="outside",
        texttemplate="<b>%{text}</b>",
    )

    fig.for_each_trace(
        lambda trace: trace.update(
            textfont_color=DATASET_COLORS.get(trace.name, "black")
        )
    )

    fig.update_layout(
        title_x=0.5,
        xaxis_tickangle=-45,
        xaxis={"categoryorder": "total descending"},
        # yaxis=dict(range=[0, 20], ticksuffix="%"),
        height=700,
        width=1200,
        font=dict(size=font_size, family="Arial"),
        legend=dict(
            font=dict(size=font_size - 1, family="Arial"),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
        bargap=0.1,
    )

    fig.update_layout(
        shapes=[
            dict(
                type="line",
                x0=i - 0.5,
                y0=0,
                x1=i - 0.5,
                y1=df["Count"].max(),
                line=dict(
                    color="black",
                    width=1,
                    dash="dash",
                ),
            )
            for i in range(1, len(nouns_sorted))
        ]
    )

    fig.update_legends(title_text="Model")

    if output_file != "":
        fig.write_image(output_file, scale=1.8, format="svg")
        if pdf:
            save_to_pdf(output_file)


def visualize_marker_types(
    results_e1,
    results_e2,
    output_file: str = "",
    ci: bool = True,
    pdf: bool = False,
):
    """
    Visualize marker types (neutral vs inclusive) across experiments.
    """
    template = "plotly_white"

    ci_map = {}
    if ci:
        ci_map = load_ci_map("analyses/marker_rates.csv", ci_type="markers")

    def _results_to_df(results, e2=False):
        """
        Convert results to DataFrame for plotting.
        """
        # false positives
        fp_idx_e1 = [
            "llama_266",
            "llama_3610",
            "claude-3-haiku_6120",
            "claude-3-haiku_7216",
            "mistral-small_1429",
            "mistral-small_3132",
            "claude-3-haiku_1862",
            "gpt4o_mini_3936",
            "mistral-small_959",
        ]

        fp_idx_e2 = [
            "llama_431",
            "llama_432",
            "llama_788",
            "llama_1065",
            "llama_1329",
            "llama_1564",
            "llama_2007",
            "llama_2034",
            "llama_2417",
            "llama_3148",
            "llama_3352",
            "llama_3925",
            "llama_4020",
            "llama_4037",
            "llama_4117",
            "llama_4261",
            "ministral_887",
            "ministral_1328",
            "claude-3-haiku_1329",
            "gpt4o_mini_1329",
            "gpt4o_mini_3947",
            "gemini_1329",
            "mistral-small_4368",
        ]

        fp_idx = fp_idx_e2 if e2 else fp_idx_e1

        all_data = []

        for dataset_results in results:
            dataset_name = dataset_results[0]["dataset"]
            total_texts = len(dataset_results)

            metrics = {
                "Neutral Markers": set(),
                "Inclusive Markers": set(),
            }

            for result in dataset_results:
                if not result or result.get("text_index_dataset") in fp_idx:
                    continue

                if (
                    result.get("incl_greetings_logs")
                    or result.get("neutral_prons_logs")
                    or result.get("incl_pairs_logs")
                    or result.get("separator_logs")
                    or result.get("upper_logs")
                ):
                    metrics["Inclusive Markers"].add(result["text_index"])

                if result.get("neutral_logs"):
                    metrics["Neutral Markers"].add(result["text_index"])

            for marker_type, text_indices in metrics.items():
                count = len(text_indices)
                percentage = count / total_texts * 100 if total_texts else 0
                all_data.append(
                    {
                        "Metric": marker_type,
                        "Percentage": percentage,
                        "Count": count,
                        "Model": dataset_name,
                    }
                )

        return pd.DataFrame(all_data)

    df_e1 = _results_to_df(results_e1, e2=False)
    df_e2 = _results_to_df(results_e2, e2=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=(
            "<b>Non-MG Instructions (Exp. 1)</b>",
            "<b>MG Instructions (Exp. 2)</b>",
        ),
    )

    metrics_order = ["Neutral Markers", "Inclusive Markers"]

    for df, col, show_legend, exp_suffix in [
        (df_e1, 1, True, "e1"),
        (df_e2, 2, False, "e2"),
    ]:
        for model in df["Model"].unique():
            dff = df[df["Model"] == model]

            # Build ordered percentage and count lists matching metrics_order
            percentages = []
            counts = []
            for m in metrics_order:
                row = dff[dff["Metric"] == m]
                if not row.empty:
                    percentages.append(float(row["Percentage"].iloc[0]))
                    counts.append(int(row["Count"].iloc[0]))
                else:
                    percentages.append(0.0)
                    counts.append(0)

            # build combined error_y array (one entry per bar) if CI enabled
            error_y_obj = None
            if ci:
                err_vals = []
                metric_label_map = {
                    "Neutral Markers": f"neut_{exp_suffix}",
                    "Inclusive Markers": f"incl_{exp_suffix}",
                }
                for m, pct in zip(metrics_order, percentages):
                    try:
                        err_dict = ci_to_error(
                            [pct], [model], metric_label_map[m], ci_map
                        )
                    except Exception:
                        err_dict = None

                    # extract numeric error
                    if err_dict and isinstance(err_dict, dict) and "array" in err_dict:
                        # ensure numeric value
                        arr = err_dict["array"]
                        try:
                            numeric = float(arr[0])
                        except Exception:
                            numeric = 0.0
                        err_vals.append(numeric)
                    else:
                        # fallback to zero
                        err_vals.append(0.0)

                # only attach error_y if we have len==2
                if len(err_vals) == len(metrics_order):
                    error_y_obj = dict(type="data", array=err_vals)

            # add a single trace for the model (two bars)
            fig.add_trace(
                go.Bar(
                    x=metrics_order,
                    y=percentages,
                    name=model,
                    text=percentages,
                    customdata=counts,
                    textposition="outside",
                    marker_color=DATASET_COLORS.get(model),
                    showlegend=show_legend,
                    error_y=error_y_obj,
                ),
                row=1,
                col=col,
            )

    fig.update_traces(text=None)
    PADDING = 0.6  # vertical padding above CI

    # collect traces per subplot (column)
    subplot_traces = {1: [], 2: []}

    for trace in fig.data:
        col = 1 if trace.xaxis == "x" else 2
        subplot_traces[col].append(trace)

    for col, traces in subplot_traces.items():
        n_traces = len(traces)

        for t_idx, trace in enumerate(traces):
            for i, (x_cat, y, count) in enumerate(
                zip(trace.x, trace.y, trace.customdata)
            ):
                # base category index
                cat_idx = i  # 0 = neutral, 1 = inclusive

                # compute grouped-bar horizontal offset
                group_width = 0.8
                bar_width = group_width / n_traces
                x_offset = -group_width / 2 + bar_width / 2 + t_idx * bar_width

                x_numeric = cat_idx + x_offset

                # CI height
                ci_height = 0.0
                if trace.error_y and "array" in trace.error_y:
                    ci_height = trace.error_y["array"][i] or 0.0

                fig.add_annotation(
                    x=x_numeric,
                    y=y + ci_height + PADDING,
                    xref="x" if col == 1 else "x2",
                    yref="y",
                    text=f"<b>{y:.1f}%</b><br>({count})",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(
                        color=trace.marker.color,
                        size=11,
                        family="Arial",
                    ),
                )

    # define specific text colors based on marker colors
    fig.for_each_trace(lambda t: t.update(textfont_color=t.marker.color))

    # vertical separator
    fig.add_shape(
        type="line",
        x0=0.5,
        x1=0.5,
        y0=0,
        y1=1,
        xref="paper",
        yref="paper",
        line=dict(
            color="rgba(0,0,0,0.35)",
            width=2,
        ),
    )

    fig.update_layout(
        template=template,
        title="Gender-Fair Language Markers Across Models’ Responses",
        title_x=0.5,
        barmode="group",
        height=500,
        width=1100,
        yaxis=dict(title="Percentage of Responses (%)", range=[0, 25], ticksuffix="%"),
        font=dict(size=16, family="Arial"),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    fig.update_xaxes(tickangle=-45)

    if output_file:
        fig.write_image(output_file, scale=1.8, format="svg")
        if pdf:
            save_to_pdf(output_file)


def create_pareto_plot(
    m_scores,
    m_scores_e2_filtered,
    pareto_front_e1,
    pareto_front_e2,
    output_file: str = "",
    pdf: bool = False,
):
    """
    Create a Pareto front plot for Experiment 1 and Experiment 2.
    """

    # https://stackoverflow.com/a/72389439
    def improve_text_position(n):
        """Alternate text positions between top center
        and bottom center for n annotations."""
        positions = ["top center", "bottom center"]
        return [positions[i % 2] for i in range(n)]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "<b>Non-MG Instructions (Exp. 1)</b>",
            "<b>MG Instructions (Exp. 2)</b>",
        ),
        vertical_spacing=0.15,
        shared_xaxes=False,
    )

    # collect all N values for synchronized color scale
    all_neut_values = []
    all_mg_values = []
    all_incl_values = []

    for m in m_scores + m_scores_e2_filtered:
        vec = m["pareto_vector"]
        all_neut_values.append(vec["NEUT_rate"])
        all_mg_values.append(vec["MG_rate"])
        all_incl_values.append(vec["INCL_rate"])

    neut_min = min(all_neut_values)
    neut_max = max(all_neut_values)

    # Exp. 1
    exp1_regular = {"x": [], "y": [], "z": [], "labels": []}
    exp1_pareto = {"x": [], "y": [], "z": [], "labels": []}

    for m in m_scores:
        vec = m["pareto_vector"]
        dataset = m["dataset"]

        data = {
            "x": vec["MG_rate"],
            "y": vec["INCL_rate"],
            "z": vec["NEUT_rate"],
            "label": dataset,
        }

        if dataset in pareto_front_e1:
            exp1_pareto["x"].append(data["x"])
            exp1_pareto["y"].append(data["y"])
            exp1_pareto["z"].append(data["z"])
            exp1_pareto["labels"].append(data["label"])
        else:
            exp1_regular["x"].append(data["x"])
            exp1_regular["y"].append(data["y"])
            exp1_regular["z"].append(data["z"])
            exp1_regular["labels"].append(data["label"])

    # Add Experiment 1 non-Pareto points
    # Manually extract and hide ministral and claude-3-haiku points
    # to change their position later and fix overlap
    ministral_x, ministral_y = None, None
    claude_x, claude_y = None, None
    if exp1_regular["x"]:
        filtered_labels = []

        for i, label in enumerate(exp1_regular["labels"]):
            if label.lower() == "ministral":
                ministral_x = exp1_regular["x"][i]
                ministral_y = exp1_regular["y"][i]
                filtered_labels.append("")
            elif label.lower() == "claude-3-haiku":
                claude_x = exp1_regular["x"][i]
                claude_y = exp1_regular["y"][i]
                filtered_labels.append("")
            else:
                filtered_labels.append(label)

        fig.add_trace(
            go.Scatter(
                x=exp1_regular["x"],
                y=exp1_regular["y"],
                mode="markers+text",
                marker=dict(
                    symbol="cross-thin",
                    size=20,
                    color=exp1_regular["z"],
                    colorscale="Agsunset",
                    cmin=neut_min,
                    cmax=neut_max,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="N̂ (%) ↑"),
                        x=1.15,
                        len=0.4,
                        y=0.5,
                        yanchor="middle",
                        xanchor="center",
                    ),
                    line=dict(
                        width=2,
                        cmin=neut_min,
                        cmax=neut_max,
                        color=exp1_regular["z"],
                        colorscale="Agsunset",
                    ),
                ),
                text=filtered_labels,
                textposition=improve_text_position(len(filtered_labels)),
                textfont=dict(size=17),
                name="Non-Pareto optimal",
                legendgroup="combined",
            ),
            row=1,
            col=1,
        )

    # Add Experiment 1 Pareto points
    gpt_x, gpt_y = None, None
    if exp1_pareto["x"]:
        filtered_labels = []

        for i, label in enumerate(exp1_pareto["labels"]):
            if label.lower() == "gpt4o_mini":
                gpt_x = exp1_pareto["x"][i]
                gpt_y = exp1_pareto["y"][i]
                filtered_labels.append("")
            else:
                filtered_labels.append(label)

        fig.add_trace(
            go.Scatter(
                x=exp1_pareto["x"],
                y=exp1_pareto["y"],
                mode="markers+text",
                marker=dict(
                    symbol="circle",
                    size=22,
                    color=exp1_pareto["z"],
                    colorscale="Agsunset",
                    cmin=neut_min,
                    cmax=neut_max,
                    showscale=False,
                ),
                text=filtered_labels,
                textposition=improve_text_position(len(filtered_labels)),
                textfont=dict(size=20, color="black", weight="bold"),
                name="Pareto optimal",
                legendgroup="combined",
            ),
            row=1,
            col=1,
        )

        # Connect Pareto front points for Exp 1
        pareto_sorted = sorted(
            zip(exp1_pareto["x"], exp1_pareto["y"]), key=lambda p: p[0]
        )
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pareto_sorted],
                y=[p[1] for p in pareto_sorted],
                mode="lines",
                line=dict(color="rgba(0, 0, 0, 0.7)", width=3, dash="dash"),
                name="Pareto Line",
                legendgroup="combined",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Exp. 2
    exp2_regular = {"x": [], "y": [], "z": [], "labels": []}
    exp2_pareto = {"x": [], "y": [], "z": [], "labels": []}

    for m in m_scores_e2_filtered:
        vec = m["pareto_vector"]
        dataset = m["dataset"]

        data = {
            "x": vec["MG_rate"],
            "y": vec["INCL_rate"],
            "z": vec["NEUT_rate"],
            "label": dataset,
        }

        if dataset in pareto_front_e2:
            exp2_pareto["x"].append(data["x"])
            exp2_pareto["y"].append(data["y"])
            exp2_pareto["z"].append(data["z"])
            exp2_pareto["labels"].append(data["label"])
        else:
            exp2_regular["x"].append(data["x"])
            exp2_regular["y"].append(data["y"])
            exp2_regular["z"].append(data["z"])
            exp2_regular["labels"].append(data["label"])

    # Add Experiment 2 non-Pareto points
    if exp2_regular["x"]:
        fig.add_trace(
            go.Scatter(
                x=exp2_regular["x"],
                y=exp2_regular["y"],
                mode="markers+text",
                marker=dict(
                    symbol="cross-thin",
                    size=20,
                    color=exp2_regular["z"],
                    colorscale="Agsunset",
                    cmin=neut_min,
                    cmax=neut_max,
                    showscale=False,
                    line=dict(
                        width=2,
                        cmin=neut_min,
                        cmax=neut_max,
                        color=exp2_regular["z"],
                        colorscale="Agsunset",
                    ),
                ),
                text=exp2_regular["labels"],
                textposition=improve_text_position(len(exp2_regular["labels"])),
                textfont=dict(size=17),
                name="Non-Pareto optimal",
                legendgroup="combined",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Add Experiment 2 Pareto points
    gemini_x, gemini_y = None, None
    if exp2_pareto["x"]:
        filtered_labels = []

        for i, label in enumerate(exp2_pareto["labels"]):
            if label.lower() == "gemini":
                gemini_x = exp2_pareto["x"][i]
                gemini_y = exp2_pareto["y"][i]
                filtered_labels.append("")
            else:
                filtered_labels.append(label)

        fig.add_trace(
            go.Scatter(
                x=exp2_pareto["x"],
                y=exp2_pareto["y"],
                mode="markers+text",
                marker=dict(
                    symbol="circle",
                    size=20,
                    color=exp2_pareto["z"],
                    colorscale="Agsunset",
                    cmin=neut_min,
                    cmax=neut_max,
                    showscale=False,
                ),
                text=filtered_labels,
                textposition=improve_text_position(len(filtered_labels)),
                textfont=dict(size=20, color="black", weight="bold"),
                name="Pareto optimal",
                legendgroup="combined",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Connect Pareto front points for Exp 2
        pareto_sorted = sorted(
            zip(exp2_pareto["x"], exp2_pareto["y"]), key=lambda p: p[0]
        )
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pareto_sorted],
                y=[p[1] for p in pareto_sorted],
                mode="lines",
                line=dict(color="rgba(0, 0, 0, 0.7)", width=3, dash="dash"),
                name="Pareto Line",
                legendgroup="combined",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    if ministral_x is not None:
        fig.add_annotation(
            x=ministral_x,
            y=ministral_y,
            text="ministral",
            showarrow=False,
            yshift=20,
            font=dict(size=17),
            row=1,
            col=1,
        )

    if claude_x is not None:
        fig.add_annotation(
            x=claude_x,
            y=claude_y,
            text="claude-3-haiku",
            showarrow=False,
            yshift=20,
            xshift=8,
            font=dict(size=15),
            row=1,
            col=1,
        )

    if gpt_x is not None:
        fig.add_annotation(
            x=gpt_x,
            y=gpt_y,
            text="gpt4o_mini",
            showarrow=False,
            yshift=-10,
            xshift=0,
            font=dict(size=17, weight="bold", color="black"),
            row=1,
            col=1,
        )

    if gemini_x is not None:
        fig.add_annotation(
            x=gemini_x,
            y=gemini_y,
            text="gemini",
            showarrow=False,
            yshift=2,
            xshift=-45,
            font=dict(size=20, weight="bold", color="black"),
            row=2,
            col=1,
        )

    fig.for_each_xaxis(lambda a: a.title.update(font=dict(size=22)))

    # Update axes for Experiment 1
    fig.update_xaxes(
        title_text="M̂ (%) ↓",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        range=[50, 70],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Î (%) ↑",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        range=[0, 1],
        row=1,
        col=1,
    )

    # Update axes for Experiment 2
    fig.update_xaxes(
        title_text="M̂ (%) ↓",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        range=[70, 85],
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Î (%) ↑",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        range=[0, 1],
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Pareto Optimization of Models/Datasets<br>for M̂, Î and N̂",
            x=0.42,
            y=0.96,
            xanchor="center",
            font=dict(size=25),
        ),
        width=900,
        height=1000,
        legend=dict(
            x=1.3,
            y=0.98,
            xanchor="right",
            yanchor="top",
            orientation="h",
            font=dict(size=25),
        ),
        hovermode=False,
        plot_bgcolor="white",
        font=dict(size=25),
    )

    if output_file != "":
        fig.write_image(output_file, scale=1.8, format="svg")
        if pdf:
            save_to_pdf(output_file)
