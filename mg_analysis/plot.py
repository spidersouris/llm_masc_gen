import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import pandas as pd
import numpy as np


def visualize_m_scores(
    m_score_results_array,
    output_file: str = "",
) -> None:
    n_plots = len(m_score_results_array)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    dataset_colors = {
        dataset: f"rgba({r}, {g}, {b}, 0.8)"
        for dataset, (r, g, b) in zip(
            [
                "oracle_assistant",
                "oasst2_assistant",
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
                (0, 97, 218),
                (255, 141, 51),
                (255, 68, 51),
            ],
        )
    }

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f'{res["dataset"]}' for res in m_score_results_array],
    )

    template = "plotly_white"
    font_size = 14

    for idx, m_score_results in enumerate(m_score_results_array, 1):
        row = math.ceil(idx / n_cols)
        col = idx - (row - 1) * n_cols

        if m_score_results["dataset"] == "":
            continue

        scores_df = pd.DataFrame.from_dict(m_score_results["detailed_scores"])
        scores = scores_df["m_score"]

        fig.add_trace(
            go.Histogram(
                x=scores,
                xbins=dict(
                    start=0.0,
                    end=1.01,
                    size=0.1,
                ),
                autobinx=False,
                name="Distribution",
                showlegend=False,
                opacity=0.7,
                # histnorm="percent",
                marker=dict(color=dataset_colors[m_score_results["dataset"]]),
            ),
            row=row,
            col=col,
        )

        overall_m_score = m_score_results["overall_m_score"]
        average_m_score = m_score_results["average_m_score"]

        if overall_m_score > average_m_score:
            # Add the "Overall M Score" vline first
            fig.add_vline(
                x=overall_m_score,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"{overall_m_score:.3f}",
                annotation_position="top right",
                annotation=dict(
                    yanchor="top",
                    yshift=5,
                    xshift=5,
                ),
                annotation_font=dict(
                    family="sans serif",
                    color="blue",
                    weight=700,
                ),
                row=row,
                col=col,
            )
            # Add the "Mean M Score" vline second
            fig.add_vline(
                x=average_m_score,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{average_m_score:.3f}",
                annotation_position="top right",
                annotation=dict(
                    yanchor="top",
                    yshift=-10,
                    xshift=-45,
                ),
                annotation_font=dict(
                    family="sans serif",
                    color="red",
                    weight=700,
                ),
                row=row,
                col=col,
            )
        else:
            # Add the "Mean M Score" vline first
            fig.add_vline(
                x=average_m_score,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{average_m_score:.3f}",
                annotation_position="top right",
                annotation=dict(
                    yanchor="top",
                    yshift=5,
                    xshift=5,
                ),
                annotation_font=dict(
                    family="sans serif",
                    color="red",
                    weight=700,
                ),
                row=row,
                col=col,
            )
            # Add the "Overall M Score" vline second
            fig.add_vline(
                x=overall_m_score,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"{overall_m_score:.3f}",
                annotation_position="top right",
                annotation=dict(
                    yanchor="top",
                    yshift=-10,
                    xshift=-45,
                ),
                annotation_font=dict(
                    family="sans serif",
                    color="blue",
                    weight=700,
                ),
                row=row,
                col=col,
            )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="blue", dash="dash"),
            name="Overall M Score",
            showlegend=True,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Mean M Score",
            showlegend=True,
        ),
        row=row,
        col=col,
    )

    fig.update_layout(
        template=template,
        font=dict(size=font_size, family="Arial"),
        height=400 * n_rows,
        width=400 * n_cols,
        showlegend=True,
        legend=dict(
            # orientation="h",
            x=1.0,
            y=1.1,
            xanchor="right",
            yanchor="top",
        ),
        margin=dict(t=100, l=50, r=50, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # update axis settings to show labels on condition
    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            if row in [1, 2, 3] and col == 1:
                fig.update_yaxes(
                    title_text="Frequency",
                    gridcolor="lightgray",
                    showticklabels=True,
                    row=row,
                    col=col,
                )
            # if row in [1, 2, 3] and col == 2:
            fig.update_xaxes(
                title_text="M Score",
                gridcolor="lightgray",
                showticklabels=True,
                row=row,
                col=col,
            )

    fig.update_xaxes(
        tickmode="array",
        ticksuffix="   ",
    )

    fig.update_yaxes(
        ticksuffix="   ",
        tickmode="array",
    )

    if output_file != "":
        fig.write_image(output_file, scale=2, format="svg")


def visualize_classes(
    m_score_results_array,
    datasets,
    output_file: str = "",
) -> None:
    df = pd.read_pickle("dfs/masc_gen_df.pkl")

    dataset_colors = {
        dataset: f"rgba({r}, {g}, {b}, 0.8)"
        for dataset, (r, g, b) in zip(
            [
                "oracle_assistant",
                "oasst2_assistant",
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
                (0, 97, 218),
                (255, 141, 51),
                (255, 68, 51),
            ],
        )
    }

    symbol_map = {
        "Human": "circle",
        "LLM": "square",
    }

    human_datasets = ["oracle_assistant", "oasst2_assistant"]
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

        print(matched_data)

        all_classes = matched_data["merged_classes"].str.split(", ").explode()
        print("all_classes", all_classes)
        class_counts = all_classes.value_counts().head(5)
        print("class_counts", class_counts)

        class_data[dataset] = class_counts

    # top 5 classes across all datasets
    top_classes = (
        pd.concat(class_data.values())
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    class_to_index = {cls: i for i, cls in enumerate(top_classes)}

    scatter_data = []
    for dataset, class_counts in class_data.items():
        for cls in top_classes:
            group = "Human" if dataset in human_datasets else "LLM"

            jitter = (
                np.random.uniform(-0.3, 0)
                if group == "Human"
                else np.random.uniform(0, 0.3)
            )

            scatter_data.append(
                {
                    "Class": cls,
                    "Jittered Class": class_to_index[cls] + jitter,
                    "Frequency": class_counts.get(
                        cls, 0
                    ),  # use 0 if class is not in the dataset
                    "Dataset": f"{dataset} ({group})",
                    "Group": group,
                }
            )

    fig = px.scatter(
        pd.DataFrame(scatter_data),
        x="Jittered Class",
        y="Frequency",
        color="Dataset",
        # symbol="Group",
        symbol_map=symbol_map,
        # color_discrete_map=dataset_colors,
        # size="Frequency",
        hover_data=["Dataset"],
        template="plotly_white",
        opacity=1,
    )
    # .update_traces(mode="lines")

    for dataset in human_datasets:
        fig.for_each_trace(
            lambda trace: (
                trace.update(
                    marker_symbol="circle", marker_color=dataset_colors[dataset]
                )
                if dataset in trace.name
                else None
            )
        )
    for dataset in llm_datasets:
        fig.for_each_trace(
            lambda trace: (
                trace.update(
                    marker_symbol="square", marker_color=dataset_colors[dataset]
                )
                if dataset in trace.name
                else None
            )
        )

    fig.update_layout(
        title="Top 5 MG Human Noun Classes Across Responses",
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


def visualize_bias_rate(
    m_score_results, general_results, datasets, output_file: str = ""
) -> None:
    template = "plotly_white"
    font_size = 14

    masc_gen_percentages = []
    bias_rates = []

    for m_score_result, general_result in zip(m_score_results, general_results):
        bias_rates.append(m_score_result["bias_rate"] * 100)

        total_texts = len(general_result)

        masc_gen_count = sum(
            1 for result in general_result if result.get("masc_gen_logs")
        )
        masc_gen_percentage = (
            (masc_gen_count / total_texts) * 100 if total_texts > 0 else 0
        )
        masc_gen_percentages.append(masc_gen_percentage)

    fig = go.Figure()

    sorted_indices = np.argsort(masc_gen_percentages)[::-1]
    sorted_datasets = [datasets[i] for i in sorted_indices]

    fig.add_trace(
        go.Bar(
            x=datasets,
            y=bias_rates,
            name="% of responses where MG >= 1          <br>across responses with human nouns          ",
            marker=dict(color="#bcbddc"),
            text=[f"{x:.2f}%" for x in bias_rates],
            textposition="auto",
            opacity=0.8,
        )
    )

    fig.add_trace(
        go.Bar(
            x=datasets,
            y=masc_gen_percentages,
            name="% of responses where MG >= 1          <br>across all responses          ",
            marker=dict(color="#756bb1"),
            text=[f"{x:.2f}%" for x in masc_gen_percentages],
            textposition="auto",
            opacity=0.8,
        )
    )

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(
            tickangle=-45,
            categoryorder="array",
            categoryarray=sorted_datasets,
            tickfont=dict(size=font_size),
        ),
        title=dict(
            text="Masculine Generics (MG) Use Rate", x=0.5, y=0.95, font=dict(size=16)
        ),
        xaxis_title="Model",
        yaxis_title="Percentage",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        height=600,
        width=800,
        template=template,
        font=dict(size=font_size),
        legend=dict(
            font=dict(size=14),
            orientation="h",
            yanchor="bottom",
            y=0.99,
            xanchor="center",
            x=0.5,
        ),
    )

    if output_file != "":
        fig.write_image(output_file, scale=1.8, format="svg")


def visualize_marker_types(results, output_file: str = ""):
    template = "plotly_white"

    # false positives
    fp_idx = [
        "llama_266",
        "llama_3610",
        "claude-3-haiku_6120",
        "claude-3-haiku_7216",
        "mistral-small_1429",
        "mistral-small_3132",
        # neutral_prons
        "claude-3-haiku_1862",
        "gpt4o_mini_3936",
        "mistral-small_959",
    ]

    dataset_colors = {
        dataset: f"rgba({r}, {g}, {b}, 0.8)"
        for dataset, (r, g, b) in zip(
            [
                "gemini",
                "gpt4o_mini",
                "claude-3-haiku",
                "llama",
                "ministral",
                "mistral-small",
            ],
            [
                (78, 170, 153),
                (8, 8, 8),
                (204, 120, 92),
                (0, 97, 218),
                (255, 141, 51),
                (255, 68, 51),
            ],
        )
    }

    all_data = []

    for i, dataset_result in enumerate(results):
        dataset_name = dataset_result[i]["dataset"]
        dataset_results = dataset_result

        metrics = {
            "incl_greetings": set(),
            "neutral_prons": set(),
            "incl_pairs": set(),
            "neutral_words": set(),
            "fem_endings": set(),
            # "masc_gen": set(),
        }

        total_texts = len(dataset_results)

        for result in dataset_results:
            if not result or result.get("text_index_dataset") in fp_idx:
                continue

            if result.get("incl_greetings_logs"):
                metrics["incl_greetings"].add(result["text_index"])
            if result.get("neutral_prons_logs"):
                metrics["neutral_prons"].add(result["text_index"])
            if result.get("incl_pairs_logs"):
                metrics["incl_pairs"].add(result["text_index"])
            if result.get("neutral_logs"):
                metrics["neutral_words"].add(result["text_index"])
            if result.get("separator_logs"):
                metrics["fem_endings"].add(result["text_index"])
            if result.get("upper_logs"):
                metrics["fem_endings"].add(result["text_index"])
            # if result.get('masc_gen_logs'):
            #     metrics['masc_gen'].add(result['text_index'])

        for marker_type, text_indices in metrics.items():
            count = len(text_indices)
            percentage = len(text_indices) / total_texts * 100
            all_data.append(
                {
                    "Metric": marker_type,
                    "Percentage": percentage,
                    "Count": count,
                    "Model": dataset_name,
                }
            )

    df = pd.DataFrame(all_data)

    fig = px.bar(
        df,
        x="Metric",
        y="Percentage",
        color="Model",
        barmode="group",
        text="Percentage",
        title="Inclusive Language Markers Across Models' Responses",
        labels={"Percentage": "Percentage of Responses (%)", "Metric": "Marker Type"},
        template=template,
        color_discrete_map=dataset_colors,
    )

    fig.update_traces(
        texttemplate="<b>%{text:.1f}%</b><br>(%{customdata})", textposition="outside"
    )

    fig.for_each_trace(
        lambda trace: trace.update(
            customdata=df[df["Model"] == trace.name]["Count"],
        )
    )
    fig.for_each_trace(lambda t: t.update(textfont_color=t.marker.color))
    fig.update_layout(
        title_x=0.5,
        xaxis_tickangle=-45,
        xaxis={"categoryorder": "total descending"},
        yaxis=dict(range=[0, 20], ticksuffix="%"),
        height=600,
        width=1000,
        font=dict(size=14, family="Arial"),
        legend=dict(
            font=dict(size=13, family="Arial"),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
        bargap=0.1,
    )

    if output_file != "":
        fig.write_image(output_file, scale=1.8, format="svg")
