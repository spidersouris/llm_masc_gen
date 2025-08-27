import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_m_scores(
    m_score_results_array,
    output_file: str = "",
    m_score_results_array_mgonly: list | None = None,
):
    n_plots = len(m_score_results_array)
    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    scores_data = {}

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
        subplot_titles=[f"<b>{res['dataset']}</b>" for res in m_score_results_array],
        vertical_spacing=0.15,
    )

    template = "plotly_white"
    font_size = 16

    for idx, m_score_results in enumerate(m_score_results_array, 1):
        row = math.ceil(idx / n_cols)
        col = idx - (row - 1) * n_cols

        dataset = m_score_results["dataset"]
        scores_df = pd.DataFrame.from_dict(m_score_results["detailed_scores"])
        scores = scores_df["m_score"]
        scores_data.update({dataset: sorted(scores)})

        fig.add_trace(
            go.Histogram(
                x=scores,
                xbins=dict(start=-1.25, end=1.25, size=0.3),
                autobinx=False,
                name=f"{dataset} (Non-MG)",
                showlegend=False,
                opacity=1,
                marker=dict(color=dataset_colors.get(dataset, "gray")),
                histnorm="percent",
            ),
            row=row,
            col=col,
        )

        # vlines for main
        overall_m_score = m_score_results["overall_m_score"]
        average_m_score = m_score_results["average_m_score"]
        vline_order = [
            (overall_m_score, "#3366cc", "Non-MG Overall"),
        ]  # (average_m_score, "red", "Non-MG Mean")]
        # if average_m_score > overall_m_score:
        #     vline_order.reverse()

        for vline_x, color, label in vline_order:
            fig.add_vline(
                x=vline_x,
                line_dash="dash",
                line_color=color,
                opacity=0.5,
                row=row,  # type: ignore
                col=col,  # type: ignore
            )

        nonmg_rank_dict = {
            res["dataset"]: i + 1
            for i, res in enumerate(
                sorted(
                    m_score_results_array,
                    key=lambda r: r["overall_m_score"],
                    reverse=True,
                )
            )
        }

        mg_rank_dict = {}

        if m_score_results_array_mgonly:
            mgonly_results = m_score_results_array_mgonly[idx - 1]

            if mgonly_results != {}:
                mgonly_scores = pd.DataFrame.from_dict(
                    mgonly_results["detailed_scores"]
                )["m_score"]

                fig.add_trace(
                    go.Histogram(
                        x=mgonly_scores,
                        xbins=dict(start=-1.25, end=1.25, size=0.3),
                        autobinx=False,
                        name=f"{dataset} (MG)",
                        showlegend=False,
                        opacity=1,
                        marker=dict(
                            color=dataset_colors.get(dataset, "gray"),
                            pattern=dict(
                                shape="\\",
                                fillmode="replace",
                                size=10,
                                solidity=0.3,
                                bgcolor=dataset_colors.get(dataset, "gray"),
                                fgcolor="white",
                            ),
                        ),
                        histnorm="percent",
                    ),
                    row=row,
                    col=col,
                )

                # vlines for MG only
                fig.add_vline(
                    x=mgonly_results["overall_m_score"],
                    line_dash="dot",
                    line_color="rgb(217,95,2)",
                    opacity=0.8,
                    row=row,  # type: ignore
                    col=col,  # type: ignore
                )

                mg_rank_dict = {
                    res["dataset"]: i + 1
                    for i, res in enumerate(
                        sorted(
                            (r for r in m_score_results_array_mgonly if r),
                            key=lambda r: r["overall_m_score"],
                            reverse=True,
                        )
                    )
                }

        rank_badge = f"(<b>#{nonmg_rank_dict[dataset]}</b>/8)"

        overall_m_score_annot = (
            f"Non-MG Overall: <b>{overall_m_score:.3f}</b> {rank_badge}"
        )
        # average_m_score_annot = f"Non-MG Mean: <b>{average_m_score:.3f}</b>"

        annots = [
            (overall_m_score_annot, "#3366cc"),
            # (average_m_score_annot, "red"),
        ]

        if m_score_results_array_mgonly:
            mgonly_results = m_score_results_array_mgonly[idx - 1]
            if mgonly_results != {}:
                mg_rank_badge = f"(<b>#{mg_rank_dict[dataset]}</b>/6)"
                annots.append(
                    (
                        f"MG Overall: <b>{mgonly_results['overall_m_score']:.3f}</b> {mg_rank_badge}",
                        "rgb(217,95,2)",
                    )
                )
                # annots.append((f"MG Mean: <b>{mgonly_results['average_m_score']:.3f}</b>", "brown"))

        y_base = 75
        y_step = 6

        for i, (text, color) in enumerate(annots):
            fig.add_annotation(
                text=text,
                xref=f"x{idx}",
                yref=f"y{idx}",
                x=-0.55,
                y=y_base - i * y_step,
                showarrow=False,
                align="center",
                font=dict(
                    family="Arial",
                    size=14.5,
                    color=color,
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
    # fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="red", dash="dash"), name='Non-MG Mean MScore'), row=1, col=1)
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
    # fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="brown", dash="dot"), name='MG Mean MScore'), row=1, col=1)

    fig.update_layout(
        template=template,
        font=dict(size=font_size, family="Arial"),
        height=400 * n_rows,
        width=400 * n_cols,
        showlegend=True,
        legend=dict(
            x=1.0,
            # temp fix for when 1 col only, may need to be changed
            y=1.1 * (n_cols / 0.5),
            xanchor="right",
            yanchor="top",
            orientation="h",
        ),
        margin=dict(t=100, l=50, r=50, b=50),
        paper_bgcolor="white",
        plot_bgcolor="#f2f2f2",
    )

    # axis labels
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_xaxes(title_text="MScore", row=r, col=c)
            if r in [1, 2, 3] and c == 1:
                fig.update_yaxes(title_text="Percentage (%)", row=r, col=c)

    fig.update_xaxes(
        tickmode="array",
        ticksuffix="   ",
        gridcolor="lightgray",
        tickvals=[-1, -0.5, 0, 0.5, 1],
    )
    fig.update_yaxes(
        tickmode="array",
        ticksuffix="   ",
        gridcolor="lightgray",
    )

    if output_file != "":
        fig.write_image(output_file, scale=2, format="svg")


def visualize_classes(
    m_score_results_array,
    datasets,
    output_file: str = "",
    e2: bool = False,
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

        # print(matched_data)

        all_classes = matched_data["merged_classes"].str.split(", ").explode()
        # print("all_classes", all_classes)
        class_counts = all_classes.value_counts().head(5)
        # print("class_counts", class_counts)

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


def visualize_bias_rate(
    m_score_results,
    general_results,
    datasets,
    output_file: str = "",
    m_score_results_array_neut=None,
    general_results_neut=None,
) -> None:
    template = "plotly_white"
    font_size = 16

    masc_gen_percentages = []
    bias_rates = []

    masc_gen_percentages_neutral = []
    bias_rates_neutral = []

    for i, (m_score_result, general_result) in enumerate(
        zip(m_score_results, general_results)
    ):
        dataset_name = m_score_result["dataset"]
        bias_rates.append(m_score_result["bias_rate"] * 100)

        if m_score_results_array_neut is not None:
            m_score_result_neut = m_score_results_array_neut[i]

        if general_results_neut is not None:
            general_result_neut = general_results_neut[i]

        total_texts = len(general_result)
        total_texts_neut = len(general_result_neut)

        masc_gen_count = sum(
            1 for result in general_result if result.get("real_masc_gen_logs")
        )
        masc_gen_percentage = (
            (masc_gen_count / total_texts) * 100 if total_texts > 0 else 0
        )
        masc_gen_percentages.append(masc_gen_percentage)
        print("EXP 1", masc_gen_percentage, dataset_name)

        if m_score_result_neut != {}:
            bias_rates_neutral.append(m_score_result_neut["bias_rate"] * 100)
            masc_gen_count_neut = sum(
                1 for result in general_result_neut if result.get("real_masc_gen_logs")
            )
            masc_gen_percentage_neut = (
                (masc_gen_count_neut / total_texts_neut) * 100
                if total_texts_neut > 0
                else 0
            )
            print("EXP 2", masc_gen_percentage_neut, dataset_name)
            masc_gen_percentages_neutral.append(masc_gen_percentage_neut)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=datasets,
            y=bias_rates,
            name="MG ≥ 1 w/ human nouns (E1)     ",
            marker=dict(color="#bcbddc"),
            text=[f"{x:.2f}%   " for x in bias_rates],
            textangle=270,
            textposition="auto",
            opacity=0.8,
            offsetgroup="original",
            legendgroup="original",
        )
    )

    fig.add_trace(
        go.Bar(
            x=datasets,
            y=masc_gen_percentages,
            name="MG ≥ 1 overall (E1)     ",
            marker=dict(color="#756bb1"),
            text=[f"  {x:.2f}%   " for x in masc_gen_percentages],
            textangle=270,
            textposition="auto",
            opacity=0.8,
            offsetgroup="original",
            legendgroup="original",
        )
    )

    if m_score_results_array_neut is not None:
        dataset_to_bias_neutral = dict(zip(datasets, bias_rates_neutral))
        dataset_to_masc_gen_neutral = dict(zip(datasets, masc_gen_percentages_neutral))

        print(dataset_to_bias_neutral)
        print(dataset_to_masc_gen_neutral)

        aligned_bias_neutral = [
            dataset_to_bias_neutral.get(ds, None) for ds in datasets
        ]
        aligned_masc_gen_neutral = [
            dataset_to_masc_gen_neutral.get(ds, None) for ds in datasets
        ]

        print(aligned_bias_neutral)
        print(aligned_masc_gen_neutral)

        fig.add_trace(
            go.Bar(
                x=datasets,
                y=aligned_bias_neutral,
                name="MG ≥ 1 w/ human nouns (E2)      ",
                marker=dict(color="#e6939a"),
                text=None,
                opacity=0.8,
                offsetgroup="neutral",
                legendgroup="neutral",
            )
        )

        fig.add_trace(
            go.Bar(
                x=datasets,
                y=aligned_masc_gen_neutral,
                name="MG ≥ 1 overall (E2)",
                marker=dict(color="#d6616b"),
                text=[
                    f"   {x:.2f}% " if x is not None else ""
                    for x in aligned_masc_gen_neutral
                ],
                textposition="auto",
                insidetextanchor="start",
                textangle=270,
                textfont=dict(color="white"),
                opacity=0.8,
                offsetgroup="neutral",
                legendgroup="neutral",
            )
        )

        fig.add_trace(
            go.Bar(
                x=datasets,
                y=aligned_bias_neutral,
                name="",
                marker=dict(color="rgba(0,0,0,0)"),
                text=[
                    f"   {x:.2f}% " if x is not None else ""
                    for x in aligned_bias_neutral
                ],
                textposition="auto",
                insidetextanchor="end",
                textangle=270,
                textfont=dict(color="white"),
                hoverinfo="skip",
                showlegend=False,
                offsetgroup="neutral",
                legendgroup="neutral",
            )
        )

    fig.update_layout(
        barmode="group",
        xaxis=dict(
            tickangle=-45,
            categoryorder="array",
            # categoryarray=combined_datasets,
            tickfont=dict(size=font_size),
        ),
        title=dict(
            text="Masculine Generics (MG) Use Rate In Responses",
            x=0.5,
            y=0.95,
            font=dict(size=16),
        ),
        xaxis_title="Model",
        yaxis_title="Percentage (%)",
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


def visualize_mg_count(
    dfs,
    total_df: pd.DataFrame,
    model_specific: bool = False,
    rangee: list[int] | None = None,
    z_score: float = 1.7,
    output_file: str = "",
    e2: bool = False,
):
    template = "plotly_white"
    font_size = 14

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

    nouns = total_df["noun"].tolist()
    ranks = total_df["rank"].tolist()

    # get individual noun count in each df in dfs
    for df in dfs:
        df_nouns = df["noun"].tolist()
        df_counts = df["count"].tolist()
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
        labels={"Count": "Count", "NounWithZScore": "Noun"}
        if model_specific
        else {"Count": "Count", "Noun": "Noun"},
        template=template,
        color_discrete_map=dataset_colors,
    )

    fig.update_traces(
        textposition="outside",
        texttemplate="<b>%{text}</b>",
    )

    # ?
    # for trace in fig.data:
    #     dataset_name = trace.name
    #     trace.textfont = dict(size=font_size, color=dataset_colors.get(dataset_name, "black"))

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

    if output_file != "":
        fig.write_image(output_file, scale=1.8, format="svg")


def visualize_marker_types(results, output_file: str = "", e2: bool = False):
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

    if e2:
        fp_idx = fp_idx_e2

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

    title_exp_label = "E2" if e2 else "E1"

    fig = px.bar(
        df,
        x="Metric",
        y="Percentage",
        color="Model",
        barmode="group",
        text="Percentage",
        title=f"Inclusive Language Markers Across Models' Responses ({title_exp_label})",
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
    max_range = 25 if e2 else 20
    fig.update_layout(
        title_x=0.5,
        xaxis_tickangle=-45,
        xaxis={"categoryorder": "total descending"},
        yaxis=dict(range=[0, max_range], ticksuffix="%"),
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
