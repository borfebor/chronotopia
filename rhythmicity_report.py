import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from io import BytesIO
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st

import plots


class RhythmicityReport:
    """
    Builds a multi-page PDF report for rhythmicity analysis results.

    Usage:
        report = RhythmicityReport(
            df=df,
            t_col=t_col,
            result_df=result_df,   # or None if analysis not run
            layout_df=layout_df,   # or None if no group layout
            phases=phases,         # or None
            method=method,
            thresh=thresh,
            ent=ent,
            ent_days=ent_days,
            ent_color=ent_color,
            bg_color=bg_color,
            unit=unit,
            T=T,
            order=order,
            t0=t0,
            t1=t1,
            conditions=conditions,
            data_cols=data_cols,
            period_len_min=period_len_min,
            period_len_max=period_len_max,
            period_estimation=period_estimation,
            sum_stats=sum_stats,   # or None
            methods=methods,       # the methods module
            file_name=file_name,
        )
        pdf_buffer = report.build().to_pdf()
    """

    def __init__(
        self,
        df,
        t_col,
        result_df=None,
        layout_df=None,
        phases=None,
        sum_stats=None,
        methods=None,
        method=None,
        thresh=0.05,
        ent=0,
        ent_days=0,
        ent_color="blue",
        bg_color="white",
        unit="a.u.",
        T=24,
        order=None,
        t0=None,
        t1=None,
        conditions=None,
        data_cols=None,
        period_len_min=18,
        period_len_max=30,
        period_estimation="",
        file_name="",
    ):
        self.df = df
        self.t_col = t_col
        self.result_df = result_df
        self.layout_df = layout_df
        self.phases = phases
        self.sum_stats = sum_stats
        self.methods = methods
        self.method = method
        self.thresh = thresh
        self.ent = ent
        self.ent_days = ent_days
        self.ent_color = ent_color
        self.bg_color = bg_color
        self.unit = unit
        self.T = T
        self.order = order
        self.t0 = t0
        self.t1 = t1
        self.conditions = conditions or []
        self.data_cols = data_cols or []
        self.period_len_min = period_len_min
        self.period_len_max = period_len_max
        self.period_estimation = period_estimation
        self.file_name = file_name
        self.figures = []

        # ── What this report is actually able to say ─────────────────────────
        # Every section consults these instead of touching result_df directly.
        # Previously the builders assumed an analysis had been run: opening the
        # report before pressing "Run analysis" raised
        # AttributeError: 'NoneType' object has no attribute 'columns'.
        self.has_results = (
            result_df is not None and getattr(result_df, "empty", True) is False
        )
        self.q_col = self._find_q_col()
        # ML columns are only present when Tempo was the selected method. It used
        # to run on every analysis and be appended to whatever else was chosen,
        # so the report always carried a second, unrequested verdict.
        self.has_ml = bool(
            self.has_results and "probability_rhythmic" in self.result_df.columns
        )
        self.is_ml_method = str(method or "").lower().startswith("tempo")
        self.has_periods = bool(
            self.has_results and "Periods" in self.result_df.columns
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _find_q_col(self):
        """The significance column for the selected method, or None."""
        if not getattr(self, "has_results", False) or not self.method:
            return None
        candidates = [c for c in self.result_df.columns if self.method in c]
        q = [c for c in candidates if "BH.Q" in c.upper()]
        return q[0] if q else None

    @staticmethod
    def _fmt_q(q):
        """q-values at sensible precision. 0.9133333333333333 helps nobody."""
        if q is None or not np.isfinite(q):
            return None
        if q < 1e-4:
            return "q < 0.0001"
        if q < 1e-3:
            return f"q = {q:.1e}"
        return f"q = {q:.3f}"

    @staticmethod
    def _fmt_pct(p):
        if p is None or not np.isfinite(p):
            return None
        return f"{p * 100:.0f}%"

    def _sample_facts(self, col):
        """
        (verdict, list of short fact strings) for one sample.

        Returns ("", []) when no analysis has been run, so callers can fall back
        to just the sample name rather than branching on result_df themselves.
        """
        if not self.has_results:
            return "", []

        focus = self.result_df[self.result_df["CycID"] == col]
        if focus.empty:
            return "", []

        facts, verdict = [], ""

        if self.has_periods and np.isfinite(focus["Periods"].mean()):
            facts.append(f"τ {focus['Periods'].mean():.1f} h")

        if self.is_ml_method and self.has_ml:
            pct = self._fmt_pct(focus["probability_rhythmic"].mean())
            conf = focus["confidence"].values[0] if "confidence" in focus else None
            if pct:
                facts.append(f"P {pct}" + (f" ({conf})" if conf else ""))
            if "is_rhythmic" in focus:
                verdict = "rhythmic" if bool(focus["is_rhythmic"].values[0]) else "arrhythmic"
        elif self.q_col is not None:
            q = focus[self.q_col].mean()
            qs = self._fmt_q(q)
            if qs:
                facts.append(qs)
            if np.isfinite(q):
                verdict = "rhythmic" if q <= self.thresh else "arrhythmic"

        return verdict, facts

    def _get_sample_title(self, col):
        """
        Two lines at most: the sample name, then the numbers that matter.

        This used to be five lines carrying the period, the raw q-value, the
        reject flag, the threshold, a separate ML verdict and an unformatted
        probability (0.9133333333333333). In a grid of panels the titles were
        taller than the plots.
        """
        verdict, facts = self._sample_facts(col)
        if not facts and not verdict:
            return col
        tail = " · ".join(facts + ([verdict] if verdict else []))
        return f"{col}\n{tail}"

    def _draw_entrainment(self, ax):
        """
        Zeitgeber shading on one axes.

        Every panel that shows a trace against absolute time gets this. The
        per-sample grids inside `add_group_traces` drew raw traces with no
        shading at all, so a plate under entrainment looked free-running.
        """
        if self.ent_days is None or self.ent_days <= 0:
            return
        xmin = self.df[self.t_col].min()
        xmax = self.df[self.t_col].max()
        plots.plot_entrainment_ax(
            ax, self.df, self.t_col,
            (xmin // 24) * 24, ((xmax // 24) + 1) * 24,
            self.ent_days, order=self.order, T=self.T, color=self.ent_color,
        )

    def _make_grid(self, N, scale_w=4, scale_h=3):
        """Return (fig, flat_axes, cols, rows) for an N-panel grid."""
        cols = math.ceil(math.sqrt(N))
        rows = math.ceil(N / cols)
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * scale_w, rows * scale_h),
            layout="tight",
        )
        flat = np.array(axes).flatten() if np.size(axes) > 1 else np.array([axes])
        return fig, flat, cols, rows

    def _hide_unused(self, fig, flat_axes, N):
        """Delete subplot panels that exceed the data count."""
        for j in range(N, len(flat_axes)):
            fig.delaxes(flat_axes[j])

    def _make_section_page(self, title, subtitle=""):
        """A simple divider page to separate report sections."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.plot([0.1, 0.9], [0.55, 0.55], color="steelblue", linewidth=3,
                transform=ax.transAxes)
        ax.text(0.5, 0.62, title, transform=ax.transAxes,
                fontsize=28, weight="bold", ha="center", va="center")
        if subtitle:
            ax.text(0.5, 0.44, subtitle, transform=ax.transAxes,
                    fontsize=14, ha="center", va="center", color="gray")
        return fig

    def _make_cover_page(self):
        """Title / metadata cover page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        ax.text(0.5, 0.72, "Rhythmicity Analysis Report",
                transform=ax.transAxes, fontsize=32, weight="bold",
                ha="center", va="center")

        if self.file_name:
            ax.text(0.5, 0.60, f"Dataset: {self.file_name}",
                    transform=ax.transAxes, fontsize=14,
                    ha="center", va="center", color="#444444")

        meta_lines = [
            f"Method: {self.method}" if self.method else "",
            f"Significance threshold: {self.thresh}",
            f"Period range: {self.period_len_min}–{self.period_len_max} h",
            f"Entrainment days: {self.ent_days}",
            f"Conditions: {', '.join(self.conditions)}" if self.conditions else "",
        ]
        meta_text = "\n".join(line for line in meta_lines if line)
        ax.text(0.5, 0.42, meta_text, transform=ax.transAxes,
                fontsize=12, ha="center", va="center",
                linespacing=2.0, color="#333333")

        ax.text(0.5, 0.10,
                f"Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M')}",
                transform=ax.transAxes, fontsize=10, ha="center",
                va="center", color="gray")

        ax.plot([0.05, 0.95], [0.82, 0.82], color="steelblue", linewidth=3,
                transform=ax.transAxes)
        ax.plot([0.05, 0.95], [0.18, 0.18], color="steelblue", linewidth=1,
                transform=ax.transAxes)
        return fig

    def _make_summary_table(self):
        """
        One-page overview: rhythmicity rate and median period per condition.
        Only built when both layout_df and result_df are available.
        """
        if self.layout_df is None or not self.has_results:
            return None
        if self.q_col is None and not self.has_ml:
            return None

        summary = []
        for cond in self.layout_df.Condition.unique():
            names = self.layout_df[self.layout_df.Condition == cond]["name"]
            sub = self.result_df[self.result_df["CycID"].isin(names)]
            if sub.empty:
                continue

            if self.is_ml_method and self.has_ml:
                rhythmic = int((sub["probability_rhythmic"] > 0.5).sum())
                label = "Rhythmic (Tempo) (%)"
            else:
                rhythmic = int((sub[self.q_col] <= self.thresh).sum())
                label = f"Rhythmic ({self.method}) (%)"

            row = {
                "Condition": cond,
                "N": len(sub),
                "Rhythmic (n)": rhythmic,
                label: f"{100 * rhythmic / len(sub):.0f}%",
            }
            if self.has_periods:
                row["Median Period (h)"] = f"{sub['Periods'].median():.1f}"
                row["Mean Period (h)"] = f"{sub['Periods'].mean():.1f}"
            summary.append(row)

        if not summary:
            return None

        df_sum = pd.DataFrame(summary)
        n_rows = len(df_sum)
        fig, ax = plt.subplots(figsize=(11, max(3, n_rows * 0.6 + 2)))
        ax.axis("off")

        tbl = ax.table(
            cellText=df_sum.values,
            colLabels=df_sum.columns,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 2.0)

        # Style header row
        for j in range(len(df_sum.columns)):
            tbl[0, j].set_facecolor("steelblue")
            tbl[0, j].set_text_props(color="white", weight="bold")

        ax.set_title("Summary Statistics", weight="bold", fontsize=16, pad=20)
        return fig

    # ------------------------------------------------------------------ #
    #  Section builders                                                    #
    # ------------------------------------------------------------------ #

    def add_phase_plots(self):
        if self.ent_days <= 0 or self.phases is None:
            return self

        n_conditions = (
            self.layout_df.Condition.unique()
            if self.layout_df is not None
            else self.data_cols
        )
        N = len(n_conditions)
        fig, flat_axes, _, _ = self._make_grid(N)

        # Recreate with polar projection (can't use _make_grid directly)
        cols = math.ceil(math.sqrt(N))
        rows = math.ceil(N / cols)
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 4, rows * 3),
            layout="tight",
            subplot_kw={"polar": True},
        )
        flat_axes = np.array(axes).flatten() if np.size(axes) > 1 else np.array([axes])

        for n, condition in enumerate(n_conditions):
            ax = flat_axes[n]
            group = (
                self.layout_df[self.layout_df.Condition == condition]["name"].to_list()
                if self.layout_df is not None
                else condition
            )
            plots.phase_plot(
                self.phases, ax, self.phases.loc[group],
                pal=[self.bg_color, self.ent_color], order=self.order,
            )
            ax.set_title(condition)

        self._hide_unused(fig, flat_axes, N)
        plt.suptitle("Phase Calculation", fontsize=20, weight="bold")
        self.figures.append(fig)
        return self

    def add_period_estimation(self):
        if not self.has_results or not self.has_periods:
            return self

        res = self.result_df.set_index("CycID")
        mix = res.copy()
        # `reject` is written by the analysis run. A result table that lacks it
        # (e.g. loaded from an older export) must not take the report down.
        has_reject = "reject" in mix.columns
        hue_unit = "reject" if has_reject else None
        rows = self.result_df.shape[0]

        if self.layout_df is not None:
            trans = self.layout_df.set_index("name")
            mix = pd.concat([res, trans], axis=1)
            rows = mix.Condition.nunique()
            hue_unit = "Condition"

        per_col = "Periods"

        if self.layout_df is not None:
            fig, axes = plt.subplots(1, 2, figsize=(8, rows), layout="constrained")
            for n, ax in enumerate(axes):
                only_rhythmic = n == 1 and has_reject
                plot_data = mix[mix["reject"] == True] if only_rhythmic else mix
                title = "Only rhythmic" if only_rhythmic else "All samples"
                if plot_data.empty:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "No rhythmic samples", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
                    continue
                sns.pointplot(
                    plot_data, y="Condition", x=per_col, hue=hue_unit,
                    ax=ax, capsize=0.2,
                ).set(xlim=(self.period_len_min, self.period_len_max))
                sns.stripplot(
                    plot_data, y="Condition", x=per_col, hue=hue_unit,
                    edgecolor="k", linewidth=1, alpha=0.7, legend=False, ax=ax,
                )
                ax.set_ylabel("")
                ax.set_title(title)
        else:
            fig, _ = plt.subplots(1, 1, figsize=(4, rows / 2), layout="tight")
            sns.pointplot(
                mix, y=mix.index, x=per_col, hue=hue_unit,
                markeredgecolor="k", markeredgewidth=1, alpha=0.7,
            ).set(xlim=(self.period_len_min, self.period_len_max), ylabel="")

        plt.suptitle(
            f"Period Estimation ({self.period_estimation}-calculated)",
            fontsize=20, weight="bold",
        )
        self.figures.append(fig)
        return self

    def add_pie_charts(self):
        if not self.conditions or self.layout_df is None:
            return self
        if not self.has_results or self.q_col is None:
            return self

        N = len(self.conditions)
        fig, flat_axes, _, _ = self._make_grid(N)

        for n, group in enumerate(self.conditions):
            ax = flat_axes[n]
            sorter = self.layout_df[self.layout_df.Condition == group]["name"].unique()
            sorted_result = self.result_df[self.result_df["CycID"].isin(sorter)]
            plots.pie_chart(
                ax, sorted_result, method=self.method,
                group=group, thresh=self.thresh,
            )
            ax.set_title(group)

        self._hide_unused(fig, flat_axes, N)
        plt.legend(ncol=2)
        plt.suptitle(f"Rhythmicity Classification by {self.method}", fontsize=20, weight="bold")

        self.figures.append(fig)
        return self

    def add_pie_charts_model(self):
        # Only when Tempo was the selected method. This page used to appear in
        # every report, presenting a second verdict nobody asked for.
        if not self.conditions or self.layout_df is None:
            return self
        if not (self.has_ml and self.is_ml_method):
            return self

        vlag_pal = sns.color_palette('vlag', 6)

        rhythm_confidence = ["True high","True medium", "True low",
            "False low", "False medium", "False high"]
        pal = {k:vlag_pal[n] for n, k in enumerate(rhythm_confidence)}
            
        N = len(self.conditions)
        fig, flat_axes, _, _ = self._make_grid(N)

        for n, group in enumerate(self.conditions):
            ax = flat_axes[n]
            sorter = self.layout_df[self.layout_df.Condition == group]["name"].unique()
            sorted_result = self.result_df[self.result_df["CycID"].isin(sorter)]
            validation = (
                sorted_result
                .groupby(['is_rhythmic', 'confidence'])
                .size()
            )

            validation.index = [f"{ir} {conf}" for ir, conf in validation.index]
            values = {k: validation[k] for k in pal if k in validation}
            # ── pie chart ────────────────────────────────────────────────────────────────
            ax.pie(
                values.values(),
                labels=[f"{k} (n={v})" for k, v in values.items()],
                colors=[pal[k] for k in values],
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(width=0.6),                # donut style, easier to read
            )
            ax.set_title(group)

        self._hide_unused(fig, flat_axes, N)
        #plt.legend(ncol=2)
        plt.suptitle(f"Rhythmicity Classification by ML", fontsize=20, weight="bold")

        self.figures.append(fig)
        return self

    def add_statistical_comparisons(self):
        if not self.conditions or self.sum_stats is None or self.layout_df is None:
            return self
        if not self.has_results:
            return self

        result_cols = [c for c in self.result_df.columns if self.method in c]
        look_for = {
            "Rhythmicity": "BH.Q",
            "Period": "PERIOD",
            "Amplitude": "AMP",
        }
        colors = ["#F97068", "#57C4E5"]

        for cat in self.sum_stats.tested.unique():
            sorted_stats = self.sum_stats[
                (self.sum_stats.tested == cat) & (self.sum_stats.reject == True)
            ]
            look_col_matches = [c for c in result_cols if look_for[cat] in c.upper()]
            if not look_col_matches or sorted_stats.shape[0] == 0:
                continue

            look_col = look_col_matches[0]
            N = sorted_stats.shape[0]
            grid_cols = math.ceil(math.sqrt(N))
            grid_rows = math.ceil(N / grid_cols)
            fig, axes = plt.subplots(
                grid_rows, grid_cols,
                figsize=(grid_cols * 4, grid_rows * 5),
                layout="tight",
            )
            flat_axes = (
                np.array(axes).flatten() if np.size(axes) > 1 else np.array([axes])
            )

            for n, d in sorted_stats.reset_index().iterrows():
                names_g1 = self.layout_df.loc[
                    self.layout_df.Condition == d.group1, "name"
                ]
                names_g2 = self.layout_df.loc[
                    self.layout_df.Condition == d.group2, "name"
                ]
                values_g1 = self.result_df.loc[
                    self.result_df["CycID"].isin(names_g1), look_col
                ].values
                values_g2 = self.result_df.loc[
                    self.result_df["CycID"].isin(names_g2), look_col
                ].values

                ax = flat_axes[n]

                if cat == "Rhythmicity":
                    counts = [
                        (values_g1 < self.thresh).sum() / len(values_g1),
                        (values_g2 < self.thresh).sum() / len(values_g2),
                    ]
                    ax.bar([d.group1, d.group2], counts, color=colors, width=0.8)
                else:
                    bplot = ax.boxplot(
                        [values_g1, values_g2],
                        widths=0.8,
                        patch_artist=True,
                        showmeans=True,
                    )
                    for patch, color in zip(bplot["boxes"], colors):
                        patch.set_facecolor(color)
                    for mean in bplot["means"]:
                        mean.set_color("k")
                        mean.set_linewidth(2)
                    for median in bplot["medians"]:
                        median.set_color("black")
                        median.set_linewidth(2)

                    # Significance annotation
                    p = d["p-val"]
                    stars = (
                        "***" if p < 0.001 else
                        "**" if p < 0.01 else
                        "*" if p < 0.05 else "ns"
                    )
                    y_max = max(np.max(values_g1), np.max(values_g2))
                    ax.annotate(
                        "", xy=(2, y_max * 1.08), xytext=(1, y_max * 1.08),
                        arrowprops=dict(arrowstyle="-", color="black"),
                    )
                    ax.text(1.5, y_max * 1.12, stars, ha="center", fontsize=14)

                n1, n2 = len(values_g1), len(values_g2)
                ax.set_xticklabels([f"{d.group1}\n(n={n1})", f"{d.group2}\n(n={n2})"])
                ax.set_ylabel(cat)
                ax.set_title(f"p-val: {d['p-val']:.4f}")

            self._hide_unused(fig, flat_axes, N)
            plt.suptitle(f"{cat} Differences", fontsize=20, weight="bold")
            self.figures.append(fig)

        return self

    def add_conditions_overview(self):
        if not self.conditions or self.layout_df is None:
            return self

        N = len(self.conditions)
        fig, ax, _, _ = self._make_grid(N)

        xmin = self.df[self.t_col].min()
        xmax = self.df[self.t_col].max()
        xtick_start = (xmin // 24) * 24
        xtick_end = ((xmax // 24) + 1) * 24
        xticks = list(range(int(xtick_start), int(xtick_end), 24))


        for n, group in enumerate(self.conditions):
            plots.grouped_plot_traces_export(
                ax[n], self.df, self.t_col, self.t0, self.t1,
                group=group, layout=self.layout_df,
                bg_color=self.bg_color, ent=self.ent,
                ent_days=self.ent_days, order=self.order,
                T=self.T, color=self.ent_color, unit=self.unit,
            )

            # Numbers only if there are numbers. This block used to run
            # unconditionally and was the crash when no analysis had been run.
            if not self.has_results:
                continue

            sorter = self.layout_df[self.layout_df.Condition == group]["name"].unique()
            sorted_result = self.result_df[self.result_df["CycID"].isin(sorter)]
            if sorted_result.empty:
                continue

            lines = []
            if self.is_ml_method and self.has_ml:
                pct = 100 * (sorted_result["probability_rhythmic"] > 0.5).sum() / len(sorted_result)
                lines.append(f"$\\bf{{Rhythmic}}$: {pct:.0f}% of {len(sorted_result)} (Tempo)")
            elif self.q_col is not None:
                pct = 100 * (sorted_result[self.q_col] <= self.thresh).sum() / len(sorted_result)
                lines.append(f"$\\bf{{Rhythmic}}$: {pct:.0f}% of {len(sorted_result)} ({self.method})")
            if self.has_periods and np.isfinite(sorted_result["Periods"].mean()):
                lines.append(
                    f"$\\bf{{\\tau}}$: {sorted_result['Periods'].mean():.1f} "
                    f"± {sorted_result['Periods'].std():.1f} h"
                )
            if lines:
                ax[n].annotate(
                    "\n".join(lines), xy=(0.98, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CFCFCB", alpha=0.85),
                )

        self._hide_unused(fig, ax, N)
        self.figures.append(fig)
        return self


    def add_group_traces(self):
        """Grouped overview trace + per-sample individual traces, one page per condition."""
        if not self.conditions or self.layout_df is None:
            return self

        for group in self.conditions:
            # --- overview trace (with optional stats text panel) ---
            if self.has_results and self.q_col is not None:
                fig, ax = plt.subplots(
                    2, 1, figsize=(10, 6.5), height_ratios=(1, 5)
                )
                plots.grouped_plot_traces_export(
                    ax[1], self.df, self.t_col, self.t0, self.t1,
                    group=group, layout=self.layout_df,
                    bg_color=self.bg_color, ent=self.ent,
                    ent_days=self.ent_days, order=self.order,
                    T=self.T, color=self.ent_color, unit=self.unit,
                )
                sorter = self.layout_df[
                    self.layout_df.Condition == group
                ]["name"].unique()
                sorted_result = self.result_df[
                    self.result_df["CycID"].isin(sorter)
                ]
                plots.text(
                    ax[0], sorted_result,
                    method=self.method, group=group, thresh=self.thresh,
                )
            else:
                fig, ax_single = plt.subplots(1, figsize=(10, 4))
                plots.grouped_plot_traces_export(
                    ax_single, self.df, self.t_col, self.t0, self.t1,
                    group=group, layout=self.layout_df,
                    bg_color=self.bg_color, ent=self.ent,
                    ent_days=self.ent_days, order=self.order,
                    T=self.T, color=self.ent_color, unit=self.unit,
                )
            self.figures.append(fig)

            # --- individual sample traces for this group ---
            sorter_df = self.layout_df[self.layout_df["Condition"] == group]
            names = sorter_df["name"].to_list()
            N = len(names)
            fig, flat_axes, _, _ = self._make_grid(N)

            xmin = self.df[self.t_col].min()
            xmax = self.df[self.t_col].max()
            xtick_start = (xmin // 24) * 24
            xtick_end = ((xmax // 24) + 1) * 24
            xticks = list(range(int(xtick_start), int(xtick_end), 24))

            for n, subgroup in enumerate(names):
                ax = flat_axes[n]
                ax.set_facecolor(self.bg_color)
                ax.plot(self.df[self.t_col], self.df[subgroup])
                self._draw_entrainment(ax)
                ax.set_title(self._get_sample_title(subgroup), loc='left', fontsize=10)
                ax.set_xlabel("Time (h)")
                ax.set_ylabel(self.unit)
                ax.set_xticks(xticks)

            self._hide_unused(fig, flat_axes, N)
            plt.suptitle(group, weight="bold", fontsize=20)
            self.figures.append(fig)

        return self

    def add_individual_traces(self):
        """Split or simple per-column traces (no layout grouping)."""
        for col in self.data_cols:
            title = self._get_sample_title(col)

            if self.ent_days > 0:
                fig = plots.split_plot(
                    self.df, self.t_col, col,
                    ent=self.ent, ent_days=self.ent_days,
                    unit=self.unit, bg_color=self.bg_color,
                    band_color=self.ent_color, order=self.order,
                    T=self.T, title=title,
                )
            else:
                fig = plots.simple_plot(
                    self.df, self.t_col, col, title=title
                )
            self.figures.append(fig)

        return self

    # ------------------------------------------------------------------ #
    #  Orchestration                                                       #
    # ------------------------------------------------------------------ #

    def build(self):
        """
        Assemble all sections in order. Returns self so you can chain .to_pdf().
        """
        self.figures = []

        # Cover + summary
        self.figures.append(self._make_cover_page())
        summary_fig = self._make_summary_table()
        if summary_fig is not None:
            self.figures.append(summary_fig)

        # Section: Phase
        if self.ent_days > 0 and self.phases is not None:
            self.figures.append(self._make_section_page("Phase Analysis"))
            self.add_phase_plots()

        # Section: Period estimation
        if self.has_results and self.has_periods:
            self.figures.append(self._make_section_page("Period Estimation"))
            self.add_period_estimation()

        # Section: Group-level results
        if self.conditions:
            self.add_conditions_overview()
            self.figures.append(self._make_section_page("Group Results"))
            if self.has_results:
                self.add_pie_charts()
                self.add_pie_charts_model()
            if self.sum_stats is not None:
                self.add_statistical_comparisons()
            self.add_group_traces()

        # Section: Individual traces
        if self.data_cols:
            self.figures.append(self._make_section_page("Individual Traces"))
            self.add_individual_traces()

        return self

    def to_pdf(self):
        """Render all figures to a BytesIO PDF buffer."""
        buffer = BytesIO()
        with PdfPages(buffer) as pdf:
            for i, fig in enumerate(self.figures, 1):
                # Page number in bottom-right corner
                fig.text(
                    0.98, 0.01, str(i),
                    ha="right", va="bottom",
                    fontsize=9, color="gray",
                    transform=fig.transFigure,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            d = pdf.infodict()
            d["Title"] = "Rhythmicity Report"
            d["Author"] = "CycleAnalysis"
            d["CreationDate"] = datetime.now()

        buffer.seek(0)
        return buffer
