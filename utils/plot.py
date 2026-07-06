import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score

import numpy as np
import pandas as pd


def plot_durations(tss):
    for y, (name, ts) in enumerate(tss):
        plt.scatter(ts['timestamp'], [y+1]*len(ts['timestamp']))
        plt.annotate(text=name, xy=(ts['timestamp'].values[-1], y+1))


def plot_silhouette(x, x_2d, cluster_labels, n_clusters, ax1=None, ax2=None):
    '''
    left plot: For each cluster, silhouette coefficient values for all points in the cluster:
    points of the cluster are arranged along the vertical axis, the right edge of each
    bar shows the silhouette coefficient value for that point.
    dashed line: is the mean silhouette coefficient value (over all points).
    right plot: points in 2D space colored according to their cluster
    '''
    if ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])

    silhouette_avg = silhouette_score(x, cluster_labels)
    print(f"n_clusters = {n_clusters}\n" + f"avg silhouette_score = {silhouette_avg}")
    silhouettes = silhouette_samples(x, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        i_cluster_silhouettes = silhouettes[cluster_labels == i]
        i_cluster_silhouettes.sort()
        cluster_i_size = i_cluster_silhouettes.shape[0]
        y_upper = y_lower + cluster_i_size
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            i_cluster_silhouettes,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax1.text(-0.05, y_lower + 0.5 * cluster_i_size, str(i))
        y_lower = y_upper + 10

    ax1.set_title(f"Silhouette plot, n_clusters={n_clusters}")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        x_2d[:, 0], x_2d[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )
    ax2.set_title(f"Clustered data, n_clusters={n_clusters}")



# plots of silhouette score, gap statistic, and elbow method versus the number of clusters

def _plot_metric_by_cluster_count(metric, label, ylabel, title, best='max'):
    plt.plot(metric, linewidth=3, label=label)
    plt.legend(loc='upper left')
    plt.xlabel('cluster count')
    plt.ylabel(ylabel)
    plt.xticks(metric.index)
    plt.title(title)

    idx_best = metric.idxmax() if best == 'max' else metric.idxmin()
    plt.scatter(idx_best, metric.loc[idx_best], s=250, c='r')
    plt.grid(True)

def plot_silhouettes(silhouettes, label):
    _plot_metric_by_cluster_count(
        silhouettes, label,
        ylabel='silhouette Value',
        title='silhouette by cluster count',
        best='max',
    )

def plot_gap_statistics(gap_statistics, label):
    _plot_metric_by_cluster_count(
        gap_statistics, label,
        ylabel='gap statistic',
        title='gap statistic by cluster count',
        best='max',
    )

def plot_elbows(elbows, label):
    _plot_metric_by_cluster_count(
        elbows, label,
        ylabel='elbow',
        title='elbow by cluster count',
        best='min',
    )

def plot_tss_in_clusters(
    tss, 
    labels: np.ndarray, labels_names: np.ndarray, n_clusters: int,
    header_text: str,
    out_path: str,
    out_mode: tuple[str, ...] = ('all', 'separately'),
):
    '''
    Plots time series within each cluster for the given clustering (number of clusters), model, frequency, and period.
    
    Parameters
    ----------
    tss - dict storing preprocessed time series for the given optimal frequency and given period
    labels
    labels_names
    n_clusters - clusters count
    out_path
    out_mode - display mode: 
        separately - each cluster as one column in a separate file
        all - all clusters in one file, each in its own column
    '''
    def cluster_names_for(i_cluster):
        return labels_names[labels == i_cluster]

    def plot_series(ax, name, legend_fontsize=None):
        ax.plot(tss[name]['timestamp'].values, tss[name]['target'].values, label=name)
        legend_kwargs = {'loc': 'upper left'}
        if legend_fontsize is not None:
            legend_kwargs['fontsize'] = legend_fontsize
        ax.legend(**legend_kwargs)

    def scaled_fontsize(height_inches, frac=0.35, min_fs=8, max_fs=50):
        return max(min_fs, min(max_fs, height_inches * 72 * frac))

    def reserve_top_margin(fig_height_inches, title_fs, n_lines, line_spacing=1.3, pad_inches=0.15):
        '''
        Calculates the proportion of the figure (0..1) that needs to be left above the title, 
        based on the actual height of the text in inches.
        '''
        title_height_inches = (title_fs / 72) * n_lines * line_spacing + pad_inches
        top_margin = 1 - title_height_inches / fig_height_inches
        return max(0.5, min(0.98, top_margin))

    def add_cluster_label(ax, i_cluster, fontsize):
        ax.text(0.4, 0.0, f'Cluster {i_cluster}', fontsize=fontsize)
        ax.axis('off')

    if 'separately' in out_mode:
        os.makedirs(out_path, exist_ok=True)

        for i_cluster in range(n_clusters):
            cluster_names = cluster_names_for(i_cluster)
            size = len(cluster_names)
            fig_height = min(5 * size, 500)
            fig, axes = plt.subplots(nrows=size + 1, ncols=1, figsize=(40, fig_height), squeeze=False)
            axes = axes[:, 0]

            row_height = fig_height / (size + 1)
            header_fs = scaled_fontsize(row_height, frac=0.5, min_fs=8, max_fs=40)
            label_fs = scaled_fontsize(row_height, frac=0.6, min_fs=10, max_fs=50)
            legend_fs = scaled_fontsize(row_height, frac=0.25, min_fs=6, max_fs=15)

            axes[0].text(0.05, 0.3, header_text, fontsize=header_fs)
            add_cluster_label(axes[0], i_cluster, fontsize=label_fs)

            for row, name in enumerate(cluster_names, start=1):
                plot_series(axes[row], name, legend_fontsize=legend_fs)

            plt.subplots_adjust(hspace=0.5)
            fig.savefig(f'{out_path}/cluster_{i_cluster}.jpg')
            plt.close(fig)
            plt.close('all')
            print(f'cluster_{i_cluster}.jpg')
        
    if 'all' in out_mode:
        _, cluster_sizes = np.unique(labels, return_counts=True)
        max_cluster_size = max(cluster_sizes)
        fig_width = min(30 * n_clusters, 350)
        fig_height = min(3 * max_cluster_size, 350)

        nrows_total = max_cluster_size + 1
        fig, axes = plt.subplots(
            nrows=nrows_total, ncols=n_clusters, 
            figsize=(fig_width, fig_height),
            squeeze=False
        )

        row_height = fig_height / nrows_total
        n_header_lines = header_text.count('\n') + 1
        title_fs = scaled_fontsize(fig_height, frac=0.06, min_fs=14, max_fs=40)
        label_fs = scaled_fontsize(row_height, frac=0.6, min_fs=10, max_fs=50)
        legend_fs = scaled_fontsize(row_height, frac=0.25, min_fs=6, max_fs=15)

        top_margin = reserve_top_margin(fig_height, title_fs, n_header_lines)

        fig.suptitle(header_text, fontsize=title_fs, y=0.995, linespacing=1.3)
        plt.subplots_adjust(top=top_margin, wspace=0.05)
    
        for i_cluster in range(n_clusters):
            cluster_names = cluster_names_for(i_cluster)
            add_cluster_label(axes[0, i_cluster], i_cluster, fontsize=label_fs)

            row = 1
            for name in cluster_names:
                plot_series(axes[row, i_cluster], name, legend_fontsize=legend_fs)
                row += 1
            for empty_row in range(row, max_cluster_size + 1):
                axes[empty_row, i_cluster].axis('off')
        
        out_file = f'{out_path}.jpg'
        fig.savefig(out_file)
        plt.close(fig)
        plt.close('all')
        print(out_file)

def plot_clusters(x, labels, n_clusters, names, out_path=None, show=False):
    import plotly.express as px
    import pandas as pd
    import numpy as np

    labels_arr = np.asarray(labels).astype(str)
    names_arr = np.asarray(names)

    df = pd.DataFrame({
        "x": np.asarray(x[:, 0]),
        "y": np.asarray(x[:, 1]),
        "cluster": labels_arr,
        "name": names_arr,
    })

    # считаем размер каждого кластера
    sizes = df["cluster"].value_counts().to_dict()

    # метка для легенды: "0 (n=123)"
    df["cluster_label"] = df["cluster"].apply(lambda c: f"{c} (n={sizes[c]})")

    # сохраняем порядок кластеров по номеру, но уже с новыми подписями
    unique_labels = sorted(df["cluster"].unique(), key=lambda s: int(s))
    label_order = [f"{c} (n={sizes[c]})" for c in unique_labels]

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster_label",
        text="name",
        color_discrete_sequence=px.colors.qualitative.Plotly[:n_clusters],
        hover_name="name",
        category_orders={"cluster_label": label_order},
    )
    fig.update_traces(textposition="top center", textfont=dict(size=8))
    fig.update_layout(
        width=1200, height=800, dragmode='pan',
        legend_title_text="cluster (size)",
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                x=1.04, y=0.5,
                xanchor="left",
                yanchor="top",
                showactive=True,
                buttons=[
                    dict(label="Show captions", method="restyle", args=[{"mode": "markers+text"}]),
                    dict(label="Hide captions", method="restyle", args=[{"mode": "markers"}]),
                ],
            )
        ],
    )
    if show:
        fig.show(config={'scrollZoom': True})
    if out_path:
        fig.write_html(out_path, config={'scrollZoom': True})