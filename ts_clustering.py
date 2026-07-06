'''
Running the Script:

```bash
python ts_clustering.py \
--tss_path "data/tss.h5" \
--method n2d \
--method_config "model_configs/n2d_config.yml" \
--min_clusters 10 \
--max_clusters 15 \
--periods DAY WEEK \
--freqs 60 650 \
--optim_freq \
--threshold 0.001
```

```powershell
python ts_clustering.py `
--tss_path "data/tss.h5" `
--method n2d `
--method_config "model_configs/n2d_config.yml" `
--min_clusters 10 `
--max_clusters 15 `
--periods DAY WEEK `
--freqs 60 650 `
--optim_freq `
--threshold 0.001
```

'''

import os, time, warnings, argparse
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib
# Allows to avoid the error "Fail to create pixmap with Tk_GetPixmap in TkImgPhotoInstanceSetSize" 
# of releasing resources of the Tk/X11Tk backend when there are a large number of graphics 
matplotlib.use('Agg')       
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from utils.preprocessing import optimize_freq, resample_df
from clustering.clusterizers import CONFIGS, CLUSTERIZERS
from utils.metrics import gap_statistic, elbow
from utils.plot import (
    plot_silhouette,
    plot_silhouettes,
    plot_gap_statistics,
    plot_elbows,
    plot_clusters,
    plot_tss_in_clusters
)


N_CLUSTERS_METRICS = ['silhouette', 'gap_statistic', 'elbow']
METRICS = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
PERIODS_ALL = ['INFINITE', 'HALF_YEAR', 'MONTH', 'WEEK', 'DAY']
BEST_METRICS = [
    ('silhouette', 'max'),
    ('calinski_harabasz', 'max'),
    ('davies_bouldin', 'min'),
]

def validate_args(args, parser):
    errors = []
    if args.min_clusters > args.max_clusters:
        errors.append(f'--min_clusters ({args.min_clusters}) must be <= --max_clusters ({args.max_clusters})')
    if args.threshold < 0:
        errors.append(f'--threshold must be >= 0: {args.threshold}')
    if any(f <= 0 for f in args.freqs):
        errors.append(f'all values --freqs must be positive: {args.freqs}')
    if not set(args.periods) <= set(PERIODS_ALL):
        errors.append(f'all periods --periods must be contained in {PERIODS_ALL}: {args.periods}')
    if args.method not in CLUSTERIZERS:
        errors.append(f'method --method must be contained in {list(CLUSTERIZERS)}: {args.method}')
    if args.tss_path is not None and not os.path.isfile(args.tss_path):
        errors.append(f'--tss_path does not exist or is not a file: {args.tss_path}')
    if args.method_config is not None and not os.path.isfile(args.method_config):
        errors.append(f'--method_config does not exist or is not a file: {args.method_config}')    
    if errors:
        parser.error('\n'.join(errors))

def bound_times(tss, bound):
    return [ts['timestamp'].iloc[bound] for _, ts in tss.items()]

def median_bounds(tss):
    start_times = sorted(bound_times(tss, 0))
    end_times = sorted(bound_times(tss, -1))
    # median start (across all ts) + 10min
    start_median = start_times[len(start_times) // 2] + pd.Timedelta('10min')
    # median end (across all ts) - 10min
    end_median = end_times[len(end_times) // 2] - pd.Timedelta('10min')
    start_median = pd.to_datetime(start_median)
    end_median = pd.to_datetime(end_median)
    return start_median, end_median    

def process_tss(tss_rp, in_period_tss, periods, freqs):
    tss_proc = {
        period:{freq:{} for freq in freqs[period]} 
        for period in periods
    }
    # сохраняем до обработки для визуализации рядов в методе plot_tss_in_clusters
    clipped_tss = deepcopy(tss_proc)

    for period in periods:
        start_time, end_time = median_bounds(tss_rp[period])
        for freq in freqs[period]:
            for name, ts in in_period_tss[period].items():
                clipped_ts = ts[(ts['timestamp'] >= start_time) & (ts['timestamp'] <= end_time)]
                proc_ts = resample_df(ts, freq=freq, return_freq=False)
                proc_ts['target'] = proc_ts['target'].interpolate(limit_direction="both")
                proc_ts = proc_ts[(proc_ts['timestamp'] >= start_time) & (proc_ts['timestamp'] <= end_time)]

                # after truncation, empty series may have reappeared, also skip constant series
                if proc_ts['target'].any() and len(proc_ts['target'].unique()) > 1:
                    proc_ts['target'] = MinMaxScaler().fit_transform(proc_ts[['target']])
                    clipped_tss[period][freq][name] = clipped_ts
                    tss_proc[period][freq][name] = proc_ts         
    return tss_proc, clipped_tss

def clustering(tss_proc, n_clusters_range, method, cfg, period_freq_pairs):
    all_names = []
    periods = list({p for p, _ in period_freq_pairs})
    for p, tss_p in tss_proc.items():
        if not p in periods:
            continue
        for f, tss_f in tss_p.items():
            all_names.extend(tss_f.keys())
    all_names = np.unique(all_names)

    columns = [f'{i} clusters' for i in n_clusters_range]
    
    columns_index = pd.MultiIndex.from_tuples(
        [(period, freq, col) for period, freq in period_freq_pairs for col in columns],
        names=['period', 'freq', 'n_clusters']
    )
    # rows: ts, columns: period -> frequency -> n_clusters
    tss_clusterings = pd.DataFrame(columns=columns_index, index=all_names).sort_index(axis=1)

    # freq -> n_clusters -> labels, embeds
    results = defaultdict(dict)

    clusterizer_cls = CLUSTERIZERS[method]

    for period, freq in period_freq_pairs:
        names, tss_ = zip(*tss_proc[period][freq].items())
        x = np.array([ts['target'] for ts in tss_])          # todo проверить что все ряды одинаковой длины
        results[period][freq] = {}

        for n_clusters in n_clusters_range:
            start = time.time()
            print(f'freq={freq}  n_clusters={n_clusters}  method={method}')

            clusterizer = clusterizer_cls(n_clusters, cfg)
            clusterizer.fit(x)
            labels, embeds, embed_2d = clusterizer.predict(x), clusterizer.embed(x), clusterizer.embed_2d(x)
            results[period][freq][n_clusters] = labels, embeds, embed_2d
            tss_clusterings.loc[names, (period, freq, f'{n_clusters} clusters')] = labels
            end = time.time()
            print('elapsed time (seconds) =', end - start)
    return results, tss_clusterings

def compute_metrics(results, tss_proc, period_freq_pairs, n_clusters_range, method, cfg):
    columns_index = pd.MultiIndex.from_tuples(
        [(period, freq, metric) for period, freq in period_freq_pairs for metric in N_CLUSTERS_METRICS],
        names=['period', 'freq', 'metric']
    )
    n_clusters_metrics = pd.DataFrame(columns=columns_index, index=n_clusters_range, dtype=float)
    n_clusters_metrics.index.name = 'cluster_count'

    # metrics: frequency -> df[n_clusters, clustering metrics]
    metrics = defaultdict(dict)

    for period, freq in period_freq_pairs:
        # all metrics are computed on the embeddings.
        metrics[period][freq] = pd.DataFrame(
            columns=METRICS,
            index=[f'{n_clusters} clusters' for n_clusters in n_clusters_range],
            dtype=float
        )
        
        for n_clusters in n_clusters_range:
            print('cluster: ', n_clusters)
            labels, embeds, _ = results[period][freq][n_clusters]
            tss_ = tss_proc[period][freq]
            # you can use a simpler clusterizer to speed up
            clusterizer = CLUSTERIZERS[method](n_clusters, cfg)
            kshape_args = {}
            gap_elbow_input = embeds
            if method == 'kshape':
                gap_elbow_input = np.array([ts['target'] for ts in tss_.values()])
                kshape_args = {'dist': clusterizer.dists}
                silhouette = silhouette_score(embeds, labels, metric='precomputed')
            else:
                silhouette = silhouette_score(embeds, labels)
            
            gap_statistic_ = gap_statistic(gap_elbow_input, labels, clusterizer, n_clusters, **kshape_args)
            # elbow = WCSS (within cluster distance squared sum)
            elbow_ = elbow(gap_elbow_input, labels, n_clusters, **kshape_args)

            n_clusters_metrics.loc[n_clusters, (period, freq, N_CLUSTERS_METRICS)] = [
                silhouette, gap_statistic_, elbow_    
            ]
            metrics_uncommon = [None, None]
            if method != 'kshape':
                metrics_uncommon = [
                    calinski_harabasz_score(embeds, labels), 
                    davies_bouldin_score(embeds, labels)
                ]
            row = f'{n_clusters} clusters'
            metrics[period][freq].loc[row, METRICS] = [silhouette, *metrics_uncommon]

    return n_clusters_metrics, metrics

def join_path(paths):
    return os.path.join(*paths)

def mkdirs(dir):
    os.makedirs(dir, exist_ok=True)


def main(args):
    method = args.method
    min_clusters = args.min_clusters
    max_clusters = args.max_clusters
    periods = args.periods
    freqs = [str(f) + 's' for f in args.freqs]
    freqs = {period:freqs for period in periods}

    dir_res = 'results'
    dir_plots = os.path.join(dir_res, 'plots')
    mkdirs_ = [(dir_res, 'plots'), (dir_plots, 'clusters_scatter'), (dir_plots, 'clusters')]
    mkdirs_ = [join_path(paths) for paths in mkdirs_]
    for dir_ in mkdirs_:
        mkdirs(dir_)

    tss_file = args.tss_path
    with pd.HDFStore(tss_file) as store:
        tss = {key.lstrip("/"): store[key] for key in store.keys()}

    tss_rp = defaultdict(dict)
    for name, ts in tss.items():
        rp = name.split('__')[-1]
        if not ts['target'].any():
            continue
        tss_rp[rp][name] = ts

    in_period_tss = defaultdict(dict)

    for period in PERIODS_ALL:
        start_median, end_median = median_bounds(tss_rp[period])
        for tss_ in list(tss_rp.values()):
            for name, ts in tss_.items():
                if (ts['timestamp'].iloc[0] <= start_median) and (ts['timestamp'].iloc[-1] >= end_median):
                    in_period_tss[period][name] = ts
        print(f'Period: {period:10}  overlap: {len(in_period_tss[period]):<5}')

    if args.optim_freq:
        optim_freqs = {}
        pbar_periods = tqdm(periods, ncols=80, desc='finding optimal frequencies ...')
        for period in pbar_periods:
            pbar_periods.set_description(f"Period: {period}")
            optim_freqs_ = [
                optimize_freq(ts, threshold=args.threshold) 
                for ts in in_period_tss[period].values() if len(ts) > 2
            ]
            freqs_, counts = np.unique(optim_freqs_, return_counts=True)
            optim_freqs[period] = freqs_[counts.argmax()]
        freqs = {period : list(set(freqs[period] + [optim_freqs[period]])) for period in periods}
        print(f'optim_freqs = {optim_freqs}')

    period_freq_pairs = [(period, freq) for period in periods for freq in freqs[period]]
    tss_proc, clipped_tss = process_tss(tss_rp, in_period_tss, periods, freqs)

    for period, freq in period_freq_pairs:
        maxlen = max([len(ts['target']) for name, ts in tss_proc[period][freq].items()])
        for name, ts in tss_proc[period][freq].items():
            if len(ts['target']) < maxlen:
                last_timestamp, last_target = ts['timestamp'].loc[ts.index[-1]], ts['target'].loc[ts.index[-1]]
                ts.loc[ts.index[-1]+1] = [last_timestamp + pd.Timedelta(freq), last_target]

    cfg = CONFIGS[method]()
    if args.method_config:
        cfg = cfg.from_config(args.method_config)

    n_clusters_range = range(min_clusters, max_clusters+1)
    results, tss_clusterings = clustering(
        tss_proc, n_clusters_range, method, cfg, period_freq_pairs)
  
    n_clusters_metrics, metrics = compute_metrics(
        results, tss_proc, period_freq_pairs, n_clusters_range, method, cfg
    )

    out_metrics = os.path.join(dir_res, 'metrics')
    os.makedirs(out_metrics, exist_ok=True)
    n_clusters_metrics.sort_index(axis=1, inplace=True)
    n_clusters_metrics.to_csv(os.path.join(out_metrics, f'clusters_count_metrics__{method}.csv'))

    # plot_silhouette for each clustering (with a different number of clusters)
    for period, freq in period_freq_pairs:
        n_range = list(range(min_clusters, max_clusters + 1))
        n_rows = len(n_range)
        fig, axes = plt.subplots(n_rows, 2, figsize=(18, 7 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, 2)

        for row, n_clusters in enumerate(n_range):
            labels, embeds, embed_2d = results[period][freq][n_clusters]
            plot_silhouette(embeds, embed_2d, labels, n_clusters, ax1=axes[row, 0], ax2=axes[row, 1])

        plt.tight_layout()
        plt.savefig(os.path.join(dir_plots, f'silhouettes__{method}__{period}__{freq}.png'), dpi=150)

    # clustering quality metric plots (silhouette, calinski harabasz score, davies bouldin score) 
    metrics_count = len(N_CLUSTERS_METRICS)
    row_count = len(period_freq_pairs)

    plt.figure(figsize=(20, 5 * row_count))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    plot_func = {
        'silhouette': plot_silhouettes,
        'gap_statistic': plot_gap_statistics,
        'elbow': plot_elbows
    }

    i = 0
    for i, (period, freq) in enumerate(period_freq_pairs):
        for k, metric in enumerate(N_CLUSTERS_METRICS):
            plt.subplot(row_count, metrics_count, i * metrics_count + k + 1)
            metric_values = pd.to_numeric(
                n_clusters_metrics.loc[min_clusters:max_clusters, (period, freq, metric)]
            )
            plot_func[metric](metric_values, f'period={period}\nfreq={freq}')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_plots, f'metrics__{method}.png'), dpi=150)

    for period, freq in period_freq_pairs:
        run_name = f'{method}__{period}__{freq}__{min_clusters}_{max_clusters}_clusters'
        out_file = os.path.join(out_metrics, f'metrics__{run_name}.csv')
        metrics[period][freq].to_csv(out_file)
        metrics_best_s = ''
        for m, mode in BEST_METRICS:
            if method == 'kshape' and m != 'silhouette':
                continue
            values = pd.to_numeric(metrics[period][freq][m])
            idx = values.idxmax() if mode == 'max' else values.idxmin()
            val = values.max() if mode == 'max' else values.min()
            metrics_best_s += f'{mode} {m:18}  {idx:12}  {val}\n'
        with open(os.path.join(out_metrics, f'best_metrics__{run_name}.txt'), 'w') as f:
            f.write(metrics_best_s)
        
    tss_clusterings.to_csv(os.path.join(out_metrics, f'metrics__{method}.csv'))

    for period, freq in period_freq_pairs:
        for n_clusters in n_clusters_range:
            labels, _, embeds_2d = results[period][freq][n_clusters]
            names = list(tss_proc[period][freq])
            out_file = os.path.join(
                dir_plots, 
                'clusters_scatter', 
                f'cluster__{method}__{period}__{freq}__{n_clusters}_clusters.html'
            )
            plot_clusters(embeds_2d, labels, n_clusters, names, out_file)

    for period, freq in period_freq_pairs:
        for n_clusters in range(min_clusters, max_clusters + 1):
            out_path = os.path.join(dir_plots, 'clusters', f'clusters__{method}__{period}__{freq}__{n_clusters}_clusters')
            header_text = f'\nMethod: {method}\nPeriod: {period}\nFrequency: {freq}\nClusters count: {n_clusters}'

            names = list(clipped_tss[period][freq].keys())
            col = (period, freq, str(n_clusters) + ' clusters')
            labels = tss_clusterings.loc[names, col].values
            labels_names = tss_clusterings.loc[names, col].index.values

            plot_tss_in_clusters(
                clipped_tss[period][freq],  
                labels, labels_names, n_clusters,
                header_text=header_text,
                out_path=out_path, 
                out_mode=('all',)
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--tss_path", type=str, default=os.path.join('data', 'tss.h5'), help='input time series path')
    parser.add_argument("--method", type=str, default='n2d', help='clustering method ("n2d", "idec", "kshape")')
    parser.add_argument("--method_config", type=str, help="clustering method configuration")
    parser.add_argument("--min_clusters", type=int, default=10, help='minimum number of clusters in the range of clusters being tested')
    parser.add_argument("--max_clusters", type=int, default=15, help='maximum number of clusters in the range of clusters to be tested')
    parser.add_argument("--periods", nargs='+', type=str, default=['DAY', 'WEEK'], help="time series periods (retention policies) to be tested")
    parser.add_argument("--freqs", nargs='+', type=int, default=[60, 650], help="time series frequencies to be tested")
    parser.add_argument("--optim_freq", action='store_true', help='whether to calculate and test at the optimal frequency')
    parser.add_argument("--threshold", type=float, default=0.001, help='error threshold for calculating the optimal frequency')

    args = parser.parse_args()
    validate_args(args, parser)
    main(args)
