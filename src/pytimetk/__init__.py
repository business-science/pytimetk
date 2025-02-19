# *** Import everything to make pytimetk a standalone package ***

from .plot import (
    plot_timeseries,
    plot_anomalies,
    plot_anomalies_decomp,
    plot_anomalies_cleaned,
    plot_correlation_funnel,
)
from .plot.theme import (
    theme_timetk, 
    palette_timetk,
)
from .core import (
    summarize_by_time,
    apply_by_time,
    anomalize,
    correlate, 
    binarize,
    future_frame,
    filter_by_time,
    make_future_timeseries,
    pad_by_time,
    ts_features,
    ts_summary, 
    get_diff_summary, 
    get_date_summary,
    get_frequency_summary, 
    get_frequency, 
    get_seasonal_frequency, 
    get_trend_frequency, 
    timeseries_unit_frequency_table, 
    time_scale_template,
    make_weekday_sequence, 
    make_weekend_sequence,    
)

from .feature_engineering import (
    get_timeseries_signature, 
    augment_timeseries_signature,
    augment_hilbert,
    augment_wavelet,
    augment_holiday_signature, 
    get_holiday_signature,
    augment_lags,
    augment_leads,
    augment_diffs,
    augment_pct_change,
    augment_rolling, 
    augment_rolling_apply, 
    augment_expanding,
    augment_expanding_apply,
    augment_fourier,
    augment_ewm,
)

from .crossvalidation import (
    TimeSeriesCV, 
    TimeSeriesCVSplitter
)

from .finance import (
    augment_cmo,
    augment_macd,
    augment_bbands,
    augment_ppo,
    augment_rsi,
    augment_atr,
    augment_roc,
    augment_qsmomentum,
    augment_drawdown,
    augment_rolling_risk_metrics,
    augment_fip_momentum,
    augment_stochastic_oscillator,
    augment_adx,
    augment_hurst_exponent,
    augment_ewma_volatility,
    augment_regime_detection,
)
from .datasets import (
    load_dataset, 
    get_available_datasets
)
from .utils import (
    week_of_month, 
    floor_date, 
    ceil_date, 
    is_holiday,
    reduce_memory_usage,
    flatten_multiindex_column_names, 
    glimpse, 
    drop_zero_variance, 
    transform_columns, 
    sort_dataframe,
    parallel_apply, 
    progress_apply,
    parse_freq_str,
)



from importlib.metadata import version
__version__ = version('pytimetk')
__author__ = 'Matt Dancho (Business Science)'
