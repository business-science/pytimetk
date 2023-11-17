
import pandas as pd
import polars as pl

'''
The Chande Momentum Oscillator (CMO), developed by Tushar Chande, is a technical analysis tool used to gauge the momentum of a financial instrument. It is similar to other momentum indicators like the Relative Strength Index (RSI), but with some distinct characteristics. Here's what the CMO tells us:

Momentum of Price Movements:

The CMO measures the strength of trends in price movements. It calculates the difference between the sum of gains and losses over a specified period, normalized to oscillate between -100 and +100.
Overbought and Oversold Conditions:

Values close to +100 suggest overbought conditions, indicating that the price might be too high and could reverse.
Conversely, values near -100 suggest oversold conditions, implying that the price might be too low and could rebound.
Trend Strength:

High absolute values (either positive or negative) indicate strong trends, while values near zero suggest a lack of trend or a weak trend.
Divergences:

Divergences between the CMO and price movements can be significant. For example, if the price is making new highs but the CMO is declining, it may indicate weakening momentum and a potential trend reversal.
Crossing the Zero Line:

When the CMO crosses above zero, it can be seen as a bullish signal, whereas a cross below zero can be interpreted as bearish.
Customization:

The period over which the CMO is calculated can be adjusted. A shorter period makes the oscillator more sensitive to price changes, suitable for short-term trading. A longer period smooths out the oscillator for a longer-term perspective.
It's important to note that while the CMO can provide valuable insights into market momentum and potential price reversals, it is most effective when used in conjunction with other indicators and analysis methods. Like all technical indicators, the CMO should not be used in isolation but rather as part of a comprehensive trading strategy.

References:
1. https://www.fmlabs.com/reference/default.htm?url=CMO.htm
'''

def _calculate_cmo_pandas(series: pd.Series, period=14):
    # Calculate the difference in closing prices
    delta = series.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate the sum of gains and losses using a rolling window
    sum_gains = gains.rolling(window=period).sum()
    sum_losses = losses.rolling(window=period).sum()

    # Calculate CMO
    cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
    return cmo


def _calculate_cmo_polars(series: pl.Series, period=14):
    # Calculate the difference in closing prices
    delta = series.diff()

    # Separate gains and losses
    gains = delta.apply(lambda x: x if x > 0 else 0)
    losses = delta.apply(lambda x: -x if x < 0 else 0)

    # Calculate the sum of gains and losses using a rolling window
    sum_gains = gains.rolling_sum(window_size=period)
    sum_losses = losses.rolling_sum(window_size=period)

    # Calculate CMO
    cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
    return cmo

