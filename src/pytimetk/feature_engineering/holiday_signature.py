# Dependencies
import pandas as pd
import numpy as np
import polars as pl
import pandas_flavor as pf
import warnings

try:
    import holidays
except ImportError:
    pass

from typing import Union

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
    check_series_or_datetime,
    check_installed,
)
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    normalize_engine,
    resolve_pandas_groupby_frame,
    restore_output_type,
)


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_holiday_signature(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    country_name: str = "UnitedStates",
    reduce_memory: bool = False,
    engine: str = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Engineers 4 different holiday features from a single datetime for 137 countries
    and 2 financial markets.

    Note: Requires the `holidays` package to be installed. See
          https://pypi.org/project/holidays/ for more information.

    Parameters
    ----------
    data (pd.DataFrame):
        The input DataFrame.
    date_column (str or pd.Series):
        The name of the datetime-like column in the DataFrame.
    country_name (str):
        The name of the country for which to generate holiday features. Defaults
        to United States holidays, but the following countries are currently
        available and accessible by the full name or ISO code: See NOTES.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for
        augmenting holidays. It can be either "pandas" or "polars".

        - The default value is "pandas".

        - When "polars", the function will internally use the `polars` library
          for augmenting holidays. This can be faster than using "pandas" for
          large datasets.

    Returns
    -------
    pd.DataFrame:
        A pandas DataFrame with three holiday-specific features:
        - is_holiday: (0, 1) indicator for holiday
        - before_holiday: (0, 1) indicator for day before holiday
        - after_holiday: (0, 1) indicator for day after holiday
        - holiday_name: name of the holiday

    Notes
    -----

    Any of the following are acceptable keys for `country_name`:

    | Available Countries                       | Full Country                     | Code |
    |:-----------------------------------------:|:--------------------------------:|:----:|
    | Albania                                   | Albania                          | AL   |
    | Algeria                                   | Algeria                          | DZ   |
    | American Samoa                            | AmericanSamoa                    | AS   |
    | Andorra                                   | Andorra                          | AD   |
    | Angola                                    | Angola                           | AO   |
    | Argentina                                 | Argentina                        | AR   |
    | Armenia                                   | Armenia                          | AM   |
    | Aruba                                     | Aruba                            | AW   |
    | Australia                                 | Australia                        | AU   |
    | Austria                                   | Austria                          | AT   |
    | Azerbaijan                                | Azerbaijan                       | AZ   |
    | Bahrain                                   | Bahrain                          | BH   |
    | Bangladesh                                | Bangladesh                       | BD   |
    | Barbados                                  | Barbados                         | BB   |
    | Belarus                                   | Belarus                          | BY   |
    | Belgium                                   | Belgium                          | BE   |
    | Belize                                    | Belize                           | BZ   |
    | Bolivia                                   | Bolivia                          | BO   |
    | Bosnia and Herzegovina                    | BosniaandHerzegovina             | BA   |
    | Botswana                                  | Botswana                         | BW   |
    | Brazil                                    | Brazil                           | BR   |
    | Brunei                                    | Brunei                           | BN   |
    | Bulgaria                                  | Bulgaria                         | BG   |
    | Burkina Faso                              | BurkinaFaso                      | BF   |
    | Burundi                                   | Burundi                          | BI   |
    | Laos                                      | Laos                             | LA   |
    | Latvia                                    | Latvia                           | LV   |
    | Lesotho                                   | Lesotho                          | LS   |
    | Liechtenstein                             | Liechtenstein                    | LI   |
    | Lithuania                                 | Lithuania                        | LT   |
    | Luxembourg                                | Luxembourg                       | LU   |
    | Madagascar                                | Madagascar                       | MG   |
    | Malawi                                    | Malawi                           | MW   |
    | Malaysia                                  | Malaysia                         | MY   |
    | Maldives                                  | Maldives                         | MV   |
    | Malta                                     | Malta                            | MT   |
    | Marshall Islands                          | MarshallIslands                  | MH   |
    | Mexico                                    | Mexico                           | MX   |
    | Moldova                                   | Moldova                          | MD   |
    | Monaco                                    | Monaco                           | MC   |
    | Montenegro                                | Montenegro                       | ME   |
    | Morocco                                   | Morocco                          | MA   |
    | Mozambique                                | Mozambique                       | MZ   |
    | Namibia                                   | Namibia                          | NA   |
    | Netherlands                               | Netherlands                      | NL   |
    | New Zealand                               | NewZealand                       | NZ   |
    | Nicaragua                                 | Nicaragua                        | NI   |
    | Nigeria                                   | Nigeria                          | NG   |
    | Northern Mariana Islands                  | NorthernMarianaIslands           | MP   |
    | North Macedonia                           | NorthMacedonia                   | MK   |
    | Norway                                    | Norway                           | NO   |
    | Pakistan                                  | Pakistan                         | PK   |
    | Panama                                    | Panama                           | PA   |
    | Paraguay                                  | Paraguay                         | PY   |
    | Peru                                      | Peru                             | PE   |
    | Philippines                               | Philippines                      | PH   |
    | Poland                                    | Poland                           | PL   |
    | Portugal                                  | Portugal                         | PT   |
    | Puerto Rico                               | PuertoRico                       | PR   |
    | Romania                                   | Romania                          | RO   |
    | Russia                                    | Russia                           | RU   |
    | San Marino                                | SanMarino                        | SM   |
    | Saudi Arabia                              | SaudiArabia                      | SA   |
    | Serbia                                    | Serbia                           | RS   |
    | Singapore                                 | Singapore                        | SG   |
    | Slovakia                                  | Slovakia                         | SK   |
    | Slovenia                                  | Slovenia                         | SI   |
    | South Africa                              | SouthAfrica                      | ZA   |
    | South Korea                               | SouthKorea                       | KR   |
    | Spain                                     | Spain                            | ES   |
    | Sweden                                    | Sweden                           | SE   |
    | Switzerland                               | Switzerland                      | CH   |
    | Taiwan                                    | Taiwan                           | TW   |
    | Tanzania                                  | Tanzania                         | TZ   |
    | Thailand                                  | Thailand                         | TH   |
    | Tunisia                                   | Tunisia                          | TN   |
    | Turkey                                    | Turkey                           | TR   |
    | Ukraine                                   | Ukraine                          | UA   |
    | United Arab Emirates                      | UnitedArabEmirates               | AE   |
    | United Kingdom                            | UnitedKingdom                    | GB   |
    | United States Minor Outlying Islands      | UnitedStatesMinorOutlyingIslands | UM   |
    | United States of America                  | UnitedStatesofAmerica            | US   |
    | United States Virgin Islands              | UnitedStatesVirginIslands        | VI   |
    | Uruguay                                   | Uruguay                          | UY   |
    | Uzbekistan                                | Uzbekistan                       | UZ   |
    | Vanuatu                                   | Vanuatu                          | VU   |
    | Vatican City                              | VaticanCity                      | VA   |
    | Venezuela                                 | Venezuela                        | VE   |
    | Vietnam                                   | Vietnam                          | VN   |
    | Virgin Islands (U.S.)                     | VirginIslandsUS                  | VI   |
    | Zambia                                    | Zambia                           | ZM   |
    | Zimbabwe                                  | Zimbabwe                         | ZW   |


    These are the Available Financial Markets:

    | Available Financial Markets  | Full Country           | Code |
    |:----------------------------:|:----------------------:|:----:|
    | European Central Bank        | EuropeanCentralBank    | ECB  |
    | New York Stock Exchange      | NewYorkStockExchange   | XNYS |

    Example
    -------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    # Make a DataFrame with a date column
    start_date = '2023-01-01'
    end_date = '2023-01-10'
    df = pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['date'])

    # Add holiday features for US
    tk.augment_holiday_signature(df, 'date', 'UnitedStates')
    ```

    ```{python}
    # Add holiday features for France
    tk.augment_holiday_signature(df, 'date', 'France')
    ```

    ```{python}
    # Add holiday features for France
    tk.augment_holiday_signature(df, 'date', 'France', engine='polars')
    ```
    """
    # This function requires the holidays package to be installed

    # Common checks
    check_installed("holidays")
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)

    engine_resolved = normalize_engine(engine, data)

    if reduce_memory and engine_resolved == "polars":
        warnings.warn(
            "`reduce_memory=True` is only supported for pandas data.",
            RuntimeWarning,
            stacklevel=2,
        )

    conversion: FrameConversion = convert_to_engine(data, "pandas")
    prepared_data = conversion.data

    if isinstance(prepared_data, pd.core.groupby.generic.DataFrameGroupBy):
        base_df = resolve_pandas_groupby_frame(prepared_data).copy()
    else:
        base_df = prepared_data.copy()

    result = _augment_holiday_signature_pandas(
        base_df,
        date_column,
        country_name,
    )

    if reduce_memory and engine_resolved == "pandas":
        result = reduce_memory_usage(result)

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored

    return restored


def _augment_holiday_signature_pandas(
    data: pd.DataFrame,
    date_column: str,
    country_name: str = "UnitedStates",
) -> pd.DataFrame:
    features = _compute_holiday_signature(data, date_column, country_name)
    result = data.copy()
    result[features.columns] = features
    return result


def _compute_holiday_signature(
    data: pd.DataFrame,
    date_column: str,
    country_name: str,
) -> pd.DataFrame:
    columns = ["is_holiday", "before_holiday", "after_holiday", "holiday_name"]
    if data.empty:
        empty = pd.DataFrame(index=data.index, columns=columns)
        empty[["is_holiday", "before_holiday", "after_holiday"]] = 0
        return empty

    dates_normalized = data[date_column].dt.normalize()
    start_date = dates_normalized.min()
    end_date = dates_normalized.max()

    years = list(range(start_date.year, end_date.year + 1))
    if not years:
        raise ValueError("No valid years found for holiday calculations.")

    for key in holidays.__dict__.keys():
        if key.lower() == country_name.lower():
            country_module = holidays.__dict__[key]
            break
    else:
        raise ValueError(f"Country '{country_name}' not found in holidays package.")

    holiday_map = country_module(years=years)
    holiday_dates = [pd.Timestamp(date) for date in holiday_map.keys()]
    holiday_names = {pd.Timestamp(date): name for date, name in holiday_map.items()}

    date_range = pd.date_range(start_date, end_date)
    holiday_data = pd.DataFrame(index=date_range)
    holiday_data["is_holiday"] = holiday_data.index.isin(holiday_dates).astype(int)
    holiday_data["holiday_name"] = holiday_data.index.map(holiday_names.get)
    holiday_data["before_holiday"] = (
        holiday_data["is_holiday"].shift(-1).fillna(0).astype(int)
    )
    holiday_data["after_holiday"] = (
        holiday_data["is_holiday"].shift(1).fillna(0).astype(int)
    )

    selected = holiday_data.loc[dates_normalized.tolist()]
    selected.index = data.index
    return selected[columns]


@pf.register_series_method
def get_holiday_signature(
    idx: Union[pd.DatetimeIndex, pd.Series],
    country_name: str = "UnitedStates",
    engine: str = "auto",
) -> pd.DataFrame:
    """
    Engineers 4 different holiday features from a single datetime for 137 countries
    and 2 financial markets.

    Note: Requires the `holidays` package to be installed. See
    https://pypi.org/project/holidays/ for more information.

    Parameters
    ----------
    idx (Union[pd.DatetimeIndex, pd.Series]):
        The input series.
    country_name (str):
        The name of the country for which to generate holiday features. Defaults
        to United States holidays, but the following countries are currently
        available and accessible by the full name or ISO code: See NOTES.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for
        getting holidays. It can be either "pandas" or "polars".

        - The default value is "pandas".

        - When "polars", the function will internally use the `polars` library
          for getting holidays. This can be faster than using "pandas" for
          large datasets.

    Returns
    -------
    pd.DataFrame:
        A pandas DataFrame with three holiday-specific features:
        - is_holiday: (0, 1) indicator for holiday
        - before_holiday: (0, 1) indicator for day before holiday
        - after_holiday: (0, 1) indicator for day after holiday
        - holiday_name: name of the holiday

    Notes
    -----
    Any of the following are acceptable keys for `country_name`:

    | Available Countries                       | Full Country                     | Code |
    |:-----------------------------------------:|:--------------------------------:|:----:|
    | Albania                                   | Albania                          | AL   |
    | Algeria                                   | Algeria                          | DZ   |
    | American Samoa                            | AmericanSamoa                    | AS   |
    | Andorra                                   | Andorra                          | AD   |
    | Angola                                    | Angola                           | AO   |
    | Argentina                                 | Argentina                        | AR   |
    | Armenia                                   | Armenia                          | AM   |
    | Aruba                                     | Aruba                            | AW   |
    | Australia                                 | Australia                        | AU   |
    | Austria                                   | Austria                          | AT   |
    | Azerbaijan                                | Azerbaijan                       | AZ   |
    | Bahrain                                   | Bahrain                          | BH   |
    | Bangladesh                                | Bangladesh                       | BD   |
    | Barbados                                  | Barbados                         | BB   |
    | Belarus                                   | Belarus                          | BY   |
    | Belgium                                   | Belgium                          | BE   |
    | Belize                                    | Belize                           | BZ   |
    | Bolivia                                   | Bolivia                          | BO   |
    | Bosnia and Herzegovina                    | BosniaandHerzegovina             | BA   |
    | Botswana                                  | Botswana                         | BW   |
    | Brazil                                    | Brazil                           | BR   |
    | Brunei                                    | Brunei                           | BN   |
    | Bulgaria                                  | Bulgaria                         | BG   |
    | Burkina Faso                              | BurkinaFaso                      | BF   |
    | Burundi                                   | Burundi                          | BI   |
    | Laos                                      | Laos                             | LA   |
    | Latvia                                    | Latvia                           | LV   |
    | Lesotho                                   | Lesotho                          | LS   |
    | Liechtenstein                             | Liechtenstein                    | LI   |
    | Lithuania                                 | Lithuania                        | LT   |
    | Luxembourg                                | Luxembourg                       | LU   |
    | Madagascar                                | Madagascar                       | MG   |
    | Malawi                                    | Malawi                           | MW   |
    | Malaysia                                  | Malaysia                         | MY   |
    | Maldives                                  | Maldives                         | MV   |
    | Malta                                     | Malta                            | MT   |
    | Marshall Islands                          | MarshallIslands                  | MH   |
    | Mexico                                    | Mexico                           | MX   |
    | Moldova                                   | Moldova                          | MD   |
    | Monaco                                    | Monaco                           | MC   |
    | Montenegro                                | Montenegro                       | ME   |
    | Morocco                                   | Morocco                          | MA   |
    | Mozambique                                | Mozambique                       | MZ   |
    | Namibia                                   | Namibia                          | NA   |
    | Netherlands                               | Netherlands                      | NL   |
    | New Zealand                               | NewZealand                       | NZ   |
    | Nicaragua                                 | Nicaragua                        | NI   |
    | Nigeria                                   | Nigeria                          | NG   |
    | Northern Mariana Islands                  | NorthernMarianaIslands           | MP   |
    | North Macedonia                           | NorthMacedonia                   | MK   |
    | Norway                                    | Norway                           | NO   |
    | Pakistan                                  | Pakistan                         | PK   |
    | Panama                                    | Panama                           | PA   |
    | Paraguay                                  | Paraguay                         | PY   |
    | Peru                                      | Peru                             | PE   |
    | Philippines                               | Philippines                      | PH   |
    | Poland                                    | Poland                           | PL   |
    | Portugal                                  | Portugal                         | PT   |
    | Puerto Rico                               | PuertoRico                       | PR   |
    | Romania                                   | Romania                          | RO   |
    | Russia                                    | Russia                           | RU   |
    | San Marino                                | SanMarino                        | SM   |
    | Saudi Arabia                              | SaudiArabia                      | SA   |
    | Serbia                                    | Serbia                           | RS   |
    | Singapore                                 | Singapore                        | SG   |
    | Slovakia                                  | Slovakia                         | SK   |
    | Slovenia                                  | Slovenia                         | SI   |
    | South Africa                              | SouthAfrica                      | ZA   |
    | South Korea                               | SouthKorea                       | KR   |
    | Spain                                     | Spain                            | ES   |
    | Sweden                                    | Sweden                           | SE   |
    | Switzerland                               | Switzerland                      | CH   |
    | Taiwan                                    | Taiwan                           | TW   |
    | Tanzania                                  | Tanzania                         | TZ   |
    | Thailand                                  | Thailand                         | TH   |
    | Tunisia                                   | Tunisia                          | TN   |
    | Turkey                                    | Turkey                           | TR   |
    | Ukraine                                   | Ukraine                          | UA   |
    | United Arab Emirates                      | UnitedArabEmirates               | AE   |
    | United Kingdom                            | UnitedKingdom                    | GB   |
    | United States Minor Outlying Islands      | UnitedStatesMinorOutlyingIslands | UM   |
    | United States of America                  | UnitedStatesofAmerica            | US   |
    | United States Virgin Islands              | UnitedStatesVirginIslands        | VI   |
    | Uruguay                                   | Uruguay                          | UY   |
    | Uzbekistan                                | Uzbekistan                       | UZ   |
    | Vanuatu                                   | Vanuatu                          | VU   |
    | Vatican City                              | VaticanCity                      | VA   |
    | Venezuela                                 | Venezuela                        | VE   |
    | Vietnam                                   | Vietnam                          | VN   |
    | Virgin Islands (U.S.)                     | VirginIslandsUS                  | VI   |
    | Zambia                                    | Zambia                           | ZM   |
    | Zimbabwe                                  | Zimbabwe                         | ZW   |


    These are the Available Financial Markets:

    | Available Financial Markets  | Full Country           | Code |
    |:----------------------------:|:----------------------:|:----:|
    | European Central Bank        | EuropeanCentralBank    | ECB  |
    | New York Stock Exchange      | NewYorkStockExchange   | XNYS |

    Example
    -------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    # Make a DataFrame with a date column
    start_date = '2023-01-01'
    end_date = '2023-01-10'
    df = pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['date'])

    # Add holiday features for US
    tk.get_holiday_signature(df['date'], 'UnitedStates')
    ```

    ```{python}
    # Add holiday features for France
    tk.get_holiday_signature(df['date'], 'France')
    ```
    """
    check_installed("holidays")
    check_series_or_datetime(idx)

    engine_normalised = (engine or "").strip().lower()
    if engine_normalised in ("", "auto"):
        engine_normalised = "pandas"

    if engine_normalised not in ("pandas", "polars"):
        raise ValueError("Invalid engine. Use 'pandas', 'polars', or 'auto'.")

    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")

    series_name = idx.name or "idx"
    idx = idx.rename(series_name)

    features = _compute_holiday_signature(idx.to_frame(), series_name, country_name)

    result = idx.to_frame()
    result[features.columns] = features

    if engine_normalised == "polars":
        return pl.from_pandas(result)

    return result

