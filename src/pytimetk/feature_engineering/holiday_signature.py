# Dependencies
import pandas as pd
import numpy as np
import polars as pl
import math
import pandas_flavor as pf

try:
    import holidays
except ImportError:
    pass 

from typing import Union

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_series_or_datetime, check_installed
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_dataframe_method
def augment_holiday_signature(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    country_name: str = 'UnitedStates',
    reduce_memory: bool = False,
    engine: str = 'pandas',
) -> pd.DataFrame:
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
    check_installed('holidays')
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    
    if reduce_memory:
        data = reduce_memory_usage(data)
        
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)

    if engine == 'pandas':
        ret = _augment_holiday_signature_pandas(data, date_column, country_name)
    elif engine == 'polars':
        ret = _augment_holiday_signature_polars(data, date_column, country_name)
        # Polars Index to Match Pandas
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
            
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
        
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_holiday_signature = augment_holiday_signature

def _augment_holiday_signature_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    country_name: str = 'UnitedStates'
) -> pd.DataFrame:

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # Data is a GroupBy object, use apply to get a DataFrame
        pandas_df = data.obj.copy()
    elif isinstance(data, pd.DataFrame):
        # Data is already a DataFrame
        pandas_df = data
     
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = data.obj

    
    # Extract start and end years directly from the Series
    start_year = data[date_column].min().year
    end_year = data[date_column].max().year

    # Create a list of years (integers) from start year to end year
    years = list(range(math.ceil(start_year), math.floor(end_year) + 1))
    
    # Check if valid years were found
    if not years:
        raise ValueError("No valid years found for holiday calculations.")
    
    # Create a DataFrame of the full length of the Series
    date_range = pd.date_range(data[date_column].min(), data[date_column].max())
    holiday_data = pd.DataFrame({'date': date_range})

    # Retrieve the corresponding country's module using regular expressions
    for key in holidays.__dict__.keys():
        if key.lower() == country_name.lower():
            country_module = holidays.__dict__[key]
            break
    else:
        raise ValueError(f"Country '{country_name}' not found in holidays package.")

    # Create an empty list to store holidays
    series_holidays = []
    series_holidays_names = []
    
    # Append holidays from the selected country to the list
    for date in country_module(years=years).items():
        series_holidays.append(str(date[0]))
        series_holidays_names.append(date[1])

    holidays_lookup_df = pd.DataFrame({'date': series_holidays, 'holiday_name': series_holidays_names})
    
    # Add (0, 1) indicator for holiday to the DataFrame
    holiday_data['is_holiday'] = holiday_data['date'].dt.strftime('%Y-%m-%d').isin(series_holidays).astype(int)
    
    holiday_data['date'] = pd.to_datetime(holiday_data['date'])
    holidays_lookup_df['date'] = pd.to_datetime(holidays_lookup_df['date'])  
    
    holiday_data = pd.merge(holiday_data, holidays_lookup_df, on='date', how='left')

    # Add (0, 1) indicators for day before and day after holiday
    holiday_data['before_holiday'] = holiday_data['is_holiday'].shift(-1).fillna(0).astype(int)
    holiday_data['after_holiday'] = holiday_data['is_holiday'].shift(1).fillna(0).astype(int)

    # Merge the two DataFrames on the 'date' column with an outer join
    merged_data = data.merge(holiday_data, left_on=date_column, right_on='date', how='outer')

    # Drop the 'date' column
    merged_data = merged_data.drop(columns=['date'])

    # Fill NaN values in columns 'holiday', 'before_holiday', and 'after_holiday' with 0
    merged_data[['is_holiday', 'before_holiday', 'after_holiday']] = merged_data[['is_holiday', 'before_holiday', 'after_holiday']].fillna(0).astype(int)

    merged_data.index = data.index
    
    ret = pd.concat([data, merged_data[['is_holiday', 'before_holiday', 'after_holiday', 'holiday_name']]], axis=1)
    
    return ret

def _augment_holiday_signature_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    country_name: str = 'UnitedStates'
) -> pd.DataFrame:
     
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # Data is a GroupBy object, use apply to get a DataFrame
        pandas_df = data.obj.copy()
    elif isinstance(data, pd.DataFrame):
        # Data is already a DataFrame
        pandas_df = data
     
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = data.obj

    # Convert to Polars DataFrame
    df_pl = pl.DataFrame(data)

    # Get Start and End Dates
    start_date = df_pl.sort(date_column).row(0)[0]
    end_date   = df_pl.sort(date_column).row(-1)[0]

    # Get List of Holidays (0/1)
    start         = pl.date(start_date.year, start_date.month, start_date.day)
    end           = pl.date(end_date.year, end_date.month, end_date.day)
    holidays_list = list(holidays.country_holidays(country_name, years=[start_date.year,end_date.year]))

    # Create Expression 
    expr = pl.date_range(start, end)

    # Non-Holiday Expression
    expr_non = expr.filter(~expr.is_in(holidays_list))
    non_holidays = pl.DataFrame(pl.select(expr_non).to_series())
    non_holidays = non_holidays.with_columns(pl.lit(0).alias("is_holiday"))

    # Holiday Expression
    expr_is  = expr.filter(expr.is_in(holidays_list))
    is_holidays = pl.DataFrame(pl.select(expr_is).to_series())
    is_holidays = is_holidays.with_columns(pl.lit(1).alias("is_holiday"))

    # Join
    df = pl.concat((non_holidays, is_holidays), how="diagonal").sort('date')

    # shift
    df = df.with_columns(
        pl.col('is_holiday').shift(-1).alias('before_holiday').fill_null(0),
        pl.col('is_holiday').shift(1).alias('after_holiday').fill_null(0)
        )
    df = df.with_columns(pl.col(["is_holiday","before_holiday","after_holiday"]).cast(pl.Int8))

    # Add holiday name
    dict_holidays = (holidays.country_holidays(country_name, years=[start_date.year,end_date.year]))
    holiday_name_list = [{"date": str(key), "holiday_name": value} for key, value in dict_holidays.items()]

    # Create a DataFrame from the list of dictionaries
    df_holidays = pl.DataFrame(holiday_name_list).with_columns(
        pl.col("date").str.strptime(pl.Date),
        pl.col("holiday_name").cast(pl.Categorical)
        )

    # join
    df = pl.concat((df, df_holidays), how="align")
    df = (df.filter(df['date'] <= end_date)).to_pandas().fillna(value=np.nan) 

    return df

@pf.register_series_method
def get_holiday_signature(
    idx: Union[pd.DatetimeIndex, pd.Series],
    country_name: str = 'UnitedStates',
    engine: str = 'pandas',
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
    # Common checks
    check_series_or_datetime(idx)
    
    if engine == 'pandas':
        return _get_holiday_signature_pandas(idx, country_name)
    elif engine == 'polars':
        return _get_holiday_signature_polars(idx, country_name)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

def _get_holiday_signature_pandas(
    idx: Union[pd.DatetimeIndex, pd.Series],
    country_name: str = 'UnitedStates'
) -> pd.DataFrame:
    
    # This function requires the holidays package to be installed
    try:
        import holidays
    except ImportError:
        raise ImportError("The 'holidays' package is not installed. Please install it by running 'pip install holidays'.")
    
    df = pd.DataFrame(idx)
    
    if df.columns[0] == 0:
        df.columns = ['idx']
    
    ret = df.pipe(augment_holiday_signature, date_column = df.columns[0], country_name = country_name)

    return ret

def _get_holiday_signature_polars(
    idx: Union[pd.DatetimeIndex, pd.Series],
    country_name: str = 'UnitedStates'
) -> pd.DataFrame:
    
    # This function requires the holidays package to be installed
    try:
        import holidays
    except ImportError:
        raise ImportError("The 'holidays' package is not installed. Please install it by running 'pip install holidays'.")
    
    df = pd.DataFrame(idx)
    
    if df.columns[0] == 0:
        df.columns = ['idx']
    
    ret = augment_holiday_signature(df, date_column = df.columns[0], country_name = country_name, engine='polars')

    return ret
