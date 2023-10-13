# Dependencies
import pandas as pd
import math
import pandas_flavor as pf

try:
    import holidays
except ImportError:
    pass 

from typing import Union

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_series_or_datetime, check_installed


@pf.register_dataframe_method
def augment_holiday_signature(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    country_name: str = 'UnitedStates'
) -> pd.DataFrame:
    """
    Engineers 4 different holiday features from a single datetime for 80+ countries.
    
    Note: Requires the `holidays` package to be installed. See https://pypi.org/project/holidays/ for more information.

    Parameters
    ----------
    data (pd.DataFrame): 
        The input DataFrame.
    date_column (str or pd.Series): 
        The name of the datetime-like column in the DataFrame.
    country_name (str): 
        The name of the country for which to generate holiday features. Defaults to United States holidays, but the following countries are currently available and accessible by the full name or ISO code:
            
        Any of the following are acceptable keys for `country_name`:

            Available Countries:    Full Country, Abrv. #1,   #2,   #3
            Angola:                 Angola,             AO,   AGO, 
            Argentina:              Argentina,          AR,   ARG,
            Aruba:                  Aruba,              AW,   ABW,
            Australia:              Australia,          AU,   AUS, 
            Austria:                Austria,            AT,   AUT, 
            Bangladesh:             Bangladesh,         BD,   BGD,
            Belarus:                Belarus,            BY,   BLR,
            Belgium:                Belgium,            BE,   BEL,
            Botswana:               Botswana,           BW,   BWA,
            Brazil:                 Brazil,             BR,   BRA,
            Bulgaria:               Bulgaria,           BG,   BLG,
            Burundi:                Burundi,            BI,   BDI,
            Canada:                 Canada,             CA,   CAN,
            Chile:                  Chile,              CL,   CHL,
            Colombia:               Colombia,           CO,   COL,
            Croatia:                Croatia,            HR,   HRV,
            Curacao:                Curacao,            CW,   CUW,
            Czechia:                Czechia,            CZ,   CZE,
            Denmark:                Denmark,            DK,   DNK,
            Djibouti:               Djibouti,           DJ,   DJI,
            Dominican Republic:     DominicanRepublic,  DO,   DOM,
            Egypt:                  Egypt,              EG,   EGY,
            England:                England,
            Estonia:                Estonia,            EE,   EST,
            European Central Bank:  EuropeanCentralBank,
            Finland:                Finland,            FI,   FIN,
            France:                 France,             FR,   FRA,
            Georgia:                Georgia,            GE,   GEO,
            Germany:                Germany,            DE,   DEU,
            Greece:                 Greece,             GR,   GRC,
            Honduras:               Honduras,           HN,   HND,
            Hong Kong:              HongKong,           HK,   HKG,
            Hungary:                Hungary,            HU,   HUN,
            Iceland:                Iceland,            IS,   ISL,
            India:                  India,              IN,   IND,
            Ireland:                Ireland,            IE,   IRL,
            Isle Of Man:            IsleOfMan,
            Israel:                 Israel,             IL,   ISR,
            Italy:                  Italy,              IT,   ITA,
            Jamaica:                Jamaica,            JM,   JAM,
            Japan:                  Japan,              JP,   JPN,
            Kenya:                  Kenya,              KE,   KEN,
            Korea:                  Korea,              KR,   KOR,
            Latvia:                 Latvia,             LV,   LVA,
            Lithuania:              Lithuania,          LT,   LTU,
            Luxembourg:             Luxembourg,         LU,   LUX,
            Malaysia:               Malaysia,           MY,   MYS,
            Malawi:                 Malawi,             MW,   MWI,
            Mexico:                 Mexico,             MX,   MEX,
            Morocco:                Morocco,            MA,   MOR,
            Mozambique:             Mozambique,         MZ,   MOZ,
            Netherlands:            Netherlands,        NL,   NLD,
            NewZealand:             NewZealand,         NZ,   NZL,
            Nicaragua:              Nicaragua,          NI,   NIC,
            Nigeria:                Nigeria,            NG,   NGA,
            Northern Ireland:       NorthernIreland,
            Norway:                 Norway,             NO,   NOR,
            Paraguay:               Paraguay,           PY,   PRY,
            Peru:                   Peru,               PE,   PER,
            Poland:                 Poland,             PL,   POL,
            Portugal:               Portugal,           PT,   PRT,
            Portugal Ext:           PortugalExt,        PTE,
            Romania:                Romania,            RO,   ROU,
            Russia:                 Russia,             RU,   RUS,
            Saudi Arabia:           SaudiArabia,        SA,   SAU,
            Scotland:               Scotland,
            Serbia:                 Serbia,             RS,   SRB,
            Singapore:              Singapore,          SG,   SGP,
            Slovokia:               Slovokia,           SK,   SVK,
            Slovenia:               Slovenia,           SI,   SVN,
            South Africa:           SouthAfrica,        ZA,   ZAF,
            Spain:                  Spain,              ES,   ESP,
            Sweden:                 Sweden,             SE,   SWE, 
            Switzerland:            Switzerland,        CH,   CHE,
            Turkey:                 Turkey,             TR,   TUR,
            Ukraine:                Ukraine,            UA,   UKR,
            United Arab Emirates:   UnitedArabEmirates, AE,   ARE,
            United Kingdom:         UnitedKingdom,      GB,   GBR,   UK,
            United States:          UnitedStates,       US,   USA,
            Venezuela:              Venezuela,          YV,   VEN,
            Vietnam:                Vietnam,            VN,   VNM,
            Wales:                  Wales

    Returns
    -------
    pd.DataFrame: 
        A pandas DataFrame with three holiday-specific features:
        - is_holiday: (0, 1) indicator for holiday
        - before_holiday: (0, 1) indicator for day before holiday
        - after_holiday: (0, 1) indicator for day after holiday
        - holiday_name: name of the holiday

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
    """
    # This function requires the holidays package to be installed
    
    # Common checks
    check_installed('holidays')
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
        
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

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_holiday_signature = augment_holiday_signature

@pf.register_series_method
def get_holiday_signature(
    idx: Union[pd.DatetimeIndex, pd.Series],
    country_name: str = 'UnitedStates'
) -> pd.DataFrame:
    """
    Engineers 4 different holiday features from a single datetime for 80+ countries.
    
    Note: Requires the `holidays` package to be installed. See https://pypi.org/project/holidays/ for more information.

    Parameters
    ----------
    idx : pd.DatetimeIndex or pd.Series
        A pandas DatetimeIndex or Series containing the dates for which you want to get the holiday signature.
    country_name (str): 
        The name of the country for which to generate holiday features. Defaults to United States holidays, but the following countries are currently available and accessible by the full name or ISO code:
            
        Any of the following are acceptable keys for `country_name`:

            Available Countries:    Full Country, Abrv. #1,   #2,   #3
            Angola:                 Angola,             AO,   AGO, 
            Argentina:              Argentina,          AR,   ARG,
            Aruba:                  Aruba,              AW,   ABW,
            Australia:              Australia,          AU,   AUS, 
            Austria:                Austria,            AT,   AUT, 
            Bangladesh:             Bangladesh,         BD,   BGD,
            Belarus:                Belarus,            BY,   BLR,
            Belgium:                Belgium,            BE,   BEL,
            Botswana:               Botswana,           BW,   BWA,
            Brazil:                 Brazil,             BR,   BRA,
            Bulgaria:               Bulgaria,           BG,   BLG,
            Burundi:                Burundi,            BI,   BDI,
            Canada:                 Canada,             CA,   CAN,
            Chile:                  Chile,              CL,   CHL,
            Colombia:               Colombia,           CO,   COL,
            Croatia:                Croatia,            HR,   HRV,
            Curacao:                Curacao,            CW,   CUW,
            Czechia:                Czechia,            CZ,   CZE,
            Denmark:                Denmark,            DK,   DNK,
            Djibouti:               Djibouti,           DJ,   DJI,
            Dominican Republic:     DominicanRepublic,  DO,   DOM,
            Egypt:                  Egypt,              EG,   EGY,
            England:                England,
            Estonia:                Estonia,            EE,   EST,
            European Central Bank:  EuropeanCentralBank,
            Finland:                Finland,            FI,   FIN,
            France:                 France,             FR,   FRA,
            Georgia:                Georgia,            GE,   GEO,
            Germany:                Germany,            DE,   DEU,
            Greece:                 Greece,             GR,   GRC,
            Honduras:               Honduras,           HN,   HND,
            Hong Kong:              HongKong,           HK,   HKG,
            Hungary:                Hungary,            HU,   HUN,
            Iceland:                Iceland,            IS,   ISL,
            India:                  India,              IN,   IND,
            Ireland:                Ireland,            IE,   IRL,
            Isle Of Man:            IsleOfMan,
            Israel:                 Israel,             IL,   ISR,
            Italy:                  Italy,              IT,   ITA,
            Jamaica:                Jamaica,            JM,   JAM,
            Japan:                  Japan,              JP,   JPN,
            Kenya:                  Kenya,              KE,   KEN,
            Korea:                  Korea,              KR,   KOR,
            Latvia:                 Latvia,             LV,   LVA,
            Lithuania:              Lithuania,          LT,   LTU,
            Luxembourg:             Luxembourg,         LU,   LUX,
            Malaysia:               Malaysia,           MY,   MYS,
            Malawi:                 Malawi,             MW,   MWI,
            Mexico:                 Mexico,             MX,   MEX,
            Morocco:                Morocco,            MA,   MOR,
            Mozambique:             Mozambique,         MZ,   MOZ,
            Netherlands:            Netherlands,        NL,   NLD,
            NewZealand:             NewZealand,         NZ,   NZL,
            Nicaragua:              Nicaragua,          NI,   NIC,
            Nigeria:                Nigeria,            NG,   NGA,
            Northern Ireland:       NorthernIreland,
            Norway:                 Norway,             NO,   NOR,
            Paraguay:               Paraguay,           PY,   PRY,
            Peru:                   Peru,               PE,   PER,
            Poland:                 Poland,             PL,   POL,
            Portugal:               Portugal,           PT,   PRT,
            Portugal Ext:           PortugalExt,        PTE,
            Romania:                Romania,            RO,   ROU,
            Russia:                 Russia,             RU,   RUS,
            Saudi Arabia:           SaudiArabia,        SA,   SAU,
            Scotland:               Scotland,
            Serbia:                 Serbia,             RS,   SRB,
            Singapore:              Singapore,          SG,   SGP,
            Slovokia:               Slovokia,           SK,   SVK,
            Slovenia:               Slovenia,           SI,   SVN,
            South Africa:           SouthAfrica,        ZA,   ZAF,
            Spain:                  Spain,              ES,   ESP,
            Sweden:                 Sweden,             SE,   SWE, 
            Switzerland:            Switzerland,        CH,   CHE,
            Turkey:                 Turkey,             TR,   TUR,
            Ukraine:                Ukraine,            UA,   UKR,
            United Arab Emirates:   UnitedArabEmirates, AE,   ARE,
            United Kingdom:         UnitedKingdom,      GB,   GBR,   UK,
            United States:          UnitedStates,       US,   USA,
            Venezuela:              Venezuela,          YV,   VEN,
            Vietnam:                Vietnam,            VN,   VNM,
            Wales:                  Wales

    Returns
    -------
    pd.DataFrame: 
        A pandas DataFrame with three holiday-specific features:
        - is_holiday: (0, 1) indicator for holiday
        - before_holiday: (0, 1) indicator for day before holiday
        - after_holiday: (0, 1) indicator for day after holiday
        - holiday_name: name of the holiday

    Example
    -------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    # Make a DataFrame with a date column
    start_date = '2023-01-01'
    end_date = '2023-01-10'
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Get holiday features for US
    tk.get_holiday_signature(dates, 'UnitedStates')
    ```
    
    ```{python}
    # Get holiday features for France
    tk.get_holiday_signature(dates, 'France')
    ```
    
    ```{python}
    # Pandas Series
    pd.Series(dates, name='dates').get_holiday_signature('UnitedStates')
    ```    
    """
    # Common checks
    check_series_or_datetime(idx)
    
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

