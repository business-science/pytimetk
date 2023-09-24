# Dependencies
import pandas as pd
import numpy as np
import holidays
import math
import pandas_flavor as pf
from typing import Union

@pf.register_dataframe_method
def tk_augment_holiday_signature(
    df: pd.DataFrame,
    date_col: Union[str, pd.Series],
    country_name: str = 'UnitedStates'
) -> pd.DataFrame:
    """
    Engineers three different holiday features from a single datetime for 80+ countries.

    Parameters
    ----------
    df (pd.DataFrame): The input DataFrame.
    date_col (str or pd.Series): The name of the datetime-like column in the DataFrame.
    country_name (str): The name of the country for which to generate holiday features. 
                        Defaults to United States holidays, but the following countries are currently 
                        available and accessible by the full name or ISO code:
            
    Any of the following are acceptable keys for country_name

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
    pd.DataFrame: A pandas DataFrame with three holiday-specific features.

    Example
    -------
    ```{python}
    import pandas as pd
    import timetk as tk
     
    start_date = '2023-01-01'
    end_date = '2023-01-10'
    date_range = pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['date'])
   
    tk_augment_holiday_signature(date_range, 'date', 'France').head()
    ```
    """
    
    # Ensure the date column exists in the DataFrame
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' not found in DataFrame columns.")
    
    # Extract start and end years directly from the Series
    start_year = df[date_col].min().year
    end_year = df[date_col].max().year

    # Create a list of years (integers) from start year to end year
    years = list(range(math.ceil(start_year), math.floor(end_year) + 1))
    
    # Check if valid years were found
    if not years:
        raise ValueError("No valid years found for holiday calculations.")
    
    # Create a DataFrame of the full length of the Series
    date_range = pd.date_range(df[date_col].min(), df[date_col].max())
    holiday_df = pd.DataFrame({'date': date_range})
    
    # Create an empty list to store holidays
    series_holidays = []

    # Retrieve the corresponding country's module using regular expressions
    for key in holidays.__dict__.keys():
        if key.lower() == country_name.lower():
            country_module = holidays.__dict__[key]
            break
    else:
        raise ValueError(f"Country '{country_name}' not found in holidays package.")

    # Append holidays from the selected country to the list
    for date in country_module(years=years).items():
        series_holidays.append(str(date[0]))

    # Add (0, 1) indicator for holiday to the DataFrame
    holiday_df['holiday'] = holiday_df['date'].dt.strftime('%Y-%m-%d').isin(series_holidays).astype(int)

    # Add (0, 1) indicators for day before and day after holiday
    holiday_df['before_holiday'] = holiday_df['holiday'].shift(1).fillna(0).astype(int)
    holiday_df['after_holiday'] = holiday_df['holiday'].shift(-1).fillna(0).astype(int)

    # Merge the two DataFrames on the 'date' column with an outer join
    merged_df = df.merge(holiday_df, left_on=date_col, right_on='date', how='outer')

    # Drop the 'date' column
    merged_df = merged_df.drop(columns=['date'])

    # Fill NaN values in columns 'holiday', 'before_holiday', and 'after_holiday' with 0
    merged_df[['holiday', 'before_holiday', 'after_holiday']] = merged_df[['holiday', 'before_holiday', 'after_holiday']].fillna(0).astype(int)

    return merged_df
