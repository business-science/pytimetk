# Dependencies
import pandas as pd
import numpy as np
import datetime as dt
import math 
import pandas_flavor as pf
import holidays


@pf.register_dataframe_method
def tk_augment_holiday_signature(x, country = holidays.UnitedStates):
    """Engineers 3 different holiday features from a single datetime for 80+ countries.

    Args:
        x ([datetime64[ns]]): 
            A datatime64[ns] dtype column.
        country (module from holidays package, optional): 
            This parameter must always follow the naming pattern 'holidays.CountryName' 
            Defaults to United States holidays, but the following countries are currently 
            available and accessible by the full name or ISO code:
            
            Any of the Following Are Acceptable Keys for 'CountryName'

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
            Jamaica:                Jamaica,            JM,  JAM,
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
            Nigeria:                Nigeria,            NG,  NGA,
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
            Venezuala:              Venezuala,          YV,   VEN,
            Vietnam:                Vietnam,            VN,   VNM,
            Wales:                  Wales

    Returns:
        DataFrame: A pandas data frame that leverages the date to generate three holiday specific features:
            - holiday: A logical (0,1) feature that captures if the day is an official national holiday
            - before_holiday: A logical (0,1) feature that captures if the day is the day before an official national holiday
            - after_holiday: A logical (0,1) feature that captures if the day is the day after an official national holiday
    """

    # Start Timestamp of Series
    start = pd.to_datetime(x.min())

    # End Time Stamp of the Series
    end = pd.to_datetime(x.max())

    # Start Year of the Series
    start_year = start.year

    # End Year of the Series
    end_year = end.year

    # Create a List of Years (Integers) from start year to end year
    years = list(range(math.ceil(start_year), math.floor(end_year) + 1))

    # Create Dataframe of Full Length of the Series
    df = pd.DataFrame({'Dates':pd.date_range(start, end)})

    # Create Empty Holiday List
    series_holidays = []

    # Append Holidays from Selected Country to List
    for date in country(years=years).items():
        series_holidays.append(str(date[0]))

    # Add Boolean (0,1) Indicator for Holiday to Series
    df['holiday'] = [
        1 if str(val).split()[0] in series_holidays else 0 for val in df['Dates']
        ]

    # Add Boolean (0,1) Indicator Day Before Holiday (Lead)
    df['before_holiday'] = df.holiday.shift(-1)

    # Add Booleaan (0,1) Indicator for Day After Holiday (Lag)
    df['after_holiday'] = df.holiday.shift(1)

    return df