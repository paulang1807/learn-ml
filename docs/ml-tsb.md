## Modeling Steps
- Data Preparation - Generate or Read data
    - Ensure that date column is of date datatype
<p id="cust-id-tsb-resampl"></p>
- Resampling or Changing Frequency
    - Downsampling (e.g. Quarterly to Yearly)
    - Upsampling (e.g. Quarterly to Monthly)
        - May result in missing data for lower level rows. Also see [Handle Missing Data]()
    - Set Frequency
        - Set date column as index (if not the index already)
- Handle missing data 
- Feature Engineering
    - Add Basic Date Time features
    - Add Lag features
    - Add Windowing features
    - Add Expanding/Cumulative features
    - Add / Remove other columns as needed
- Explore Data
    - Test for Stationarity
    - Test for Seasonality
        - Apply [Differencing](../stats-tsb/#cust-id-tsb-diff) as needed
    - Test for ACF and PACF
- Split training and test data
    - Walk Forward Validation

*[Resampling]: Changing the frequency of the available data to match the desired forecast frequency

### Data Prep

!!! abstract "Read Data"

    ```python
    # Parse date column as date when reading data
    # dayfirst = True uses the dd/mm/yyyy format instead of the default mm/dd/yyyy; not needed if date is in default format
    # we pass the index of the date column to the parse_dates parameter
    df_ts_base1 = pd.read_csv('data/TimeSeriesData1.csv', dayfirst=True, parse_dates=[0])

    # Alternatively, we can also convert the date column after reading the csv
    # We would do this if for some reason we read the date column without parsing it as a date
    df_ts_base2 = pd.read_csv('data/TimeSeriesData1.csv')
    df_ts_base2.date = pd.to_datetime(df_ts_base2.date, dayfirst = True)
    ```

#### Resampling
| Alias  | Description           |
|--------|-----------------------|
| B      | Business day          |
| D      | Calendar day          |
| W      | Weekly                |
| M      | Month end             |
| Q      | Quarter end           |
| A      | Year end              |
| BA     | Business year end     |
| AS     | Year start            |
| H      | Hourly frequency      |
| T, min | Minutely frequency    |
| S      | Secondly frequency    |
| L, ms  | Millisecond frequency |
| U, us  | Microsecond frequency |
| N, ns  | Nanosecond frequency  |


!!! abstract "Sample Code"
    [Resample](https://pandas.pydata.org/pandas-docs/stable//reference/api/pandas.DataFrame.resample.html#pandas.DataFrame.resample)

    ```python    
    # Downsample
    df_ts1_mth = df_ts1.resample('M', on='date').mean()

    # Upsample
    df_ts1_hr = df_ts1.resample('H', on='date').mean()
    ```

!!! abstract "Set Index"

    ```python
    # Set date column as index if it has not been set as the index based on some prior operation
    df_ts_base1.set_index("date", inplace=True)
    ```

!!! abstract "Set desired frequency"

    ```python
    # This will generate new rows for missing periods for the desired frequency
    # Make sure to set the frequency only after setting the date column as index
    # Parameter 'b' signifies business days
    df_ts_base1 = df_ts_base1.asfreq("b")
    ```

#### Handle missing data
| Method  | Description                                               |
|---------|-----------------------------------------------------------|
| bfill   | Backward fill                                             |
| count   | Count of values                                           |
| ffill   | Forward fill                                              |
| first   | First valid data value                                    |
| last    | Last valid data value                                     |
| max     | Maximum data value                                        |
| mean    | Mean of values in time range                              |
| median  | Median of values in time range                            |
| min     | Minimum data value                                        |
| nunique | Number of unique values                                   |
| ohlc    | Opening value, highest value, lowest value, closing value |
| pad     | Same as forward fill                                      |
| std     | Standard deviation of values                              |
| sum     | Sum of values                                             |
| var     | Variance of values                                        |

!!! abstract "Handle Missing Data"
    [Interpolate](https://pandas.pydata.org/pandas-docs/stable//reference/api/pandas.DataFrame.interpolate.html)

    ```python
    # Front fill NaNs
    df_ts_base1.spx = df_ts_base1.spx.ffill()

    # Back fill NaNs
    df_ts_base1.spx = df_ts_base1.spx.bfill()

    # Populate NaNs using mean value
    df_ts_base1.spx = df_ts_base1.spx.fillna(value=df_ts_base1.spx.mean())

    # Fill in missing values with linear interpolation (euqally spaced values)
    df_ts1_hr_interpolated = df_ts1_hr.interpolate(method='linear')
    ```

### Feature Engineering

!!! abstract "Add Basic Date Columns"
    [Pandas Datetime](https://pandas.pydata.org/pandas-docs/stable//reference/series.html#datetimelike-properties)

    ```python
    # Add additional date columns as needed
    df_ts1['year'] = df_ts1.index.year
    df_ts1['month'] = df_ts1.index.month
    df_ts1['day'] = df_ts1.index.day
    ```

!!! abstract "Add Basic Date Columns"
    [Datetime](https://pandas.pydata.org/pandas-docs/stable//reference/series.html#datetimelike-properties)

    ```python
    # Add additional date columns as needed
    df_ts1['year'] = df_ts1.index.year
    df_ts1['month'] = df_ts1.index.month
    df_ts1['day'] = df_ts1.index.day
    ```

!!! abstract "Lag Features"
    [Shift](https://pandas.pydata.org/pandas-docs/stable//reference/api/pandas.Series.shift.html#pandas.Series.shift)

    ```python
    df_ts1['last_day'] =  df_ts1.spx.shift(1)   # last data point's value
    df_ts1['last_week'] =  df_ts1.spx.shift(7)  # values for current -7th data point
    ```

!!! abstract "Window Features"
    [Rolling](https://pandas.pydata.org/pandas-docs/stable//reference/api/pandas.Series.rolling.html#pandas.Series.rolling)

    ```python
    # Aggreates over specified rolling windows
    df_ts1['2day_mean'] =  df_ts1.spx.rolling(window=2).mean()
    df_ts1['2day_max'] =  df_ts1.spx.rolling(window=2).max()
    ```

!!! abstract "Expanding/Cumulative Features"
    [Expanding](https://pandas.pydata.org/pandas-docs/stable//reference/api/pandas.Series.expanding.html#pandas.Series.expanding)

    ```python
    # Cumulative max
    df_ts1['cum_max'] =  df_ts1.spx.expanding().max()
    ```

### Explore Data

!!! abstract "Test for Stationarity"
    [Stationarity](../stats-tsb/#stationarity)

    [Augmented Dickey Fuller Test](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html)

    ```python
    import statsmodels.tsa.stattools as sts

    adfuller_stats = sts.adfuller(df_ts_base1.market_value)
    adfuller_stats
    ```

!!! abstract "Test for Seasonality"
    [Seasonality](../stats-tsb/#seasonality)

    [Seasonal Decompose](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)

    ```python
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Use model="multiplicative" for testing multiplicative naive decomposition
    s_dec_additive = seasonal_decompose(df_ts_base1.market_value, model = "additive")
    s_dec_additive.plot()
    plt.show()
    ```

!!! abstract "Differencing"
    [Differencing](https://pandas.pydata.org/pandas-docs/stable//reference/api/pandas.Series.diff.html#pandas.Series.diff)

    ```python    
    # Remove Trend
    # Lag 1 Differencing to get rid of trends
    # Uses the difference of current and 'Lag1' values (col - col.shift(1))
    df_ts_base3['diff1'] = df_ts_base3['MilesMM'].diff(periods=1)

    # Remove Seasonality
    # Lag 12 Differencing to get rid of monthly seasonality
    # Uses the difference of current and 'Lag12' values (col - col.shift(12))
    df_ts_base3['diff12'] = df_ts_base3['diff1'].diff(periods=12)
    ```

!!! abstract "Test for Autocorrelation (ACF)"
    [Autocorrelation](../stats-tsb/#autocorrelation)

    [Plot ACF](https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_acf.html)

    ```python
    import statsmodels.graphics.tsaplots as sgt

    # zero = False means that the current period is not considered
    # as the correlation between the current period and itself will always be 1
    # 40 is the optimal number of lags for time series analysis
    sgt.plot_acf(df_ts_base1.market_value, lags = 40, zero = False)
    plt.title("ACF S&P", size = 24)
    plt.show()
    ```

!!! abstract "Test for Partial Autocorrelation (PACF)"
    [Partial Autocorrelation](../stats-tsb/#partial-autocorrelation)

    [Plot PACF](https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_pacf.html)

    ```python
    import statsmodels.graphics.tsaplots as sgt

    # zero = False means that the current period is not considered
    # as the correlation between the current period and itself will always be 1
    # 40 is the optimal number of lags for time series analysis
    sgt.plot_pacf(df_ts_base1.market_value, lags = 40, zero = False, method = ('ols'))
    plt.title("PACF S&P", size = 24)
    plt.show()
    ```