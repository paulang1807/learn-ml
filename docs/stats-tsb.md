## Basics 

!!! abstract "Terminology"
    - **Market Efficiency**: Measures the level of difficulty in forecasting future values 
    - **Arbitrage**: Buy and sell commodities and make a safe profit while the price adjusts

- [Sample EDA](https://colab.research.google.com/drive/1DBoFlqtpY1jBBz2mZRoac2KlQeNLpT3C)
- Intervals between data points should be identical
    - Encode / Impute missing data / time periods as needed
        - Setting the frequency (`df.asfreq()` method) will add rows corresponding to any missing periods for the desired frequency
        - When handling missing data, it is usually NOT a good idea to fill the missing values with mean values
            - This is appropriate only when all data points for the column fluctuate heavilly around the mean which is rarely the case
            - See [Handle Missing Data](../ml-tsb/#handle-missing-data) for sample code
- We normally use dates as the dataframe indexes for time series data
    - When [resampling](../ml-tsb/#cust-id-tsb-resampl) data, the date column is automatically set as the index column
- In time series analysis, we normally work with one dependent variable at a time
- ==**We cannot shuffle time series data as they need to be in chronological order**==
- If the data has any non linear distribution (quadratic, polynomial, logarithmic, exponential etc.), 
    - converting the values to get a more linear trend can be helpful  
    - helps reducing the noise and bring out underlying trends
        - Various **Power Transformation** techniques can be used in such scenarios
            - ==Used only when there is no trend or seasonality in the series==
            - ==**Moving Average Smoothing**== is the process of creating a new series where the values correspond to the **"moving averages"** of the original values
                - Involves determining the **window width** and **position** (leading, trailing, centered etc.)
                - Centered and Leading positions require a knowledge of future values and are not useful in predictions (since future value is what we are trying to predict)
            - ==**[Exponential Smoothing](../stats-cheatsheet/#exponential-smoothing)**== is the process of creating a new series where the values correspond to the **"weighted averages"** of the original values with larger weight given to the more recent values

            !!! note
                - A **Fast Learner** model is one where the smoothing constant ($\alpha$) is closer to 1

                    - More importance given to the newer values

                - A **Slow Learner** model is one where the smoothing constant ($\alpha$) is closer to 0

                    - More importance given to the older values

    - may result in better forecasting for some of the models

- Some time series models expect **de-trended** and **de-seasonalized** data
    <p id="cust-id-tsb-diff"></p>

    - Use ==**[Differencing](../stats-cheatsheet/#lag-k-differencing)**== to remove the trend and seasonality from the data in such cases
        - Use `Lag 1 Differencing` to get rid of the linear trends
            - For getting rid of the quadratic trends, we apply differencing on the "differenced" series again.
        - Use `Lag 7 Differencing` to get rid of weekly seasonality, `Lag 12 Differencing` to get rid of monthly seasonality and so on
            - This is applied on the "differenced" data that was used to remove trends

- Plot data to understand patterns in data and use corresponding feature engineering techniques to address the patterns
    - For example, use **Lag Scatter plots** to understand if there is any relation between lag periods and current periods
        - Use Lag Features to address these patterns
    - Refer [Sample EDA](https://colab.research.google.com/drive/1DBoFlqtpY1jBBz2mZRoac2KlQeNLpT3C) for more examples
- ==For a good model the residuals will be random (stationary **white noise**) and the residual coefficients will not be significant==
- For train-test split we need to ensure that 
    - Training data is from beginning to a certain cut off point in time
    - Test data starts at the cut off point and continues till the end

### [White Noise](https://colab.research.google.com/drive/10kKCr6cRvFVMWI89cGGLmp98KdvJM1_G#scrollTo=vqSgsmJBlBC5)
- A special type of time series where the data does not follow a pattern
    - Is a random series
    - Hence difficult to model or forecast
- Conditions:
    - Have a constant mean and variance
    - No autocorrelation in any period
        - No clear relation between the past and present values in the time series
- Is [stationary](#stationarity)

### [Random Walk (Drunkard Walk)](https://colab.research.google.com/drive/10kKCr6cRvFVMWI89cGGLmp98KdvJM1_G#scrollTo=mRviui_llBC6)
- A special type of time series where the next value is only dependent on the current value
    - The best esitmator of today's value is yesterday's value
    - The best estimator of tomorrow's value is today's value
    - This process is also referred to as ==**Naive Forecasting**==
- Apart from the dependency on the previous value, the entire series is random

### [Stationarity](https://colab.research.google.com/drive/1u93tB_xbgIrgWNFWrHId_oom7ttz1T01#scrollTo=b8edb6e7-7a2c-4b3c-a09c-cb799cb22a02)
- Implies that taking consecutive sets of data **with the same size** should have **identical covariances** regarless of the starting point
- Also referred as ==**weak-form stationarity**== or ==**covariance stationarity**==
- Assumptions:
    - Have a constant mean and variance
    - Consitent covariance between periods at constant distance from one another
- White Noise is an example of a weak form stationarity 
- ==**Dickey-Fuller Test**== (also called DF Test) can be used to check if data is from a stationary process
    - Null hypothesis is that the data comes from a non-stationary process
    - ==**Reject Null hypothesis if the test statistic is less than the critical value for the desired significance level in the Dickey-Fuller table**==

### [Seasonality](https://colab.research.google.com/drive/1u93tB_xbgIrgWNFWrHId_oom7ttz1T01#scrollTo=b21f9a3d-c11c-43eb-815a-a374d3ec92eb)
- Suggests that certain trends in the data appear on a cyclical basis
- If the data is seasonal, we need to consider factors other than the current period for prediction
- Testing approaches
    - ==**Decomposition (Naive Decomposition)**==
        - Splits the time series into 3 effects:
            - Trend -> Consistent patterns in data
            - Seasonal -> Cyclical patterns
            - Residual/Noise -> Prediction Error or Random Variation
        - Expects a linear relationship between the three effects
        - Uses the previous period values as trend-setter
        - Types:
            - Approaches:
                - **Additive**: For any time period, the observed value is the sum of the three effets
                - **Multiplicative**: For any time period, the observed value is the product of the three effets
            - If the data is seasonal, the resulting plot will show the seasonal pattern

### [Autocorrelation](https://colab.research.google.com/drive/1u93tB_xbgIrgWNFWrHId_oom7ttz1T01#scrollTo=02191b9d-b9bc-488a-8d9f-9d47127665cf)
- Represents the correlation between an observation and a "lagged" version of itself
    - Includes both direct and indirect effects
        - Effect of lag t on current does includes the effects of lags t-1,t-2, etc. on lag t (which may indirectly effect the current data point)
- Autocorrelation in data 
     - at daily frequency checks for correlation between yesterday's and today's data
     - at monthly frequency checks for correlation between last month and current month data
- In the ACF plot, 
    - X Axis represnts the lags
    - Y Axis shows the correlation values
    - The Blue region around the x axis represents significance

!!! note "ACF Interpretation"
    - If the autocorrelation values are higher than the blue region, it suggests that the coefficients are significant indicating that there is time dependence in the data
    - If the values fall withiin the blue area, the coefficients are not significant
    - A sharp drop-off indicates the lag beyond which correlations are not significant

### [Partial Autocorrelation](https://colab.research.google.com/drive/1u93tB_xbgIrgWNFWrHId_oom7ttz1T01#scrollTo=83f87240-e49c-4df1-a15b-b344a1d004c8)
- Measures the correlation between an observation and its lagged values **while adjusting for the effects of intervening observations** 
- Helps identify the direct effect of a specific lagged version on the time series 
    - Removes the influence of other intermediate lags on the concerned lags
        - Effect of lag t on current does not include the effects of lags t-1,t-2, etc. on lag t (which may indirectly effect the current data point)

### Model Selection
!!! tip 
    When comparing models, we should select the one with **Higher Log Likelihood** and **Lower Information Criteria** (**AIC** and **BIC** values)