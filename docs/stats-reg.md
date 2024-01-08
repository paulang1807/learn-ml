## Ordinary Least Squares Regression
### [Simple Linear Regression Model](../stats-cheatsheet/#simple-linear-regression-model)

**Regression is the process of estimating the value of the dependent variable from the independent variable(s).**

The most common form of the Regression Model is the ==**Simple Linear Regression Model**==

- Also called ==**two-variable linear regression model**== or ==**bivariate linear regression model**==

It is represented as the equation of the line that best shows the trend in the data. The line is referred to as the ==**regression line**==.

- Also called ==**line of best fit**== or the ==**least squares line**== (see [Sum of Squared Residuals](#sum-of-squared-residuals))

!!! success "Expression"
    $$y=\beta_0 + \beta_1 x + u$$

    where 

    - $y$ is the dependent variable (==also called explained, outcome, predicted or response variable or just regressand==),
    - $\beta_0$ is the intercept,
    - $\beta_1$ is the slope parameter,
    - $x$ is the independent variable (also called control, predictor or explanatory variable or just regressor or covariate) and
    - $u$ is the error term or disturbance and includes the effect of all other factors aside from $x$
        - It is the difference between the actual value of a data point and the predicted value (value of the point on the regression line) and is also called the ==**residual**== for that data point
    
    - $\beta_0 + \beta_1 x$ is referred to as the ==**systematic part** of $y$==, the part that can be expalined by $x$  
    - $u$ is the ==**unsystematic part** of $y$==, the part that cannot be expalined by $x$

If the regression line has a

- positive slope, the data has a **positive linear relationship**
- negative slope, the data has a **negative linear relationship**

Also see [Correlation Basics Tip](../stats-basics/#cust-id-base-corr-tip)

If the data is clustered

- tightly around the regression line, it shows a **strong relationship**
- loosely around the regression line, it shows a **moderate or weak relationship** depending on the spread of the data points
    - the more outliers there are in the data, the weaker the relationship

!!! info
    In statistics, we usually write the slope and intercept at least to four decimal places to prevent severe rounding errors and getting a more accurate regression line

!!! note "[Error Assumptions](../stats-cheatsheet/#errors)"
    - Errors have zero means, $E(u) = 0$
    - Average value of $u$ does not depend on value of $x$, i.e. $u$ is **mean independent** of x. This is referred to as the ==**zero conditional mean assumption**== , $E(u|x)=E(u) \implies E(u|x) = 0$

!!! abstract "[Residual Properties](../stats-cheatsheet/#cust-id-cs-reg-res-prop)"
    - Estimated errors sum up to zero, $\sum_{i=1}^n \hat u_i = 0$
    - For any regression line, the mean of residuals is always zero, $\overline {\hat u} = 0$
    - Correlation between residuals and regressors is zero, $\sum_{i=1}^n x_i \hat u_i = 0$
    - Sample averages of y and x (point $\overline x, \overline y$) lie on a regression line, $\overline y = \hat \beta_0 + \hat \beta_1 \overline x$

#### [Sum of Squared Residuals](../stats-cheatsheet/#sum-of-squared-residuals)
To find the best fitting regression line that shows the trend in the data, we want to minimize all the residual values, because doing so would minimize the distance of the data points from the line of best fit. 

In order to minimize the residual, we actually minimize the squae of the residuals so that the positive and negative values of the residuals do not cancel out and all residuals get minimized.

#### [Measures of Variation](../stats-cheatsheet/#measures-of-variation)
##### Coefficient of Determination, $r^2$
Measures the percentage of error we eliminated by using least-squares regression instead of just mean,$\overline y$. See [Correlation](../stats-cheatsheet/#correlation)

- Tells us how well the regression line approximates the data
!!! tip
    - Expressed as percentage
        - 100% describes a line that is a perfect fit to data
        - The higher the value, the more data points pass through the line
        - Very small values indicate that the regression line does not pass through many data points

##### [Root Mean Square Error (RMSE)](../stats-cheatsheet/#root-mean-square-error)
The standard deviation of the residuals

- Also called the ==**Root Mean Square Deviation(RMSD)**==
!!! tip
    - Can be thought of as lines representing the standard deviation of the residuals
        - Parallel to the regression line
        - Distance between the lines represents the fit
            - The lesser the distance, the better the fit
    - The concept of normal distribution probability based on standard deviations(see [here](../stats-distributions/#cust-id-dst-ndist-tip)) can be applied here
        - ==68.2% of the data points will fall between 1 * RMSE of the regression line==
        - ==95.4% of the data points will fall between 2 * RMSE of the regression line==
        - ==99.7% of the data points will fall between 3 * RMSE of the regression line==

## [Chi Square Tests](../stats-cheatsheet/#chi-square-tests)
Helps investigate the relationship betweem categorical variables. Refer [Chi Square Distribution Table](https://www.math.arizona.edu/~jwatkins/chi-square-table.pdf)

!!! abstract "Conditions for Inference"
    - **Random**
        - Sample should be random
    - **Large Counts**
        - Each [expected value](#cust-id-reg-exp-val) should be 5 or greater
    - **Independent**
        - Sample with replacement AND/OR
        - $n \leq \frac{N}{10}$ (Sample size less than 10% of total population - ==10% Rule==)
    - **Categorical**
        - variables should be categorical

Three types:

- $\chi^2$ test for homogeneity
    - Whether the probability distributions of **two separate groups** are homogenous (similar) with respect to some characteristic
        - The values corresonding to a homogenous distribution are referred to as the ==<b id="cust-id-reg-exp-val">expected values</b>==
        - ==The larger the value of $\chi^2$, the more likely that the variables affect each other and are not homogenous==
        - Compare the calculated $\chi^2$ against the chi square table value based on the desired $\alpha$ and degree of freedom to test the null hypothesis
        - ==Reject null hypothesis if $\chi^2 \gt \chi_\alpha^2$==
- $\chi^2$ test for association/independence
    - Whether two variables in the **same group** are related
- $\chi^2$ test for goodness of fit
    - Whether data (in a single data table) fits a specified distribution