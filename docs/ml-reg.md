## [Simple Linear Regression](../stats-reg/#simple-linear-regression-model)

!!! abstract "SkLearn API"

    [Linear Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

    **Sample Code**
    ```python
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_train, y_train)
    intercept = lr.intercept_
    coefficient = lr.coef_
    y_pred = lr.predict(X_test)
    ```

## [Multiple Linear Regression](../stats-reg/#multiple-linear-regression-model)

!!! danger "Remember"
    For ML models, the categorical variables need to be transformed to dummy variables (using one hot encoding, for example). However, care should be taken not to include all the resulting dummy variables for multiple regression models, as it will result in **multicollinearity**. This is also referred to as the ==**Dummy Variable Trap**==. 

    In order to avoid this, we should always omit one dummy variable corresponding to each categorical variable.

    ==In the python code, the scikit learn Multiple Regression class automatically takes care of this, so we don't have to omit the dummy variables explicitly==

!!! tip
    ==In Multiple Linear Regression, there is no need to apply Feature Scaling.== 
    
    The coefficients for the independent variables will put everything on the same scale.

!!! abstract "Backward Elimination using Statsmodel"

    [Statsmodel](https://www.statsmodels.org/stable/index.html)

    **Sample Code**
    ```python
    import statsmodels.api as sm
    # Statsmodel does not take into account the constant (intercept).
    # So we need to add it in the form b0x0 where x0 is an array of 1s

    # Here X is the array after one hot encoding the independent variable
    X = X[:, 1:]   # Avoiding the Dummy Variable Trap
    X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
    regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
    regressor_OLS.summary()
    ```

## [Polynomial Regression](../stats-reg/#polynomial-regression-model)

!!! abstract "SkLearn API"

    [Polynomial Features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)

    **Sample Code**
    ```python
    from sklearn.preprocessing import PolynomialFeatures
    num_features = 4
    pf4 = PolynomialFeatures(degree=num_features, include_bias=False)
    X4 = pf4.fit_transform(X)
    lr4 = LinearRegression()
    lr4.fit(X4, y)
    print (lr4.intercept_)
    print (lr4.coef_)
    y_pred = lr4.predict(X4)
    ```