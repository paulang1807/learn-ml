# Machine Learning Home

## Modeling Steps
- Data Preparation - Generate or Read data
- Separate the independent and dependent variables
- Handle missing data (Needed if there is missing data)
- Encode Categorical Data (Needed if there is categorical data)
- Split training and test data
- Feature Scaling (Needed only for some models)

### Handle missing data
Common Approach - Replace missing value with average of all values in the column

!!! abstract "SkLearn API"

    [Simple Imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

    **Sample Code**
    ```python
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 1:3])  # Include all numeric columns
    X[:, 1:3] = imputer.transform(X[:, 1:3])
    ```

### Encode Categorical Data
- Encoding the Independent Variable
    - This usually pushes the encoded columns to the front of the array
!!! abstract "SkLearn API"
    [Column Transformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) 

    [One Hot Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

    **Sample Code**
    ```python
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    ```
- Encoding the Dependent Variable
!!! abstract "SkLearn API"
    [Label Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

    **Sample Code**
    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    ```

### Split training and test data

!!! abstract "SkLearn API"
    [Train Test Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

    **Sample Code**
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    ```

### Feature Scaling
- Used in some ML models(not all) 
- Done in order to avoid some features dominating the other features
- Feature scaling is always applied to columns
- ==Should not be applied to encoded columns==
    - Will result in loss of interpretation (of the original categories) if applied 

!!! danger "Remember"
    Feature Scaling should always be done after splitting the training and test data. The test data should be clean and not a part of the feature scaling process.

Common Forms:

- Standardization
    - This will result in all the features taking values between -3 and 3
    - Works all the time irrespective of the distribution of the features
$$ x_{stand} = \frac{x - mean(x)}{standard \ deviation (x)} $$
- Normalization
    - This will result in all the features taking values between 0 and 1
    - Recommended when the distribution for most of the features are normalized
$$ x_{norm} = \frac{x - min(x)}{max(x) - min (x)} $$

!!! abstract "SkLearn API"
    [Standard Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

    **Sample Code**
    ```python
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    # Scale only the non-encoded colummns
    # Since the encoded columns are present in the front of the array, we usually just take everything from the index of the first non encoded numerical column
    X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
    ```