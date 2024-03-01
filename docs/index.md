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
- This usually pushes the encoded columns to the front of the array

**Encoding the Categorical data when order does not matter**
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
**Encoding the Categorical data when order matters (e.g. small,medium,large etc.)**
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
    - Models where there is an implicit relation between the dependent and independent variables (e.g. Support Vector Regression Model)
- Done in order to avoid some features dominating the other features
- Feature scaling is always applied to columns
- ==Should not be applied to encoded columns==
    - Will result in loss of interpretation (of the original categories) if applied 

!!! danger "Remember"
    Feature Scaling should always be done after splitting the training and test data. The test data should be clean and not a part of the feature scaling process.

#### Standardization
- This will result in all the features taking values between -3 and 3
- Works all the time irrespective of the distribution of the features
$$ x_{stand} = \frac{x - mean(x)}{standard \ deviation (x)} $$

#### Normalization
- This will result in all the features taking values between 0 and 1
- Recommended when the distribution for most of the features are normalized
$$ x_{norm} = \frac{x - min(x)}{max(x) - min (x)} $$

!!! abstract "SkLearn API"
    [Standard Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

    **Sample Code**
    ```python
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()

    # When some independent variables have been encoded
    # Scale only the non-encoded colummns
    # Since the encoded columns are present in the front of the array, we usually just take everything from the index of the first non encoded numerical column
    # In the below code, 'n' is the number of resulting encoded columns after encoding
    X_train[:, n:] = sc.fit_transform(X_train[:, n:])

    # When no independent variables have been encoded
    # Apply feature scaling to all independent variables
    # Also apply to dependent variables, if needed
    X_train = sc.fit_transform(X_train)
    ```

### [Model Evaluation](./ml-cheatsheet/#accuracy-and-error-rates)
#### Regression Models
!!! abstract "SkLearn API"
    [R2 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)

    **Sample Code**
    ```python
    from sklearn.metrics import r2_score
    r2_score(y_test, y_pred)
    ```

#### Classification Models
##### Confusion Matrix
Displays a matrix of four categories based on the actual and predicted labels

- True positive : actual = 1, predicted = 1
- False positive : actual = 0, predicted = 1
- False negative : actual = 1, predicted = 0
- True negative : actual = 0, predicted = 0

Also see [Type I and Type II Errors](./stats-hypo-test/#type-i-and-type-ii-errors)



|             | Predicted Negative    | Predicted Positive  |
| :---------- | :-------------------: |:----------------: |
| Actual Negative | TN  | FP  |
| Actual Positive | FN | TP|

!!! abstract "SkLearn API"
    [Confusion Matrix](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)

    **Sample Code**
    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, y_pred)
    ```

##### Accuracy
Fractions of samples predicted correctly

!!! abstract "SkLearn API"
    [Accuracy Score](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)

    **Sample Code**
    ```python
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, y_pred)
    ```

##### Recall
Fractions of positive events that are predicted correctly

!!! abstract "SkLearn API"
    [Recall Score](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)

    **Sample Code**
    ```python
    from sklearn.metrics import recall_score
    recall_score(y_test, y_pred)
    ```

##### Precision
Fractions of positive events that are actually positive

!!! abstract "SkLearn API"
    [Precision Score](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)

    **Sample Code**
    ```python
    from sklearn.metrics import precision_score
    precision_score(y_test, y_pred)
    ```

##### F1 Score
Harmonic mean of recall and precision

- The higher the score the better the model

!!! abstract "SkLearn API"
    [F1 Score](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)

    **Sample Code**
    ```python
    from sklearn.metrics import f1_score
    f1_score(y_test, y_pred)
    ```

##### ROC Curve and ROC AUC Score
- Help with understanding the balance between true positive rate and false positive rates
    - The area under curve metric helps to analyze the performance
- Inputs to these functions are the actual labels and the predicted probabilities (not the predicted labels)
- ROC stands for Receiver Operating Characteristic
- The roc curve function returns three lists:
    - thresholds: all unique prediction probabilities in descending order
    - fpr: the false positive rate (FP / (FP + TN)) for each threshold
    - tpr: the true positive rate (TP / (TP + FN)) for each threshold

!!! abstract "SkLearn API"
    [ROC Curve](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)

    **Sample Code**
    ```python
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    ```

    [ROC AUC Score](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)

    **Sample Code**
    ```python
    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_test, y_pred_prob)
    ```

## Parameter Tuning
- Used for tuning hyperparameters (parameters which are not learnt by the model)