## Model Types
### [Logistic Regression](../stats-cls/#logistic-regression)

- Predict a categorical dependent variable from a set of independent variables
- It is a linear classifier
- Requires Feature Scaling
- Uses probabilistic approach 
    - Useful for ranking predictions by their probability
    - Provides info on statistical significance of features

!!! abstract "SkLearn API"
    [Logistic Regression](https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)

    **Sample Code**
    ```python
    from sklearn.linear_model import LogisticRegression
    c_lr = LogisticRegression(random_state = 0)
    c_lr.fit(X_train, y_train)
    y_pred = c_lr.predict(sc.transform(X_test))
    ```

### [K-Nearest Neighbor](../stats-cls/#k-nearest-neighbor)

- Predict which category a random point falls in
- Not a linear classifier
- Requires Feature Scaling

!!! abstract "SkLearn API"
    [K Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

    **Sample Code**
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    c_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    c_knn.fit(X_train, y_train)
    y_pred = c_knn.predict(sc.transform(X_test))
    ```

### [Support Vector Machine (SVM)](../stats-cls/#support-vector-machine-svm)

- Works with both linear and non linear problems
- Requires Feature Scaling
- Not preferred for large number of features
- Linear SVM 
    - Not preferred for non linear problems
- Kernel SVM
    - High performance on non linear problems
    - Not biased by outliers
    - Not sensitive to overfitting
- Good for segmentation use cases

!!! abstract "SkLearn API"
    [Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)

    **Sample Code**
    ```python
    from sklearn.svm import SVC
    c_svc_rbf = SVC(kernel = 'rbf', random_state=0)   # rbf kernel
    c_svc_rbf.fit(X_train, y_train)
    # Predict
    y_pred_rbf = c_svc_rbf.predict(sc.transform(X_test))
    ```

### [Naive Bayes](../stats-cls/#naive-bayes)

- Works best when there are two independent variables
- Not a linear classifier
- Not biased by outliers
- Requires Feature Scaling
- Uses probabilistic approach
    - Useful for ranking predictions by their probability

!!! abstract "SkLearn API"
    [Gaussian Naive Bayes Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)

    **Sample Code**
    ```python
    from sklearn.naive_bayes import GaussianNB
    c_gnb = GaussianNB()
    c_gnb.fit(X_train, y_train)
    # Predict
    y_pred_gnb = c_gnb.predict(sc.transform(X_test))
    ```

### [Decision Tree Classification](../stats-cls/#decision-tree-classification)

- Works with both linear and non linear problems
- Preferred for better interpretability
- Feature scaling not needed
- Not good with small datasets
    - May result in overfitting

!!! abstract "SkLearn API"
    [Decision Tree Classifier](https://scikit-learn.org/1.4/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

    **Sample Code**
    ```python
    from sklearn.tree import DecisionTreeClassifier
    c_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    c_dt.fit(X_train, y_train)
    # Predict
    y_pred_dt = c_dt.predict(sc.transform(X_test))
    ```

### [Random Forest Classification](../stats-cls/#decision-tree-classification)

- Works with both linear and non linear problems
- Also see [Random Forest Regression](../ml-reg/#random-forest-regression)

!!! abstract "SkLearn API"
    [Decision Tree Classifier](https://scikit-learn.org/1.4/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

    **Sample Code**
    ```python
    from sklearn.ensemble import RandomForestClassifier
    c_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    c_rf.fit(X_train, y_train)
    # Predict
    y_pred_dt = c_rf.predict(sc.transform(X_test))
    ```