## Model Types
### [Bag of Words](../stats-nlp/#bag-of-words)

!!! abstract "Sample Code"
    [NTLK Corpus](https://www.nltk.org/api/nltk.corpus.html)

    [NTLK Stem](https://www.nltk.org/api/nltk.stem.html)

    [Count Vectorizer](https://scikit-learn.org/1.4/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer)

    ```python
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer   # 'extracts' the root of the words from their variations

    # Clean Data and Build Corpus
    corpus = []
    ps = PorterStemmer()
        
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])  # replace all punctuations by space
        review = review.lower()    # convert to lowercase
        review = review.split()    # split each review into list of words
        
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)   # join back the stemmed words for the review
        corpus.append(review)

    # Build Model
    from sklearn.feature_extraction.text import CountVectorizer
    # max features set to a number lower than the total number of words so that it excludes the sparsely occuring words
    # total number of words can be found by checking the length of X after the fit transform step
    c_cv = CountVectorizer(max_features = 1500)  
    X = c_cv.fit_transform(corpus).toarray()
    y = data.iloc[:, -1].values

    # Train Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Train Model and Predict
    # This example uses Gaussian NB
    # Other classification models can also be used
    from sklearn.naive_bayes import GaussianNB
    c_gnb = GaussianNB()
    c_gnb.fit(X_train, y_train)
    y_pred_gnb = c_gnb.predict(X_test)
    ```