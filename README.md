# Naive Bayes Classifier from scratch

Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable. Bayes’ theorem states the following relationship, given class variable  and dependent feature vector $x_{1}$ and $y_{1}$

![image](https://user-images.githubusercontent.com/35486320/192336883-69fd6478-a607-4755-8933-c012e8f1c229.png)

Using the naive conditional independence assumption that

![image](https://user-images.githubusercontent.com/35486320/192337024-93c09aa7-7a27-49aa-8c66-71aeebb31192.png)

for all , this relationship is simplified to

![image](https://user-images.githubusercontent.com/35486320/192337095-39ff6a1d-4179-47ef-9c85-e4127cc0d1ae.png)

## Gaussian Naive Bayes 

`GaussianNB` implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian:

![image](https://user-images.githubusercontent.com/35486320/192337235-7da4a865-73c9-4217-96ed-ecab725cc4d1.png)

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.naive_bayes import GaussianNB
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> gnb = GaussianNB()
    >>> y_pred = gnb.fit(X_train, y_train).predict(X_test)
    >>> print("Number of mislabeled points out of a total %d points : %d"
    ...       % (X_test.shape[0], (y_test != y_pred).sum()))
    Number of mislabeled points out of a total 75 points : 4
    
 
