import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Function to draw meshgrid for decision boundary visualization
def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array

plt.style.use('seaborn-v0_8-bright')

st.sidebar.markdown("# Bagging Classifier")

# Sidebar inputs
estimators = st.sidebar.selectbox(
    'Select base estimator',
    ('Decision Tree', 'KNN', 'SVM')
)

n_estimators = int(st.sidebar.number_input('Enter number of estimators', min_value=1, value=10))
max_samples = st.sidebar.slider('Max Samples', 1, 375, 375, step=25)
bootstrap_samples = st.sidebar.radio("Bootstrap Samples", ('True', 'False')) == 'True'
max_features = st.sidebar.slider('Max Features', 1, 2, 2, key=1234)
bootstrap_features = st.sidebar.radio("Bootstrap Features", ('False', 'True'), key=2345) == 'True'

# Load initial graph
fig, ax = plt.subplots()
ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):
    if estimators == "Decision Tree":
        estimator = DecisionTreeClassifier()
    elif estimators == "KNN":
        estimator = KNeighborsClassifier()
    else:
        estimator = SVC()

    clf = estimator.fit(X_train, y_train)
    y_pred_tree = clf.predict(X_test)

    bag_clf = BaggingClassifier(
        estimator=estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=bootstrap_samples,
        max_features=max_features,
        bootstrap_features=bootstrap_features,
        random_state=42
    )
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)

    orig.empty()

    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)
    labels1 = bag_clf.predict(input_array)

    col1, col2 = st.columns(2)
    with col1:
        st.header(estimators)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        orig = st.pyplot(fig)
        st.subheader(f"Accuracy for {estimators}: {round(accuracy_score(y_test, y_pred_tree), 2)}")
    with col2:
        st.header("Bagging Classifier")
        ax1.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        ax1.contourf(XX, YY, labels1.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        orig1 = st.pyplot(fig1)
        st.subheader(f"Accuracy for Bagging: {round(accuracy_score(y_test, y_pred), 2)}")