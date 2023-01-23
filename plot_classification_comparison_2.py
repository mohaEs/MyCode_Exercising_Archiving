"""

downloaded from sklearn website
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

modified by me for Farzan's paper
=========================================================
Comparing different clustering algorithms on toy datasets
=========================================================

 comparison of a several classifiers in scikit-learn on synthetic datasets. 
 The point of this example is to illustrate the nature of decision boundaries 
 of different classifiers. This should be taken with a grain of salt, 
 as the intuition conveyed by these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
 linearly and the simplicity of classifiers such as naive Bayes and linear SVMs 
 might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colorsand testing points semi-transparent. 
The lower right shows the classification accuracy on the test set.
"""
print(__doc__)

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression


np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), np.random.randint(2, size=n_samples)

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

colors = np.array([x for x in 'cbgmyrkbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

plt.figure(figsize=(17, 9.5))
plt.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

# plot_num = 1

# labels=[]
# for i in ['D1','D2','D3','D4','D5']:
#     for j in ['m1','m2','m3','m4','m5','m6','m7','m8']:        
#         text=i+j
#         labels.append(text) 
# counter = -1


names = [
    "Logistic Regression",
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    LogisticRegression(penalty='l2', random_state=0),
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
        

datasets = [noisy_circles, noisy_moons, aniso, blobs,  no_structure]

figure = plt.figure(figsize=(33,15))
i=1


cm_bright_2class = ListedColormap(["g", "b"])
cm_bright_3class = ListedColormap(["c", 'g', "b"])
# cm = plt.cm.RdBu
cm = plt.cm.GnBu

for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    
    if np.max(y)==1:
        cm_bright = cm_bright_2class
    elif np.max(y)==2:
        cm_bright = cm_bright_3class
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if i_dataset == 0:
        ax.set_title("Input data (training)")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    
    # Plot the testing points
    # ax.scatter(
    #     X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    # )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5, 
            #response_method = 'predict',
        )
        
        y_pred_test= clf.predict(X_test)
        # display = DecisionBoundaryDisplay(
        #     xx0=feature_1, xx1=feature_2, response=y_pred)
        # display.plot()

        # Plot the training points
        
        # ax.scatter(
        #     X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        # )
        
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if i_dataset == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
figure.savefig('fig_example_classes.pdf', dpi=300)
plt.show()
