import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


# data reading
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
df = pd.read_excel("sdsp_patients.xlsx", engine='openpyxl')

# data preparation
# fill or drop NaN values
df = df.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace(' ', np.nan)
df = df.loc[:, df.isin(['', np.NaN, np.nan]).mean() < .09]

# Replace all non-numeric feature values with numeric values
df = df.replace('Yes', 1)
df = df.replace('No', 0)
df = df.replace('Male', 1)
df = df.replace('Female', 0)
val_rep = {'Every Day': 0, '1-2 Days a Week': 1, '3-4 Days a Week': 2, '1-2 Days a Month': 3}
df['Feature_28'] = df['Feature_28'].map(val_rep)
val_rep = {'No Difference': 0, 'Evenings': 2, 'Mornings': 1}
df['Feature_29'] = df['Feature_29'].map(val_rep)
val_rep = {'Disease_1': 1, 'Disease_2': 2, 'Disease_3': 3, 'Disease_4': 4}
df['Disease'] = df['Disease'].map(val_rep)
df = df.rename(columns={'Feature_1': 'Gender'})

# fill empty values with column mean
df['Feature_28'] = df['Feature_28'].fillna(round(df['Feature_28'].mean()))
df['Feature_32'] = df['Feature_32'].fillna(round(df['Feature_32'].mean()))
df['Feature_33'] = df['Feature_33'].fillna(round(df['Feature_33'].mean()))
df['Feature_3'] = df['Feature_3'].fillna(round(df['Feature_3'].mean()))
df['Feature_47'] = df['Feature_47'].fillna(round(df['Feature_47'].mean()))
df['Feature_48'] = df['Feature_48'].fillna(round(df['Feature_48'].mean()))
df['Feature_49'] = df['Feature_49'].fillna(round(df['Feature_49'].mean()))
df['Feature_50'] = df['Feature_50'].fillna(round(df['Feature_50'].mean()))

# final form of data
df['Feature_3'] = pd.to_numeric(df['Feature_3'])
df['Feature_47'] = pd.to_numeric(df['Feature_47'])
df['Feature_48'] = pd.to_numeric(df['Feature_48'])
df['Feature_49'] = pd.to_numeric(df['Feature_49'])
df['Feature_50'] = pd.to_numeric(df['Feature_50'])

X = df.iloc[:, 1:49]  # independent columns
y = df.iloc[:, 0]  # target column - disease_number



def make_feature_selection(X, y, k_num):
    fs = SelectKBest(score_func=chi2, k=k_num)
    return fs.fit_transform(X, y)


# functions for feature selection
def ft_slc_eva():
    acc_list = []
    k = [12, 16, 20, 24, 30, 36, 42, 48]
    for i in k:
        KBest_feature_selection = make_feature_selection(X, y, i)
        X_train, X_test, y_train, y_test = train_test_split(KBest_feature_selection, y, test_size=0.2,
                                                            random_state=0)
        rfc = LogisticRegression(solver='lbfgs', max_iter=4000, random_state=0)
        rfc.fit(X_train, y_train)
        rfc_predict = rfc.predict(X_test)
        acc = metrics.accuracy_score(y_test, rfc_predict)
        percentage = acc * 100
        formatted = "{:.2f}".format(percentage)
        acc_list.append(formatted)

    # Plot Feature Number - Accuracy Rate
    acc_list.sort(reverse=True)
    plt.plot(k, acc_list, color='red', marker='o')
    plt.xlabel('Feature Number', fontsize=14)
    plt.ylabel('Accuracy Rate', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()


ft_slc_eva()

# use inbuilt method feature_importances of tree based classifiers
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.20)


# K-BEST FEATURE IMPORTANCE
fs = SelectKBest(score_func=chi2, k=12)
fs.fit(X, y)

selected_features = fs.get_support(indices=True)

new_X = X.iloc[:, selected_features]
new_X['Gender'] = df['Gender'].values
new_y = df['Disease']
clf = LogisticRegression(solver='lbfgs', max_iter=4000, random_state=0).fit(new_X, new_y)
clf.predict(new_X)

# pickle_out = open("classifier.pkl", mode = "wb")
# pickle.dump(clf, pickle_out)
# pickle_out.close()
