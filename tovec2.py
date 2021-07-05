from gensim.models.doc2vec import Doc2Vec
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, confusion_matrix,average_precision_score,hinge_loss,balanced_accuracy_score, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# model_name = 'standard.model'
# N = 550000

model_name = '50000sample25.model'
N = 50000


model = Doc2Vec.load(model_name)

trainVectors = []
# for review_id in range(len(model.docvecs)):
for review_id in range(N):
    recovered_review = model.docvecs[review_id]
    trainVectors.append(recovered_review)


corpus1 = pd.read_json('b_r_filtered.json',lines = True)
corpus = corpus1.iloc[:N]
X_train, X_test, y_train, y_test = model_selection.train_test_split(trainVectors,corpus['stars'],test_size=0.2)

print('section 1 completed')





print("-------------Run logistic regression---------------")
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(n_jobs=3)#(solver='lbfgs', max_iter=450, n_jobs=-1, verbose=1)
print(logReg.get_params())
logReg.fit(X_train, y_train)
# run_cross_validation_kfold(logReg, X_train, y_train)
y_predicted_lr = logReg.predict(X_test)
print("logistic regression accuracy_score ", accuracy_score(y_test, y_predicted_lr))
print("logistic regression balanced_accuracy_score ", balanced_accuracy_score(y_test, y_predicted_lr))
print("logistic regression confusion_matrix_score\n", confusion_matrix(y_test, y_predicted_lr))
print('logistic linear regression','Mean squared error:',mean_squared_error(y_test, y_predicted_lr))
print('logistic linear regression','mean_absolute_error:',mean_absolute_error(y_test, y_predicted_lr))
print('logistic linear regression r2:',r2_score(y_test, y_predicted_lr))

print("-------------Run linear regression---------------")
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
regr = linear_model.LinearRegression(n_jobs=3)
print(regr.get_params())
regr.fit(X_train,y_train)
y_predicted_regr = regr.predict(X_test)

print('linear regression','Mean squared error: %.2f'% mean_squared_error(y_test, y_predicted_regr))
print('linear regression mean_absolute_error:',mean_absolute_error(y_test, y_predicted_regr))
print('linear regression r2:',r2_score(y_test, y_predicted_regr))

print("-------------Run SVCClassifier---------------")
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svc.fit(X_train, y_train)
y_predicted_sv = svc.predict(X_test)
print("SVCClassifier accuracy_score ", accuracy_score(y_test, y_predicted_sv))
print("SVCClassifier balanced_accuracy_score ", balanced_accuracy_score(y_test, y_predicted_sv))
print("SVCClassifier confusion_matrix_score\n", confusion_matrix(y_test, y_predicted_sv))

print("-------------Run Support Vector Regression---------------")
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import mean_squared_error, r2_score

reglsvc = LinearSVR()
reglsvc.fit(X_train,y_train)
y_predicted_lsvc = reglsvc.predict(X_test)
print('SVR linear regression','Mean squared error:',mean_squared_error(y_test, y_predicted_lsvc))
print('logistic linear regression','mean_absolute_error:',mean_absolute_error(y_test, y_predicted_lsvc))
print('SVR linear regression r2:',r2_score(y_test, y_predicted_lsvc))




# print("-------------Run SGD regression---------------")
# from sklearn.linear_model import SGDRegressor, SGDClassifier
# from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
#
# # SGDR = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# SGDC = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
# SGDR = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
# y_predicted_SGDC = SGDC.fit(X_train, y_train).predict(X_test)
# y_predicted_SGDR = SGDR.fit(X_train, y_train).predict(X_test)
# print("SGDClassifier accuracy_score ", accuracy_score(y_test, y_predicted_SGDC))
# print("SGDClassifier balanced_accuracy_score ", balanced_accuracy_score(y_test, y_predicted_SGDC))
# print("SGDClassifier confusion_matrix_score\n", confusion_matrix(y_test, y_predicted_SGDC))
#
# print('SGD regression', 'Mean squared error:',mean_squared_error(y_test, y_predicted_SGDR))
# print('logistic linear regression','mean_absolute_error:',mean_absolute_error(y_test, y_predicted_SGDR))
# print('SGD regression r2:', r2_score(y_test, y_predicted_SGDR))
#
# cm = confusion_matrix(y_test, y_predicted_SGDC, labels=SGDC.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SGDC.classes_)
# print(disp.plot())
