import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#Loading the Object
with open('x_res_obj.pkl', 'rb') as fv:
    X_res = pickle.load(fv)
with open('y_resobj.pkl', 'rb') as dfv:
    y_res = pickle.load(dfv)


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y =  train_test_split(X_res,y_res,train_size= 0.8 , random_state=5)


#########Gradient Boosting
## Initializing Gradient Boosting with 500 estimators and max depth as 10.
gboost_clf = GradientBoostingClassifier( n_estimators=500, max_depth=10)
## Fitting gradient boosting model to training set
gboost_clf.fit(train_x, train_y )
predicted_classes_gboost = gboost_clf.predict(test_x)
conf_mat_gboost = confusion_matrix(test_y,predicted_classes_gboost)
accuracy_gboost = accuracy_score(test_y,predicted_classes_gboost)
gboost_roc_auc = roc_auc_score(test_y, gboost_clf.predict_proba(test_x)[:,1])
precision_gboost = precision_score(test_y,predicted_classes_gboost)
recall_gboost = recall_score(test_y,predicted_classes_gboost)
F1_gboost = f1_score(test_y,predicted_classes_gboost)


sn.heatmap(conf_mat_gboost,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

print("Confusion matrix (gBoost):")
print(conf_mat_gboost)
print("accuracy score (gBoost)", accuracy_gboost)
print("Precision score (gBoost)",precision_gboost)
print("Recall score (gBoost)", recall_gboost)
print("F1 score (gBoost)", F1_gboost)

#10 fold cross validation - to validate if there is overfitting
gboost_clf = GradientBoostingClassifier( n_estimators=500, max_depth=10)
cv_scores = cross_val_score( gboost_clf, train_x, train_y, cv = 10, scoring = 'roc_auc' )
print( cv_scores )
print( "Mean Accuracy: ", np.mean(cv_scores), " with standard deviation of: ",np.std(cv_scores))
###########################################################################################


################Building the LogR model
model_lr = LogisticRegression(solver='newton-cg', C=100,penalty='l2')
model_lr.fit(train_x,train_y)

test_predicted_classes_lr = model_lr.predict(test_x)

print("Confusion Matrix for LR Model")
test_conf_mat_lr = confusion_matrix(test_y.tolist(),test_predicted_classes_lr)
print(test_conf_mat_lr)
sn.heatmap(test_conf_mat_lr,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

# accuracy_lr = accuracy_score(train_y,predicted_classes_lr)
# print(accuracy_lr)
test_accuracy_lr = accuracy_score(test_y,test_predicted_classes_lr)
precision_lr = precision_score(test_y,test_predicted_classes_lr)
recall_lr = recall_score(test_y,test_predicted_classes_lr)
F1_lr = f1_score(test_y,test_predicted_classes_lr)
print("accuracy score (LR)", test_accuracy_lr)
print("Precision score (LR)",precision_lr)
print("Recall score (LR)", recall_lr)
print("F1 score (LR)", F1_lr)
###########################################################################################



#Buliding the NB model
model_nb = GaussianNB()
model_nb.fit(train_x,train_y)

test_predicted_classes_nb = model_nb.predict(test_x)

print("Confusion Matrix for NB Model")
test_conf_mat_nb = confusion_matrix(test_y.tolist(),test_predicted_classes_nb)
print(test_conf_mat_nb)
sn.heatmap(test_conf_mat_nb,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

# accuracy_nb = accuracy_score(train_y,predicted_classes_nb)
# print(accuracy_nb)

test_accuracy_nb = accuracy_score(test_y,test_predicted_classes_nb)
precision_nb = precision_score(test_y,test_predicted_classes_nb)
recall_nb = recall_score(test_y,test_predicted_classes_nb)
F1_nb = f1_score(test_y,test_predicted_classes_nb)
print("accuracy score (NB)", test_accuracy_nb)
print("Precision score (NB)",precision_nb)
print("Recall score (NB)", recall_nb)
print("F1 score (NB)", F1_nb)
###########################################################################################


############Buliding the KNN model
model_knn = KNeighborsClassifier(n_neighbors=1225)#took the square root of the number of rows if the classes are evn take an odd number
model_knn.fit(train_x,train_y)

test_predicted_classes_knn = model_knn.predict(test_x)

print("Confusion Matrix for KNN Model")
test_conf_mat_knn = confusion_matrix(test_y.tolist(),test_predicted_classes_knn)
print(test_conf_mat_knn)
sn.heatmap(test_conf_mat_knn,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

# accuracy_knn = accuracy_score(train_y,predicted_classes_knn)
# print(accuracy_knn)

test_accuracy_knn = accuracy_score(test_y,test_predicted_classes_knn)
precision_knn = precision_score(test_y,test_predicted_classes_knn)
recall_knn = recall_score(test_y,test_predicted_classes_knn)
F1_knn = f1_score(test_y,test_predicted_classes_knn)
print("accuracy score (KNN)", test_accuracy_knn)
print("Precision score (KNN)",precision_knn)
print("Recall score (KNN)", recall_knn)
print("F1 score (KNN)", F1_knn)
###########################################################################################


#############SVC
model_svm = SVC(kernel = 'rbf', random_state = 0)
model_svm.fit(train_x, train_y)
    
test_predicted_classes_svm = model_svm.predict(test_x)

print("Confusion Matrix for SVM Model")
test_conf_mat_svm = confusion_matrix(test_y.tolist(),test_predicted_classes_svm)
print(test_conf_mat_svm)
sn.heatmap(test_conf_mat_svm,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

test_accuracy_svm = accuracy_score(test_y,test_predicted_classes_svm)
precision_svm = precision_score(test_y,test_predicted_classes_svm)
recall_svm = recall_score(test_y,test_predicted_classes_svm)
F1_svm = f1_score(test_y,test_predicted_classes_knn)
print("accuracy score (SVM)", test_accuracy_svm)
print("Precision score (SVM)",precision_svm)
print("Recall score (SVM)", recall_svm)
print("F1 score (SVM)", F1_svm)
###########################################################################################


################Random Forest
model_rf_regressor = RandomForestClassifier(n_estimators = 100, random_state = 0)
model_rf_regressor.fit(train_x, train_y)

test_predicted_classes_rf = model_rf_regressor.predict(test_x)

print("Confusion Matrix for Random Forest")
test_conf_mat_rf = confusion_matrix(test_y.tolist(),test_predicted_classes_rf)
print(test_conf_mat_rf)
sn.heatmap(test_conf_mat_rf,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

test_accuracy_rf = accuracy_score(test_y,test_predicted_classes_rf)
precision_rf = precision_score(test_y,test_predicted_classes_rf)
recall_rf = recall_score(test_y,test_predicted_classes_rf)
F1_rf = f1_score(test_y,test_predicted_classes_rf)
print("accuracy score (RF)", test_accuracy_rf)
print("Precision score (RF)",precision_rf)
print("Recall score (RF)", recall_rf)
print("F1 score (RF)", F1_rf)
