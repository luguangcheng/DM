from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import pipeline
from sklearn.decomposition import PCA

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

pipe_xgb = Pipeline([('scl',StandardScaler()),('pca',PCA(n_components =6)),('clf',XGBClassifier(silent = 1,objective='binary:logistic'))])
param_eta =[0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_depth = [1,2,3,4,5,6,7,8,9,10]
param_subsimple = [0.5,0.6,0.7,0.8,0.9,1.0]
param_colsample = [0.5,0.6,0.7,0.8,0.9,1.0]

param_xgb ={{'pca__n_components':range(1,200)},{'clf__max_depth':param_depth,'clf__eta':param_eta,'clf__subsimple':param_subsimple,'clf__colsample':param_colsample}}

gs_xgb GridSearchCV(estimator = pipe_xgb,param_grid = param_xgb,refit=True,scoring = 'accuracy',cv=10,n_jobs=-1)

gs_xgb.fit(X_train,y_train)

print gs_xgb.best_score_
print gs_xgb.best_estimator_
clf_xgb = gs_xgb.best_estimator_
clf_xgb.fit(X_train,y_train)
print 'Train Accuracy: %0.3f'%clf_xgb.score(X_train,y_train)
print 'Test Accuracy:%0.3f' %clf_xgb.score(X_test,y_test)


