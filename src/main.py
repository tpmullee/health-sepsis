import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Synthetic "sepsis" classification with class imbalance
np.random.seed(0)
N=4000
X = np.random.normal(0,1,(N,6))
w = np.array([0.9,0.6,0.3,0.2,0.1,0.05])
logit = X.dot(w) + np.random.normal(0,0.5,N)
p = 1/(1+np.exp(-logit))
y = (p>0.8).astype(int)  # imbalance
Xtr,Xte,ytr,yte = train_test_split(X,y,stratify=y,test_size=0.2,random_state=7)
clf = LogisticRegression(max_iter=1000).fit(Xtr,ytr)
proba = clf.predict_proba(Xte)[:,1]
print("AUROC:", round(roc_auc_score(yte, proba),3))
