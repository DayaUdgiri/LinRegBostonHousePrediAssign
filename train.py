# necessary Imports
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LassoCV,Lasso, RidgeCV, Ridge, ElasticNetCV, ElasticNet
from sklearn.model_selection import train_test_split
import pickle

# function for Adjusted R Squared score
def adj_r2(y_test, y_pred, X):
    r2=r2_score(y_test,y_pred)
    N=X.shape[0]
    P=X.shape[1]
    return 1-(((1-r2)*(N-1))/(N-P-1))

# loading the data
boston = load_boston()
bos = pd.DataFrame(boston.data)

# fetching features(X) and label(Y)
X = bos
y = boston.target

# scaling the dataset
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# splitting the dataset into Train and Test sets
X_sc_train, X_sc_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.33, random_state=42)

# fitting the model
# LinRegMod_sc = linear_model.LinearRegression()
# LinRegMod_sc.fit(X_sc_train, y_train)

# fitting the model with ElasticNet Regularization
elastic_cv = ElasticNetCV(alphas=None, cv=10)
elastic_cv.fit(X_sc_train, y_train)

print("ElasticCV alpha = ", elastic_cv.alpha_)
print("ElasticCV L1_Ratio = ", elastic_cv.l1_ratio)

Elastic_model = ElasticNet(alpha=elastic_cv.alpha_,l1_ratio=elastic_cv.l1_ratio)
Elastic_model.fit(X_sc_train,y_train)

# Prediction for Test Dataset
y_pred = Elastic_model.predict(X_sc_test)


# Evaluation of the model
print("Linear Regression Model Score is = ", Elastic_model.score(X_sc_test, y_test))
print("R Squared Score is = ", r2_score(y_test, y_pred))
print("Adjusted R Squared Score is = ", adj_r2(y_test, y_pred, X_sc))

# saving the model to the local file system
filename = 'Lin_Reg_final_model.pickle'
pickle.dump(Elastic_model, open(filename, 'wb'))

# prediction using the saved model
loaded_model = pickle.load(open(filename, 'rb'))

for i in range(0,500,100):
    prediction_output = loaded_model.predict(scaler.transform([X.iloc[i]]))
    print("\n *** For input", np.array(X.iloc[i],dtype=str),"\n prediction_output = ",\
          prediction_output, "\n Expected output= ", y[i])


