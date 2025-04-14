# save_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier()
model.fit(X, y)

with open(r"D:\CPE\mlops\mldeployment-cpe393\app\model.pkl", "wb") as f:
    pickle.dump(model, f)
