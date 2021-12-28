from src.utils.parameter_loader import (
    load_gridsearch_parameters,
    load_gridsearch_model_parameters,
)
from src.utils.data_utils import load_data, pop_train_test_split
from src.utils.model_viz import summary_of_model
from src.utils.model_utils import (
    find_threshold,
    find_optimal_model
)
import pickle
from src.models.predict import predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import json

def train_model():
    df_scaled = load_data()
    X_train, X_test, y_train, y_test = pop_train_test_split(df_scaled)
    grid_clf_boost = find_optimal_model(X_train, y_train)
    pickle.dump(grid_clf_boost, open("src/models/model.pkl", "wb"))

def make_prediction(input_parameters):
    print("TESTING", predict(input_parameters))
    return predict(input_parameters)

#transformed_example = ss.transform([[6, 148, 72, 0, 33.6, 0.627, 50]])
make_prediction([[6, 148, 72, 0, 25, 0.627, 50]])


#print(model.predict_proba(transformed_example))
