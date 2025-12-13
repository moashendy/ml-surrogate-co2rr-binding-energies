import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_rf(X_train, y_train, X_val=None, y_val=None, params=None):
    params = params or {}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    if X_val is not None:
        preds = model.predict(X_val)
        print('Val MAE:', mean_absolute_error(y_val, preds))
    return model

def save_model(model, path):
    joblib.dump(model, path)
