from preprocessing import load_data, clean_data, split_data
from model_handler import ModelHandler

df = load_data('../data/diabetes_012_health_indicators_BRFSS2015.csv')
df = clean_data(df)
X_train, X_test, y_train, y_test = split_data(df)

model_handler = ModelHandler()
model_handler.train(X_train, y_train)
report, cm = model_handler.evaluate(X_test, y_test)
print(report)
print("Confusion Matrix:\n", cm)

model_handler.save('../models/random_forest_model.pkl', '../models/scaler.pkl')
