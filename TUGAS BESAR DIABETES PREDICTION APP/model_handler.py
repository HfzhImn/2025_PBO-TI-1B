import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

class ModelHandler:
    def __init__(self):
        self.model = None
        self.scaler = None

    def train(self, X_train, y_train):
        # Scaling fitur numerik
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Kombinasi SMOTE + RandomUnderSampler
        smote = SMOTE(random_state=42)
        undersample = RandomUnderSampler(random_state=42)
        pipeline = Pipeline([
            ('smote', smote),
            ('undersample', undersample)
        ])
        X_res, y_res = pipeline.fit_resample(X_train_scaled, y_train)
        
        # Training model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_res, y_res)

    def evaluate(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        return report, cm

    def save(self, model_path='../models/random_forest_model.pkl', scaler_path='../models/scaler.pkl'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
    
    def load(self, model_path='models/random_forest_model.pkl', scaler_path='models/scaler.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, data_df):
        X_scaled = self.scaler.transform(data_df)
        return self.model.predict(X_scaled), self.model.predict_proba(X_scaled)
