import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(filepath):
    """Load dataset from CSV and prepare the features."""
    df = pd.read_csv(filepath)
    
    # Split features and target
    X = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
    y = df['Exited']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_pipeline():
    """Create a pipeline with preprocessing steps and Random Forest classifier."""
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())  # Standardize numeric features
    ])
    
    categorical_features = ['Geography', 'Gender']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Use RandomForestClassifier instead of LogisticRegression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return pipeline

def train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """Train the model and print evaluation metrics."""
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    y_pred = pipeline.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def save_model(pipeline, filename='random_forest_model.joblib'):
    """Save the trained model to a file."""
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', filename)
    joblib.dump(pipeline, model_path)
    print(f"Model saved successfully in {model_path}!")

def main():
    # Load and split the data
    X_train, X_test, y_train, y_test = load_data('.\data\Customer Churn.csv')
    
    # Create and train the model
    pipeline = create_pipeline()
    train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test)
    
    # Save the model
    save_model(pipeline)

if __name__ == "__main__":
    main()
