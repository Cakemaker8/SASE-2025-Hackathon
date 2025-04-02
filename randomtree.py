import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Read the dataset
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Make it usable
df = pd.DataFrame(data)

# Fixing BMI with interpolated data
df['bmi'] = df['bmi'].replace(['N/A', 'na', 'None', ''], pd.NA)
df['bmi'] = df['bmi'].interpolate()

# Normalizing the numbers on the dataset using a min max scaler
scaler = MinMaxScaler()
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Separating the stroke column from the rest
X = df.drop(columns=['stroke'])  # Features
X = df.drop(columns=['id'])
y = df['stroke']  # Target variable

# Split the dataset with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

print(f"Total rows in the original dataset: {df.shape[0]}")

# Define a column transformer to preprocess numerical and categorical columns
# Impute missing values for both numerical and categorical columns
# For numerical: use mean imputation
# For categorical: use most frequent value imputation
# Have to do this so that it understands the numbers and words
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
            ('scaler', StandardScaler())  # Standardize numerical features
        ]), ['age', 'hypertension', 'avg_glucose_level', 'bmi']),
        
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # OneHot encode categorical features
        ]), ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
    ])

# Apply preprocessing on the training data and the test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Now apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# Create a Logistic Regression model pipeline
model = Pipeline(steps=[
    ('classifier', RandomForestClassifier(random_state=42))  # Logistic Regression model
])

# Train the model using the resampled training data
model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test_processed)

# Evals
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")

#accuracy = 0.9413
#precision = 0.2500
#f1 = 0.1429
#recall = 0.1