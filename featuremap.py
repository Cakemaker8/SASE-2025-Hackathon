import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the dataset
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Make it usable
df = pd.DataFrame(data)

# Fixing BMI with interpolated data
df['bmi'] = df['bmi'].replace(['N/A', 'na', 'None', ''], pd.NA)
df['bmi'] = df['bmi'].interpolate()


df = df.drop(columns=['id'])

# Select numerical columns (including the target 'stroke' column if it's numeric)
# Ensure that 'stroke' column is included in the correlation if you want to analyze the correlation to stroke occurrence
numerical_features = df.select_dtypes(include=[np.number]).columns

# Calculate the correlation matrix
correlation_matrix = df[numerical_features].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))  # Set the figure size for better visualization
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True, square=True)

# Set labels and title
plt.title("Correlation Heatmap of Health Factors and Stroke Occurrence")
plt.show()