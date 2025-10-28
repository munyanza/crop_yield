Trained the crop yield model using XGBRegressor and produced 98.06% accuracy
Credit to GarvDTU27 for this dataset


# Pandas for data manupulation
# Matplotlib and Seaborn for data virsualization
# Scikit Learn for data preprocessing


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


# Load the data 

url = 'https://raw.githubusercontent.com/GarvDTU27/CropYieldPred/refs/heads/main/crop%20Yield/datafile.csv'
df = pd.read_csv(url)

# Number Rows and columns

df.shape

# Display the columns

print(df.columns)

# Check the information of the dataset

df.info()

# Statistical data of the datasets

df.describe()

# Check the missing values

df.isnull().sum()

# Check the crop columns and all the crops in it

df['Crop'].value_counts()

# Virsualization of the Crop vs the Yield

plt.figure(figsize=(12,10))
sns.barplot(x='Crop',y='Yield (Quintal/ Hectare) ', data=df)
plt.title('Crop vs yield',fontsize=14)
plt.xlabel('Crop', fontsize=14)
plt.ylabel('Yield (Quintal/ Hectare) ',fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Seperate the data into Features (X) and target (y)

X = df.drop(['Yield (Quintal/ Hectare) '],axis=1)
y = df['Yield (Quintal/ Hectare) ']

# Crop the columns into the numerical and categorical columns

numerical_cols = ['Cost of Cultivation (`/Hectare) A2+FL','Cost of Cultivation (`/Hectare) C2','Cost of Production (`/Quintal) C2']
categorical_cols = ['Crop', 'State']

print('Numerical Columns: \n',numerical_cols)
print('Categorical Columns: \n',categorical_cols)

 # StandardScaler standardizes features by removing the mean and scaling to unit variance
 # OneHotEncoder converts categorical features into a one-hot encoded numeric array

numerical_transformer = Pipeline(steps=[
    ('scaler',StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
    ('cat',categorical_transformer,categorical_cols),
    ('num',numerical_transformer,numerical_cols)
])

# Apply the preprocessing steps to the features
X_preprocessed = preprocessor.fit_transform(X)

# Split the preprocessed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed,y,test_size=0.2,random_state=42)

# Train an XGBoost Regressor model and evaluate its performance
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'Accuracy {r2*100:.2f}%')


