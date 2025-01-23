import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, confusion_matrix

data = pd.read_csv('./Data/recipe_site_traffic_2212.csv')


nutrient_cols = ['calories', 'carbohydrate', 'sugar', 'protein']
missing_data = data[data[nutrient_cols].isna().all(axis=1)]
print('Number of remaing rows when dropping rows where all nutrients are missing: ',
       len(data)-len(missing_data))
data.dropna(subset=nutrient_cols, inplace=True)


data['category'] = data['category'].str.replace('Chicken Breast', 'Chicken')
data['servings'] = data['servings'].str.replace(r'\D+', '', regex=True)
data['servings'] = data['servings'].astype('int')
data['high_traffic'] = np.where(data['high_traffic']=='High', 1, 0)
data.set_index('recipe', inplace=True)

X = data.drop(columns='high_traffic')
y = data['high_traffic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1, stratify=y)

numerical_cols = nutrient_cols
categorical_cols = ['servings', 'category']

numerical_transformer = FunctionTransformer(np.log1p)
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
], remainder='passthrough')

pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline_lr.fit(X_train, y_train)
y_lr = pipeline_lr.predict(X_test)
precision = precision_score(y_test, y_lr)
cm = confusion_matrix(y_test, y_lr)
TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP)
print('Logistic Regression Precision: ', precision)
print('Logistic Regression Specificity: ', specificity)