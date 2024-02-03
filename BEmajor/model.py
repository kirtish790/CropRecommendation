import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv(r'C:\Users\Admin\Downloads\BEmajor\BEmajor\Crop_recommendation.csv')
# Load your dataset and define 'features' and 'target'
features = dataset.drop('label', axis=1)  # Adjust 'target_column_name'
target = dataset['label']

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=2)

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(X_train, Y_train)
RF.score(X_test, Y_test)

Y_pred = RF.predict(X_test)

# Save the trained model
pickle.dump(RF, open('model.pkl', 'wb'))

