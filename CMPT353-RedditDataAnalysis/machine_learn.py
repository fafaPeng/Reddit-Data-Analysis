import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import FunctionTransformer, make_pipeline
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
import sys  

relative_json_directory = sys.argv[1]
json_directory = os.path.abspath(relative_json_directory)
file_paths = [os.path.join(json_directory, file) for file in os.listdir(json_directory) if file.endswith('.json')]
dataframes = [pd.read_json(file_path, lines=True) for file_path in file_paths]
combined_dataframe = pd.concat(dataframes, ignore_index=True)

print(" Finished Reading File\n");



label_encoder = LabelEncoder()
combined_dataframe['subreddit_encoded'] = label_encoder.fit_transform(combined_dataframe['subreddit'])
combined_dataframe['daytype_encoded'] = label_encoder.fit_transform(combined_dataframe['daytype'])

#sentiment_score   readability_score   subreddit_encoded
X = np.stack([combined_dataframe['readability_score'],combined_dataframe['subreddit_encoded']], axis=1)
y = combined_dataframe['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

hidden_layer_sizes = tuple([4] * 3)
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes))
)
model.fit(X_train,y_train)
model.predict(X_test)
print ("readibility_score: "+str(model.score(X_test,y_test))+"\n")




#sentiment_score   readability_score   subreddit_encoded
X = np.stack([combined_dataframe['sentiment_score'],combined_dataframe['subreddit_encoded']], axis=1)
y = combined_dataframe['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
hidden_layer_sizes = tuple([4] * 3)
model2 = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes))
)
model2.fit(X_train,y_train)

model2.predict(X_test)
print ("sentiment_score: "+str(model2.score(X_test,y_test))+"\n")


#sentiment_score   readability_score   subreddit_encoded
X = np.stack([combined_dataframe['daytype_encoded'],combined_dataframe['subreddit_encoded']], axis=1)
y = combined_dataframe['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
hidden_layer_sizes = tuple([4] * 3)
model3 = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes))
)
model3.fit(X_train,y_train)
model3.predict(X_test)
print ("daytype: "+str(model3.score(X_test,y_test))+"\n")



#sentiment_score   readability_score   subreddit_encoded
X = np.stack([combined_dataframe['readability_score'],combined_dataframe['sentiment_score'],combined_dataframe['daytype_encoded'],combined_dataframe['subreddit_encoded']], axis=1)
y = combined_dataframe['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

hidden_layer_sizes = tuple([4] * 3)
model4 = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes))
)
model4.fit(X_train,y_train)
model4.predict(X_test)
print ("All combined: "+str(model4.score(X_test,y_test))+"\n")





