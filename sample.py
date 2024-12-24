# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle

# %%
df=pd.read_csv("Warm_Up_Predict_Blood_Donations_-_Traning_Data.csv")

# %%
print("Shape of the dataset:", df.shape)
num_rows, num_columns = df.shape
print("Number of rows: " , num_rows)
print("Number of columns: " , num_columns)

# %%
print("Dataset: ")
print(df)

# %%
#To show the data type of the dataset
print("\nData Type")
print(df.dtypes)

# %%
print(df.isnull().sum())

# %%
description = df.describe()
print("\nDescription of the dataset:")
print(description)

# %%
print("\nHead of the dataset:")
print(df.head())

# %%
print("\nTail of the dataset:")
print(df.tail())

# %%
df.dropna(inplace=True)
df

# %%
df.rename(columns={'Made Donation in March 2007': 'Eligible for donation or not'}, inplace=True)
df

# %%
# Replace the column names with correct names
df = df.rename(columns={'Months since Last Donation': 'Months since Last Donation',
                        'Number of Donations       ': 'Number of Donations',
                        'Total Volume Donated (c.c.)': 'Total Volume Donated',
                        'Months since First Donation    ': 'Months since First Donation'})




# %%
# Now you can access the columns using correct names
X = df[['Months since Last Donation', 'Number of Donations', 'Total Volume Donated', 'Months since First Donation']]
y = df['Eligible for donation or not']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %%
# Initialize the SVC model
svc_model = SVC(kernel='linear', random_state=42)

# %%
# Train the SVC model
svc_model.fit(X_train_scaled, y_train)

# %%
# Make predictions
y_pred = svc_model.predict(X_test_scaled)


# %%
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# %%
# Visualize the distribution of the target variable
plt.figure(figsize=(6, 4))
df['Eligible for donation or not'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Eligibility for Donation')
plt.xlabel('Eligibility')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Not Eligible', 'Eligible'], rotation=0)
plt.show()

# %%
#save the model to a file using pickle
with open('svc_model.pkl','wb') as f:
  pickle.dump(svc_model, f)