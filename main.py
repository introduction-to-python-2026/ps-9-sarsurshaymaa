import pandas as pd

# Load the dataset
df = pd.read_csv('parkinsons.csv')
df = df.dropna()
df.head()

"""## 2. *Select features:*  

   - Choose *two features* as inputs for the model.  
   - Identify *one feature* to use as the output for the model.  

  #### Advice:  
  - You can refer to the paper available in the GitHub repository for insights into the dataset and guidance on identifying key features for the input and output.  
  - Alternatively, consider creating pair plots or using other EDA methods we learned in the last lecture to explore the relationships between features and determine which ones are most relevant.  

"""

import plotly.express as px
# Select two features as inputs (X)
x = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
# Identify one feature as the output (Y)
y = df['status']
# Create a scatter plot to visualize the selected features, colored by status
fig = px.scatter(df, x='MDVP:Fo(Hz)', y='MDVP:Jitter(%)', color='status')
fig.show()

"""## 3. *Scale the data:*

   Apply the MinMaxScaler to scale the two input columns to a range between 0 and 1.  

"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_x = scaler.fit_transform(x)

"""## 4. *Split the data:*

   Divide the dataset into a training set and a validation set.




"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2)

"""## 5. *Choose a model:*  

   Select a model to train on the data.  

   #### Advice:  
   - Consider using the model discussed in the paper from the GitHub repository as a reference.  

"""

from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

"""# 6. *Test the accuracy:*  

   Evaluate the model's accuracy on the test set. Ensure that the accuracy is at least *0.8*.  

"""

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

"""## 7. *Save and upload the model:*  

   After you are happy with your results, save the model with the .joblib extension and upload it to your GitHub repository main folder.
   
   Additionally, update the config.yaml file with the list of selected features and the model's joblib file name.  


example:  
yaml
selected_features: ["A", "B"]  
path: "my_model.joblib"  

"""
