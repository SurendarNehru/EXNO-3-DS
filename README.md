## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

import pandas as pd

df=pd.read_csv("/content/Encoding Data.csv")
df
![image](https://github.com/user-attachments/assets/9b44162c-b92c-4513-ba09-8d251a5b25b2)


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

![image](https://github.com/user-attachments/assets/a745d59b-40bc-41ef-a151-70cc6c99a961)


le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc
![image](https://github.com/user-attachments/assets/fde4ebdc-8316-4aea-aa66-6574159b8828)

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse_output=False)

df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1)

df2
![image](https://github.com/user-attachments/assets/763c895b-c0f5-40ce-a43d-74134087aa23)


pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

df=pd.read_csv("/content/data.csv")

![image](https://github.com/user-attachments/assets/3f91e084-d1bc-4671-9269-1ccac4ada3cf)


from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"])

cc=pd.concat([cc,new],axis=1)

![image](https://github.com/user-attachments/assets/a0db75e5-8cd2-4dbd-9442-096cc7fc81f8)

![image](https://github.com/user-attachments/assets/9e7e4523-0b23-442d-9d9b-87ff83377a4f)

np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Positive Skew"])

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])

df
![image](https://github.com/user-attachments/assets/5f7e4c7b-5386-4ab5-968a-6f70b9d507a6)


df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])

df.skew()
![image](https://github.com/user-attachments/assets/1c40def2-96bd-4906-9c45-e0cf6f9dfd24)


df["Highly Positive Skew_yeojhnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

df.skew()

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
![image](https://github.com/user-attachments/assets/3a471b05-3d2d-47f1-a983-9614688deaff)


from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/df0fc849-7221-4dd9-af18-a30a984ce91d)


df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])

sm.qqplot(df["Highly Negative Skew"],line='45')

plt.show()

![image](https://github.com/user-attachments/assets/d5632510-0b02-4cf0-962b-da57384f76dc)




# RESULT:

       Feature Encoding and Transformation process is successfully done for the given Datasets.

       
