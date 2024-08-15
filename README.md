# Titanic_EDA
## Titanic Dataset Description
 The dataset includes data about passengers on the Titanic and whether they survived the disaster.

• PassengerId: An integer unique identifier for each passenger.                                
• Survived: An integer indicating whether the passenger survived (1) or not (0).              
• Pclass: An integer representing the passenger's class (1 = 1st class, 2 = 2nd class, 3 = 3rd class).                                                                                        
• Name: A string containing the passenger's full name.                                         
• Sex: A string indicating the passenger's gender (male or female).                            
• Age: A float representing the passenger's age in years.            
• SibSp: An integer indicating the number of siblings or spouses the passenger had aboard the Titanic.                                                                                       
• Parch: An integer representing the number of parents or children the passenger had aboard the Titanic.                                                                                
• Ticket: A string containing the ticket number.                                              
• Fare: A float representing the fare the passenger paid for the ticket.                      
• Cabin: A string indicating the cabin number. Many values are missing (NaN).                
• Embarked: A string indicating the port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).                                                                                 

#### Data Dictionary
• PassengerId: Unique ID for each passenger.                                                  
• Survived: Survival (0 = No; 1 = Yes).                                                       
• Pclass: Ticket class (1 = 1st; 2 = 2nd; 3 = 3rd).                                           
• Name: Name of the passenger.                                                       
• Sex: Sex of the passenger.                   
• Age: Age of the passenger in years.                                   
• SibSp: Number of siblings/spouses aboard the Titanic.                            
• Parch: Number of parents/children aboard the Titanic.                     
• Ticket: Ticket number.                                            
• Fare: Fare paid by the passenger.                                        
• Cabin: Cabin number.                                       
• Embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).              

### Libraires
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
```

### Load and Check Data
```
train_df = pd.read_csv(r'C:\...\train.csv')
test_df = pd.read_csv(r'C:\...\test.csv')
```
Examining the First 5 Columns of the Dataset
```
train_df.head()
```
Then checked the columns
```
train_df.columns
```
The data was described
```
train_df.describe()
```
## Visualizing the Titanic Dataset
We made a function for bar plot visualizing the data.
```
def bar_plot(variable):
    var = train_df[variable]
    varValue= var.value_counts()
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
```
For categorical variables, we made bar plots.
```
categry1 = ["Survived","Sex","Pclass","Embarked","SiSp","Parch"]
for c in categry1:
    bar_plot(c)
```
##### Interpretation of the Survived Bar Plot
A majority of passengers did not survive the disaster. The number of passengers who did not survive is notably higher than the number of passengers who survived.
This plot indicates that survival was less common than non-survival among the Titanic passengers.
##### Interpretation of the Gender Bar Plot
There were more male passengers on the Titanic compared to female passengers. The number of male passengers is almost twice the number of female passengers.
This plot highlights the gender distribution among the passengers, indicating that a larger proportion of the passengers were male.
##### Interpretation of the Class Distribution (Pclass) Bar Plot
A large majority of Titanic passengers traveled in 3rd class, suggesting that the Titanic was a popular choice for lower-income passengers.
The relatively lower numbers of 1st and 2nd class passengers indicate that these classes were more expensive and thus chosen by fewer passengers.
##### Interpretation of the Embarkation Port Distribution (Embarked) Bar Plot
A large majority of Titanic passengers boarded from Southampton (S). This suggests that Southampton was the primary embarkation point for the Titanic.
Fewer passengers boarded from Cherbourg (C) and Queenstown (Q), indicating that these ports were smaller stops with less passenger traffic.
##### Interpretation of the Number of Siblings/Spouses Distribution (SibSp) Bar Plot
Most passengers boarded the Titanic alone or without siblings/spouses.
There is a notable group of passengers traveling with one sibling or spouse, but this number is still relatively small.
The number of passengers traveling with 2 or more siblings/spouses is very low, indicating that large family groups were rare.
##### Interpretation of the Number of Parents/Children Distribution (Parch) Bar Plot
Most passengers boarded the Titanic alone or without parents/children.
There is a notable group of passengers traveling with one parent or child, but this number is still relatively small.
The number of passengers traveling with 2 or more parents/children is very low, indicating that large family groups were rare.

#### We made a histogram plot function for visualizing the data.
```
def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable],bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} disturbution with hist".format(variable))
    plt.show()
```
For numerical variables, we made histogram plots.
```
numericVar= ["Fare","Age","PassengerId"]
for n in numericVar:
    plot_hist(n)
```
#### Fare Distribution
Most passengers paid low fares, which suggests they were likely in the third class. The right-skewness of the fare distribution indicates that there were some wealthier passengers who could afford much higher fares, likely corresponding to first-class accommodations.
#### Age Distribution
The Titanic carried a wide range of passengers of all ages, but young adults (ages 20-30) were the most common age group. This distribution provides insights into the demographics of the passengers, showing that the ship had a significant number of young people, but also included many children and older adults.

### Analyzing Relationships Between Categorical and Numerical Variables
```
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived")
```
First-class passengers had the highest survival rate, while third-class passengers had the lowest. This indicates that passengers traveling in higher classes had a better chance of survival during the Titanic disaster.
```
train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived")
```
Female passengers had a significantly higher survival rate compared to male passengers. This suggests that the "women and children first" policy was in effect during the rescue operations.
```
train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived")
```
Passengers with 1 or 2 siblings/spouses had the highest survival rates, suggesting that having a small family group may have increased chances of survival. However, having more than 2 siblings/spouses aboard significantly decreased the survival rate, possibly due to the difficulty in managing and ensuring the safety of a larger group during the chaos.
```
train_df[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived")
```
Passengers with 1, 2, or 3 parents/children had higher survival rates, indicating that family units of this size had a better chance of staying together and being rescued. Conversely, passengers with 4 or more parents/children had a significantly lower survival rate, which might be due to the increased difficulty of ensuring the safety of larger family groups during the disaster.

### Outlier Detection and Removal
We created a function to detect and remove outliers:
```
def detect_outlier(df, features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile (25th percentile)
        Q1 = np.percentile(df[c], 25)
        # 3rd quartile (75th percentile)
        Q3 = np.percentile(df[c], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
     
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers =  list(i for i, v in outlier_indices.items() if v>2)
    return multiple_outliers
    
    return outlier_indicesOutlier detection
```
We detected the outliers
```
train_df.loc[detect_outlier(train_df,["Age","SibSp","Parch","Fare"])]
```
We merged the datasets
```
train_df_len=len(train_df)
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
```
## Find Missing Value
We found the missing values
```
train_df.columns[train_df.isnull().any()]
```
We visualized the columns with missing values
```
train_df.boxplot(column="Fare",by="Embarked")
plt.show()
```
![image](https://github.com/user-attachments/assets/0e45ee0a-cadd-450c-82db-0794ad63a933)


We filled the missing values
```
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]
```
We checked for missing values
```
train_df[train_df["Fare"].isnull()]
```
We filled the missing fare values
```
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))
```
```
train_df[train_df["Fare"].isnull()]
```
