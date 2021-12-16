#!/usr/bin/env python
# coding: utf-8

# &emsp;
# &emsp;
# <center><b><font style="color: skyblue"  size="10">Obesity Data Set Analysis</font></b></center>
# <center><b><font style="color: skyblue" size="8">___________</font></b></center>
# &ensp;
# <center><b><font style="color: darkblue" size="5">Biomedical Engineering</font></b></center>
# &ensp;
# <center><b><font style="color: darkblue" size="3">Amanda Román Román</font></b></center>
# &nbsp;
# <center><b><font style="color: darkblue" size="3">Sara Valiente Jaén</font></b></center>
# 
# <center><b><font style="color: darkblue" size="3">________________________________</font></b></center>
# 
# 

# <center><b><font style="color: darkblue" size="3">_________________________________________________________________________</font></b></center>

# In this practice we will handle a challenging target as it is the elaboration and a deeper understanding of different supervised algorithms (Regression and Classification).
# 
# In order to address the challenges stated in this practice we must to select beforehand a proper database. Concretely we have selected [Estimation of obesity levels based on eating habits and physical condition Data Set](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+#), whose aim is to classify the survey respondents into different weight ranges taking into account several parameters.
# 
# After this brief overview, now it will be provided the syllabus for our study. 

# <center><b><font style="color: darkblue" size="3">_________________________________________________________________________</font></b></center>

# &ensp;

# <center><b><font style="color: skyblue" size="8">___________</font></b></center>
# &ensp;
# 
# 
# <center><b><font style="color: skyblue" size="4">SYLLABUS</font></b></center> 
# <center><b><font style="color: skyblue" size="8">___________</font></b></center>
# 

#  <b><p  style="color:skyblue;"
#       > 0-MODULES IMPORTATION
#     </p></b>

#  <b><p  style="color:skyblue;"
#       >1-LAB 1: PRE-PROCESSING DATA </p></b>

#  <b><p  style="color:skyblue;"
#       >2-LAB 2: PARAMETRIC REGRESSION </p></b>

#  <b><p  style="color:skyblue;"
#       >3-LAB 3: PARAMETRIC CLASSIFICATION </p></b>

#  <center><b><font style="color: skyblue" size="8">___________________________</font></b></center>
# &ensp;
# 
# <center><b><font style="color: skyblue" size="6">0- MODULES IMPORTATION </font></b></center>
# 
# 
#   <center><b><font style="color: skyblue" size="8">___________________________</font></b></center>
# &ensp;

# Before explaining the lab's practices we must import the modules and the functions needed for a optimal performance of the coding.
# 
#     
#     
#     

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold,GridSearchCV,cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PolynomialFeatures
from sklearn import model_selection, metrics  
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,recall_score,f1_score


# In[2]:


def bar_plot(x,y,title,xlab=None):
    plt.figure()
    plt.title(title)
    plt.bar(x,y,color='blue')

    for rect in plt.bar(x,y):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')

    plt.xticks(x, xlab, rotation='horizontal')
    plt.tight_layout()
    plt.show()
    
    return None


# In[3]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


#  <center><b><font style="color: skyblue" size="8">___________________________</font></b></center>
# &ensp;
# 
# # <center><b><font style="color: skyblue" size="6">1- LAB1 DATA PRE-PROCESSING </font></b></center>
# 
# 
#   <center><b><font style="color: skyblue" size="8">___________________________</font></b></center>
# &ensp;

# 
# ## <center><b><font style="color: darkblue" size="5"> 1.1-DATA PRESENTATION</font></b></center>
# 
# 
# <center><div class="alert alert-block alert-info">
# Description
# </div></center>
# 
# This dataset was obtained through a **survey** in which individuals from Mexico, Peru and Colombia were asked about their eating habits and physical condition.
#     
# According to the obtained information, each person was assigned a corresponding obesity degree (Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III).
# 
# <blank></blank>
# 
# <center><div class="alert alert-block alert-info">
# Parameters
# </div></center>
# 
# - Gender
# - Age
# - Height
# - Weight
# - Family history with overweight
# - FAVC: Frequent consumption of high caloric food 
# - FCVC: Frequency of consumption of vegetables
# - NCP: Number of main meals
# - CAEC: Consumption of food between meals
# - SMOKE: Whether the subject is a smoker
# - CH20: Consumption of water daily
# - SCC: Calories consumption monitoring
# - FAF: Physical activity frequency
# - TUE: Time using technology devices
# - CALC: Consumption of alcohol
# - MTRANS: Transportation used
# - NObesity: Classification according to weight, being the categories Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III.
# 
# The following parameters are related with eating habits: FAVC, FCVC, NCP, CAEC, CH20, CALC.
# 
# The following variables are related with the physical condition: SCC, FAF, TUE, MTRANS.
# 
# <center><div class="alert alert-block alert-info">
# Imbalance issue
# </div></center>
# 
# We will consider whether any imbalance issue is to be addressed in the following section, Data Visualization.
# 
# <blank></blank>
# 
# <center><div class="alert alert-block alert-info">
# Data preview
# </div></center>
# 

# In[4]:


data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
data


# <center><div class="alert alert-block alert-info">
# Some data insights
# </div></center>

# In[5]:


data.dtypes


# Even though many variables appear as "float64", we should keep in mind that that information is not always reliable. In our database's case, most variables shown as float are actually just integers.

# In[6]:


data.columns


# In[7]:


data.head()


# <center> <div class="alert alert-block alert-info">
# Renaming columns
# </div></center>
# 
# Now we rename the columns in order to facilitate us the understanding of the acronyms

# In[8]:



data.columns = ['Gender','Age','Height','Weight','Family history with overweight',
                'FAVC: Frequent consumption of high caloric food','FCVC: Frequency of consumption of vegetables',
                'NCP: Number of main meals','CAEC: Consumption of food between meals',
                'SMOKE','CH2O','SCC: Calories consumption monitoring','FAF: Physical activity frequency',
                'TUE: Time using technology devices','CALC: Consumption of alcohol',
                'MTRANS','NObeyesdad']


# The following table visualization is just for us to make sure that our column's names have been correctly changed.

# In[9]:


data


# In[10]:


data['CAEC: Consumption of food between meals'].unique()


# In[11]:


data.describe()


# In[12]:


(data == 0).sum(axis=0)


# <center><div class="alert alert-block alert-info">
# Renaming categories, creating variables 'Outcome' and 'BMI'
# </div></center>

# In[13]:


np.unique(data['MTRANS'].values)
np.unique(data['NObeyesdad'].values)


# Since the variables 'MTRANS' and 'NObeyesdad' have very long values, such as 'Public_Transportation' or 'Overweight_Level_II', we will transform them to **'MTRANS_short' and 'NObeyesdad_short'**. They will contain the same information, but the values will be renamed so that they are shorter and do not overimpose in the visualizations' labels.

# In[14]:


MTRANS_short = pd.Categorical(data['MTRANS'].values).rename_categories(
    {'Automobile':'Automobile','Bike':'Bike', 'Motorbike':'Motorbike', 'Public_Transportation':'PBT',
       'Walking':'Walk'})

NObeyesdad_short = pd.Categorical(data['NObeyesdad'].values).rename_categories(
    {'Insufficient_Weight':'Insufficient', 'Normal_Weight':'Normal', 'Obesity_Type_I':'OB 1',
       'Obesity_Type_II':'OB 2', 'Obesity_Type_III':'OB 3', 'Overweight_Level_I':'OVW 1',
       'Overweight_Level_II':'OVW 2'}).reorder_categories(new_categories = ('Insufficient','Normal','OVW 1', 'OVW 2', 'OB 1', 'OB 2', 'OB 3'), ordered = True)

#Outcome = NObeyesdad_short.rename_categories({'Insufficient':0,'Normal':0,'OVW 1':0, 'OVW 2':0, 'OB 1':1, 'OB 2':1, 'OB 3':1})


data.insert(15,'MTRANS_short', MTRANS_short)
data.insert(16,'NObeyesdad_short', NObeyesdad_short)
data = data.drop(['MTRANS','NObeyesdad'],axis=1)


# Up until now, the subjects fell into one of six obesity categories, which made our problem a multi-class one. Since we are not to deal with this kind of complex problems, we will create a **new variable that will group several obesity types into a binary classification**:
# 
# - Obese (1): 'Obesity_Type_I' ('OB 1'), 'Obesity_Type_II' ('OB 2'), 'Obesity_Type_III' ('OB 3')
# - Non-obese (0): 'Insufficient_Weight' ('Insufficient'), 'Normal_Weight' ('Normal'), 'Overweight_Level_I' ('OVW 1'),'Overweight_Level_II'('OVW 2')

# In[15]:


m1 = (data['NObeyesdad_short']=='OB 1')
m2 = (data['NObeyesdad_short']=='OB 2')
m3 = (data['NObeyesdad_short']=='OB 3')

Outcome = np.where(m1 | m2 | m3, 1, 0)
data.insert(17, 'Outcome', Outcome)


# In[16]:


data[data['NObeyesdad_short'] =='OB 1']


# In[17]:


BMI = data["Weight"]/(data["Height"]**2) 

#una vez que ya has creado bmi quitas peso y altura:
#standard_data=standard_data.drop("Weight", axis=1)
#standard_data=standard_data.drop("Height", axis=1)

data.insert(2, 'BMI', BMI)


# In[18]:


data['BMI'][(data["BMI"] !=0) & (data["BMI"] != 1)]


# In[19]:


variables = list(data.columns)
#print(variables, len(variables))


# <center> <div class="alert alert-block alert-info">
# Unique values of our parameters
# </div> </center>
# 
# Now, we are going to visualize the unique values of our features and also count its number of appeareance. This is good for us in the case of the categorical variables, due to in this way we can know how many classes the feature have.
# That is why we use the restriction "if len(unique_values)<10:" in the below code, due to if we do not use it the numerical values could have hundred of unique values and printing them requires time, so, we prefer to avoid it.

# In[20]:


for i in range(data.columns.shape[0]):
    name = data.columns[i]
    
    unique_values= data[str(name)].unique()
    if len(unique_values)<10:
        print("Feature: ",name)
        #print(unique_values)
        count_value=data[name].value_counts()
        print(count_value)
        print(" ")


# ## <center><b><font style="color: darkblue" size="5"> 1.2-DATA VISUALIZATION</font></b></center>
# 
# 
# In this section we visualize graphically all the data through different approaches as histograms, scatter plots and graphs and combining different features.
# 
# <blank></blank>
# 
# <center><div class="alert alert-block alert-info">
# Histograms
# </div></center>

# In[21]:



plt.figure(figsize=[15,20])

restriction = ['Outcome']

for i in variables:
    
    if i not in restriction:
    
        plt.subplot(6, 3, variables.index(i)+1)
        plt.hist(data[i].values, density = True, bins = 30)
        plt.title(i)
    
plt.tight_layout()
plt.show()
 


# In[22]:


obn = data[data['Outcome']==0]
oby = data[data['Outcome']==1]


plt.figure(figsize=[15,15])

excluded = ['Height','Weight','NObeyesdad_short','Outcome']
restriction = list(set(variables) - set(excluded))   

for i in restriction:
    
    plt.subplot(4, 4, restriction.index(i)+1)
    plt.hist(obn[i].values, 100, alpha=0.5, label='Non obese subjects')
    plt.hist(oby[i].values, 100, alpha=0.5, label='Obese subjects')
    plt.title(i)
    plt.legend()        

plt.tight_layout()       
plt.show()


# <center><div class="alert alert-block alert-info">
# Scatter plots
# </div></center>

# In[23]:


sns.catplot( x='Age', y='NObeyesdad_short',data=data)


# In[24]:


obn = data[data['Outcome']==0]
oby = data[data['Outcome']==1]

plt.figure(figsize=[15,20])
restriction = ['Weight','Height','Gender','FAVC: Frequent consumption of high caloric food', 'SCC: Calories consumption monitoring','Family history with overweight','CAEC: Consumption of food between meals', 'SMOKE', 'NObeyesdad_short','Outcome']
filters = [None, 'Gender', 'SMOKE', 'SCC: Calories consumption monitoring', 'Family history with overweight', 'FAVC: Frequent consumption of high caloric food']

for i in [i for i in variables if i not in restriction]:
                
    if i in ('Age','BMI'):
        huei=filters[4]
        coli=filters[1]
        
    if i in ('FCVC','NCP','CAEC','CH2O','CALC'):
        huei=filters[4]
        coli=filters[3]
        
    elif i in ('FAF','TUE', 'MTRANS'):
        huei=filters[4]
        coli=filters[3]
        
    sns.catplot(x='NObeyesdad_short', y=i, hue=huei, data=data, col=coli, sharey=True)
        
plt.tight_layout()
plt.show()


# In[25]:


plt.figure(figsize=[15,10])
variables_scatter=('Age','Height','FCVC: Frequency of consumption of vegetables',
                   'NCP: Number of main meals','CH2O','FAF: Physical activity frequency')

for i in variables_scatter:
    
    plt.subplot(2, 3,variables_scatter.index(i)+1)
    plt.scatter(data['Weight'],data[i], alpha=0.5)
    plt.xlabel('Weight classification')
    plt.ylabel(i)
    plt.title(i + '\'s effect on weight')
    
plt.tight_layout()
plt.show()


# Now that we have exhausted the useful visualizations of the variables 'Weight' and 'Height', and since the column 'BMI' is based on both previously mentioned variables, we will remove them from our dataset before moving on to other visualizations.
# 

# In[26]:


data = data.drop(['Weight','Height'],axis=1)


# In[27]:


sns.pairplot(data)


# In[28]:


sns.pairplot(data, hue = 'Outcome')


# Once we have presented and divided the data, we proceed to do the actual *PRE-PROCESSING STAGE*. This stage is really important due to a good pre-processing stage will facilitate us the work and also will accurate the obtained results.
# 
# ## <center><b><font style="color: darkblue" size="5"> 1.3-DATA CLEANING</font></b></center>
# 
# 
# Then we undergo to data cleaning task. This part is the key for obtaining data for being treated optimally. 
# 
# <blank></blank>
# 
# 
# <center><b><font style="color: skyblue" size="3"> IMBALANCES </font></b></center>
# 
# 

# In[29]:


data['Outcome'].value_counts()


# In[30]:


x = data['Outcome'].unique()
y = data['Outcome'].value_counts()

bar_plot(x,y,"Obesity Distribution",['Obese','Not obese'])


# In the above bar plot, we can see the density of obese and non-obese subjects in our database.
# 
# <u>**Conclusion:**</u> There is no need to treat imbalance in our data, since the number of non-obese subjects is not extremely lower that that of obese subjects.

# In the previous section, we deteled columns 'NObeyesdad', 'MTRANS', 'Weight' and 'Height', since by remaining in our dataset they only provided redundancy (we already had the shorter version of 'NObeyesdad', 'MTRANS', and the 'BMI' was calculated using 'Weight' and 'Height'.
# 
# As we did before and in order to start the actual cleaning process, we will get rid also of the following column:
# 
# - NObeyesdad_short: It represents the obesity classification of the subjects, which is now not necessary since 'Outcome' represents obesity (1) and non-obesity (0).

# In[31]:


data=data.drop('NObeyesdad_short', axis=1)


# <center><b><font style="color: skyblue" size="3"> MISSING VALUES </font></b></center>
# 
# The first stage for data cleaning is dealing with missing values.
# 
# <blank></blank>
# 
# <center> <div class="alert alert-block alert-info">
# N/A
# </div></center>
# 
# 
# 

# In[32]:


print('The dataset has no NA values: {}.'.format(all(data.isnull().sum()==0)))


# <u>**Conclusion:**</u> Since there is no NA values, we do not need to modify the existing data.

# <center><div class="alert alert-block alert-info">
# Zeros
# </div></center>
#  
# Next, we will assess zero values in our data.

# In[33]:


(data == 0).sum(axis=0) 


# As we can see we only have zero values in **FAF and in TUE features.**
# So, now we proceed to a deeper understanding of these two feature in order to evaluate if these zero values are consistent or not.
# 
# Note that we have disregarded "Outcome", since it assigns 0 to non-obese subjects and 1 to obese subjects.

# In[34]:


data.loc[data['FAF: Physical activity frequency'].idxmax()]


# In[35]:


data.loc[data['FAF: Physical activity frequency'].idxmin()]


# In[36]:


data[data['FAF: Physical activity frequency']==0]


# In[37]:


data['TUE: Time using technology devices'].median()


# In[38]:


data.loc[data['TUE: Time using technology devices'].idxmax()]


# With all the above research we can now draw a conclusion:
# 
# - FAF: Physical activity frequency: It is a numerical parameter that represents the times a subject performs any kind of exercise. Since our database contains both obese and non-obese subjects that have a sedentary lifestyle, zero values for FAF are considered feasible.
# 
# - TUE: Time using technology devices: This variable is numerical and represents the time that a subject uses technological devices. Zero values of this variables can be explained:
#     - Older people might not use technological devices long enough to record it.
#     - Our guess is that the authors of the database recorded TUE with "hours" as its units, since most of the database's subjects are young and regardless the maximum value for TUE is 2 and the median is 0.62535.
#     
# <u>**Conclusion:**</u> Therefore, zero values for both FAF and TUE are accepted and will not be modified.
# 

# <center><b><font style="color: skyblue" size="3"> OUTLIERS HANDLING </font></b></center>
# 
# Outliers are those values that are considered extreme when compared to the rest of the distribution.
# 
# 
# Firstly, we indentify the outliers and then we assess whether those values are correct or wrong, whether they make sense.
# 
# 
# The method of identifying the outliers implies performing **boxplots**.

# In[39]:


boxplot = data.boxplot(grid=True, rot=60, fontsize=8)


# In[40]:


data.plot(kind='box',subplots=True, layout=(4,4), sharex=False, sharey=False, figsize=(12,12))
plt.show()


# In[41]:


data['Age'].max()


# In[42]:


data.loc[data['NCP: Number of main meals'].idxmax()]


# In[43]:


Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[44]:


sns.boxplot(y='FAF: Physical activity frequency', x='Outcome', data=data, hue='SCC: Calories consumption monitoring')


# In[45]:


sns.boxplot(y='FAF: Physical activity frequency', x='Outcome',data=data, hue='Family history with overweight')


# In[46]:


sns.boxplot(y='BMI', x='MTRANS_short', data=data, hue='Family history with overweight')


# In[47]:


sns.boxplot(y='BMI', x='Outcome', data=data, hue='Family history with overweight')


# <u>**Conclusion:**</u> After plotting all the above boxplots and inspecting our data, we have decided not to handle outliers, since the values considered so are normal values that are just not close to the mean value. Foe example, an outlier for "Age" is 60 years, which is a perfectly normal value considering that the database's age parameter covers a great range of subjects' ages. It is also normal, regarding MTRANS variable, that some obese people walk and ride the bicycle, and non-obese people do too.
# 
#  

# 
# ## <center><b><font style="color: darkblue" size="5"> 1.4-DATA TRANSFORMATION</font></b></center>
# 
# 
# <center><b><font style="color: skyblue" size="3"> ONE HOT ENCODING </font></b></center>
# 
# One of the main problems with machine learning is **that many algorithms cannot work directly on categorical data.**
# 
# That is why if we  work  with **categorical**  data  we  apply  **ONE-HOT-ENCODING**, which is a representation of categorical variables as binary vectors.
# 
# This procedure  basically  consists  on making a matrix whose columns refer to the attribute of the categorical data. We insert a 1 if the parameter we are assessing has that attribute, or 0 if it does not.
# 
# Firstly we analyze the categorical values and to which column they belong. More precisely, we have the following columns:
# 
# 

# In[48]:


print("Numerical features of our database: ")
numeric= data._get_numeric_data().columns
for i in numeric:
    print("- ",i)
print(" ")
numeric_before_OHE=numeric

print("Categorical features of our database: ")
categ_var = list(set(list(data.columns))-set(list(data._get_numeric_data().columns)))
for i in categ_var:
    print("- ",i)


# As we can observe, we have a huge amount of categorical variables. Therefore, we create the categ_var list, so that we can repeat the same procedure for each categorical variable.

# In[49]:


numer_var = list(data._get_numeric_data().columns)


# In[50]:


# List of all categorical columns:
categ_var = list(set(list(data.columns))-set(list(data._get_numeric_data().columns)))

# creating instance of one-hot-encoder

for i in range(len(categ_var)):
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_data = pd.DataFrame(enc.fit_transform(data[[str(categ_var[i])]]).toarray())
    data[str(categ_var[i])].value_counts() # name of categories of each variable and its frequency
    enc_data.describe()
    data.describe()
    
    word=categ_var[i] + "_onehot"
    word2= "data." + str(categ_var[i])
    print(word)
    word_onehot = pd.get_dummies(word2, prefix=str(categ_var[i]))
    word3= word+ ".head()"
    print(word3)
    data = pd.concat([data,pd.get_dummies(data[str(categ_var[i])], prefix=str(categ_var[i]))],axis=1)
    #careful with this cell, as the above hashtag (if we unhashtag)
    #and we run it every time, it will create each time a new column in our data 
    data.describe()
    #data=data.drop(["Age_Female"], axis=1) #for deleting a column whose name is Age_female


# In[51]:


data


# In the above table we can see that we have gotten rid of every categorical value.
# 
# The only task we have left for our data to be totally and optimally encoded, is deleting categorical columns (as they contain categorical values and as we have duplicated the information of these columns by One-Hot-Encoding, but in binary information):

# In[52]:


# creating instance of one-hot-encoder

enc_data = data

for i in range(len(categ_var)):
    enc_data = enc_data.drop([categ_var[i]], axis=1)


# ## <center><b><font style="color: darkblue" size="5"> 1.5-DIVISION INTO TRAINING AND TEST SETS</font></b></center>
# 
# <blank></blank>
# 
# <center> <b> For regression (Lab 2) </b></center>
# 
# 

# In[53]:


#Create input and output data
x_reg = enc_data.drop(['BMI','Outcome'],axis=1)
y_reg= enc_data['BMI']


# The random state is used to obtain always the same partition
x_reg, y_reg = shuffle(x_reg,y_reg, random_state=0) 

# Training and Test set for regression
X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(x_reg, y_reg,random_state=0) # Regression models


# <center> <b> For classification (Lab 3) </b></center>
# 

# In[54]:


#Create input and output data
x = enc_data.drop(['Outcome'],axis=1)
y = enc_data['Outcome'] 

# The random state is used to obtain always the same partition
x, y = shuffle(x,y, random_state=0)     

# Training and Test set for classification
X_train, X_test, Y_train, Y_test = train_test_split(x, y,random_state=0) # Classification models


# <center><div class="alert alert-block alert-info">
# Subsets' Data Visualization
# </div></center>
# 
# Even though the data has already been visualized, it is important to visually re-inspect the variables once the division into train and tests sets have been performed, and the variables encoded and normalized. 
# 
# <blank></blank>
# 
# <center><b><font style="color: skyblue" size="3"> HISTOGRAMS </font></b></center>
# 
# 
# 

# In[55]:


# obese patients of the train set
oby_tr = X_train.iloc[Y_train.to_numpy().nonzero()]
# obese patients of the test set
oby_te = X_test.iloc[Y_test.to_numpy().nonzero()] 
# non-obese patients of the train set
obn_tr = X_train[~X_train.index.isin(oby_tr.index)]
# non-obese patients of the test set 
obn_te = X_test[~X_test.index.isin(oby_te.index)]


plt.figure(figsize=[15,15])

excluded = ['MTRANS_short_Bike', 'MTRANS_short_PBT',
   'FAVC: Frequent consumption of high caloric food_yes', 'Weight',
   'CAEC: Consumption of food between meals_no',
   'SCC: Calories consumption monitoring_yes', 'MTRANS_short_Walk',
   'CAEC: Consumption of food between meals_Always', 'MTRANS_short_Motorbike', 'Gender_Female',
   'MTRANS_short_Automobile',
   'FAVC: Frequent consumption of high caloric food_no', 'Height',
   'Family history with overweight_yes', 'SMOKE_no',
   'CAEC: Consumption of food between meals_Sometimes', 'Age',
   'CALC: Consumption of alcohol_no',
   'CALC: Consumption of alcohol_Sometimes',
   'Family history with overweight_no', 'Outcome', 'Gender_Male',
   'CALC: Consumption of alcohol_Frequently',
   'CAEC: Consumption of food between meals_Frequently',
   'CALC: Consumption of alcohol_Always', 'SMOKE_yes',
   'SCC: Calories consumption monitoring_no']

restriction = list(set(list(enc_data.columns)) - set(excluded))   


for i in restriction:

    plt.subplot(3, 2, restriction.index(i)+1)
    plt.hist(obn_tr[i].values, 100, alpha=0.5, label='TRAIN - Non obese subjects')
    plt.hist(obn_te[i].values, 100, alpha=0.5, label='TEST - Non obese subjects')
    plt.hist(oby_tr[i].values, 100, alpha=0.5, label='TRAIN - Obese subjects')
    plt.hist(oby_te[i].values, 100, alpha=0.5, label='TEST - Obese subjects')
    plt.title(i)
    plt.legend()        

plt.tight_layout()       
plt.show()


# In[56]:


x_tr = Y_train.unique()
y_tr = Y_train.value_counts()
x_te = Y_test.unique()
y_te = Y_test.value_counts()

bar_plot(x_tr,y_tr,"TRAIN SET - Obesity Distribution",['Obese','Not obese'])
bar_plot(x_te,y_te,"TEST SET - Obesity Distribution",['Obese','Not obese'])


# <center><b><font style="color: skyblue" size="3"> CORRELATION </font></b></center>
# 
# In other to see the relationship between the features a good option is to select a correlation matrix.
# 
# For applying this method is important taking into account that we are going to use the numerical features. 
# 
# So, firstly, we recall which features are the numerical ones:

# In[57]:


# Numerical variables only:

f, ax = plt.subplots(figsize=(20, 20))
corr = data[numer_var].corr()
sns.heatmap(corr,annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# ## <center><b><font style="color: darkblue" size="5"> 1.6- NORMALIZATION </font></b></center>
# 
# Normalization is the process of reorganizing data so that it meets two basic requirements:
# 
#     -Low redundancy
#     -All related data items are stored together, data dependencies are logical
#     
# Normalization will decrease the size of database while improving the algorithm.
# 
# Note: In order to do this normalization, it is necessary to perform previously a change of categorical values. Since we used the One-Hot-Encoder some lines ago, we can go ahead with the normalization.

# In[58]:


scaler=StandardScaler()#we calculate the parameters of the transformation 
scalerfit =  scaler.fit(X_train_reg)
X_train_norm_reg=scaler.transform(X_train_reg) #we call the .transform() method to apply the standardization to the data frame. 
#The .transform() method uses the parameters generated from the .fit() method to perform the z-score.
X_test_norm_reg=scaler.transform(X_test_reg)

scaler=StandardScaler()
scalerfit =  scaler.fit(X_train)
X_train_norm=scaler.transform(X_train)
X_test_norm=scaler.transform(X_test)


#  <center><b><font style="color: skyblue" size="8">___________________________</font></b></center>
# &ensp;
# 
# # <center><b><font style="color: skyblue" size="6">2- LAB2 PARAMETRIC REGRESSION </font></b></center>
# 
# 
#   <center><b><font style="color: skyblue" size="8">___________________________</font></b></center>
# &ensp;
# 
# 
# 
# ## <center><b><font style="color: darkblue" size="5"> 2.1-CHALLENGES FACED IN THE SELECTION OF THE TARGET VARIABLE AND INDEPENDENT VARIABLES </font></b></center>
# 
# 
# <center><b><font style="color: skyblue" size="3"> TARGET VARIABLE </font></b></center>
# 
# In order to compare the performance between the different approaches of parametric regression (simple regresion, multivariate regresion and regularized regression), we decided to select a **common target variable.**
# 
# Our database had as main goal the collection of certain parameters related with eating habits and physical condition in order to investigate a possible link to their obesity status.
# 
# As we are in parametric regression, we do not choose as dependent variable the outcome (obese, not obese), as it is a categorical value. 
# Ohterwise we must select a numerical dependent variable.
# 
# This dependent variable was initially the weight.
# But later we decide in order to avoid the redundancy of parameters to elliminate (in the lab1 section) the height and weight parameter and creating a BMI column. 
# 
# This parameter is more related with the obesity degree due to two people can weight the same kg, but with different height, one will be considered obese and the other one not.
# 
# As conclusion, we have selected **BMI as the dependent variable.**
# 
# <blank></blank>
# 
# <center><b><font style="color: skyblue" size="3"> INITIAL HYPOTHESIS </font></b></center>
# 
# Since we have already selected our target variable now we are going to make an hypothesis about how good could be our algorithm basing on the correlation of BMI and the rest of numerical variables:
# 
# The correlation matrix that was plotted in the previous section showed little correlation between the numberical variables and our dependent variable, the BMI, being the highest correlation value around 0.24 (Age-BMI and TUE-BMI)
# 
# This makes us reason that our database is not suitable to be used with parametric regression.
# 
# However we are going now to make a graphical analysis between our target variable, BMI, and the rest of numerical variables that we have in our dataset in order to perform second-verification of our hypothesis:
# 
# 

# In[59]:


plt.figure(figsize=[15,15])
variables_scatter_lr=('Age','FCVC: Frequency of consumption of vegetables','NCP: Number of main meals',
                   'CH2O','FAF: Physical activity frequency','TUE: Time using technology devices',
                    'MTRANS_short_Walk','Family history with overweight_yes',
                    'Gender_Female')

for i in variables_scatter_lr:
    
    plt.subplot(3, 3,variables_scatter_lr.index(i)+1)
    plt.scatter(enc_data[i],enc_data['BMI'], alpha=0.5)
    plt.xlabel(i)
    plt.ylabel('BMI')
    plt.title(i + ' and BMI')
    
plt.tight_layout()
plt.show()


# The scatter plots displayed before support the conclusions we got with the heat matp: **Parametric regression models would not optimally predict information around our dataset.**
# 
# Nevertheless, we will implement the different approaches (Simple linear Regression, Multivariate Regression, Regression with regularization) model to prove our previous hypothesis. 
# 
# 
# ## <center><b><font style="color: darkblue" size="5"> 2.2-LINEAR REGRESSION ANALYSIS</font></b></center>
# 
# 
# <center><b><font style="color: skyblue" size="3"> INDEPENDENT VARIABLE </font></b></center>
# 
# The first approach that we are going to perform is the linear regression analysis.
# 
# In this approach we only need one independent variable, and as we have observed before, it does not matter what variable was the selected one, all variables would lead to poor results.
# 
# 

# In[60]:


print(numeric_before_OHE) #stores the numerical features before the appliance of the ONE HOT ENCODING 


# In[61]:


from scipy.stats import pearsonr
for i in numeric_before_OHE: 
    corr_test = pearsonr(x = data[str(i)], y =  data['BMI'])
    print("BMI-"+str(i))
    print("Coeficiente de correlación de Pearson: ", corr_test[0])
    print("P-value: ", corr_test[1])
    print(" ")


# We were considering as independent variables 2 features: Age and FCVC.
# Both have correlation with BMI is really similar and although p-value is better in FCVC, it is good in both cases.
# Also we valorate that Age is a continous variable with their values more distributed.
# 
# In this case we decide to analyze our model taking into account as as the **independent variable** the **Age** (and BMI as target variable). 
# 
# <blank></blank>
# 
# <center><b><font style="color: skyblue" size="3"> TWO LINEAR REGRESSION MODELS </font></b></center>
# 
# Now we proceed to elaborate the code.
# Concretely we are going to ellaborate 2 linear regression models:
# 
#                 - non-normalized data
#                 - normalized data
# 
# 
# 
# <div class="alert alert-block alert-info">
#               <center>  Linear regression with no-normalized data<center>
# </div>
# 
# <font style="color: skyblue"> - Code  </font>

# In[63]:


r2_results=[]
mse_results=[]


print(Y_train_reg)
# Simple linear regression
regressor = LinearRegression()

#Train the model using the training set

regressor = regressor.fit(np.array(X_train_reg['Age']).reshape(-1, 1), Y_train_reg)

# Show the intercept
print("Intercept:")
print(regressor.intercept_)
print(" ")

# Show the coeffients
print("Coefficients:")
print(regressor.coef_)
print(" ")

#Predict using the test set
y_pred_reg = regressor.predict(np.array(X_test_reg['Age']).reshape(-1,1))

# Compute the MSE
mse=metrics.mean_squared_error(Y_test_reg, y_pred_reg)
mse_results.append(mse)


# Compute the R2
r_squared=metrics.r2_score(Y_test_reg, y_pred_reg)
r2_results.append(r_squared)                           

print(f'R Squared: {r_squared}\nMean Squared Error:{mse}')


# 
# <font style="color: skyblue"> - Plot the samples  </font>

# In[64]:


# Plot the samples and the predict
plt.scatter(X_test_reg['Age'],Y_test_reg,color='skyblue')
plt.plot(X_test_reg['Age'], y_pred_reg,color='blue')
plt.xlabel('Age')
plt.ylabel("BMI")
plt.show()


# <div class="alert alert-block alert-info">
#               <center>  Linear regression with normalized data<center>
# </div>

# <center><b><font style="color: skyblue" size="2"> Code </font></b></center>

# In[65]:


# Normalization

# Simple linear regression
regressor = LinearRegression()

#Train the model using the training set
regressor = regressor.fit(X_train_norm_reg[:,0].reshape(-1, 1), Y_train_reg)

# Show the intercept
print(regressor.intercept_)

# Show the coeffients
print(regressor.coef_)

#Predict using the test set
y_pred_lsr = regressor.predict(X_test_norm_reg[:,0].reshape(-1,1))

# Compute the MSE
mse_lsr=metrics.mean_squared_error(Y_test_reg, y_pred_lsr)
mse_results.append(mse_lsr)


# Compute the R2
r_squared_lsr=metrics.r2_score(Y_test_reg, y_pred_lsr)
r2_results.append(r_squared_lsr)

print(f'R Squared: {r_squared_lsr} \n Mean Squared Error:{mse_lsr}')


# <center><b><font style="color: skyblue" size="2"> Plot the samples </font></b></center>

# In[66]:


# Plot the samples and the predict
plt.scatter(X_test_norm_reg[:,0],Y_test_reg,color='blue')
plt.plot(X_test_norm_reg[:,0], y_pred_reg,color='skyblue')
plt.xlabel('Age')
plt.ylabel("BMI")
plt.show()


# <center><b><font style="color: skyblue" size="3"> CONCLUSIONS BETWEEN NORMALIZED AND NON-NORMALIZED DATA LINEAR REGRESSION</font></b></center>
# 
# <blank></blank>
# 
# <u>**Conclusion:**</u> 
# We can observe that metric results does not change, the only transformation that we can appreciate is the change of view of the graphs.
# 
# The model is able to explain **6.2%** of the variability observed in the dependent variable (BMI).
# For each unit increase in the number of hits, the BMI value increases by an average of **59.27** units.
# 
# After looking at the visualizations of the pre-processing stage, we observe that the most suitable feature to predict the BMI is the variable 'Family history with overweight'.
# 
# 
# ## <center><b><font style="color: darkblue" size="5"> 2.3- MULTIPLE LINEAR REGRESSION </font></b><center>
# 
# <blank></blank>
#     
# <center><b><font style="color: skyblue" size="3"> INDEPENDENT VARIABLE </font></b></center>
# 
# Now we are going to face to the multivariate regression analysis. 
# 
# In this part we are going to have more than one independent variables. So, the first thing that we are going to focus is to decide which independent variable are we going to add. As we have previosly seen BMI is not so much correlated with the numerical variables that we already have. But we are going to check if BMI is correlated with categorial variables. This will be done by visual inspection making use of the graphs.
# 
# 
# So, as we have said we are going to do multivariate regression analysis and the dependent variable will still be BMI, and, on the other side, the **independent variables** will be **age** and **family history with overweight.**
# 
# 
# <div class="alert alert-block alert-info">
#               <center> Facing a categorical independent variable<center>
# </div>
# 
# With this last parameter we have to be really careful due to is categorical, we have already solved that though **One Hot Encoder method**. By applying this method the family history with overweight splitted in 2 columns (Family history with overweight_no,Family history with overweight_yes). These columns have 0 and 1 values, so the categorical problem had been already solved.
# 
# 
# <div class="alert alert-block alert-info">
#               <center> Facing a categorical independent variable in multivariate regression analysis <center>
# </div>
# 
# The next problem with which we have to deal is knowing have to do multivariate regression analysis when one of the independent variables had suffered One Hot Encoder.
# 
# As a first aproach, we thought that in x_train we had to add the BMI column an also both columns Family history with overweight_no, Family history with overweight_yes, so, we would have 3 coefficients in total.
# 
# But after a research, that assumption was wrong due to we would violating the absence of **multicolinearity principle**. This principle says that if one independent variable can be extracted by other selected independent variable is a bad situation and one of them must be elliminated. 
# 
# We can clearly extrapolate this to our example, always that our Family history with overweight_yes column has a 0 value the Family history with overweight_no has value 1 and viceversa. So, we decided to select only family history with overweight_yes as our added independent variable and therefore we discard the  family history with overweight_no (the point is selecting one of them, it does not matter which).
# 
# Knowing all the above we are going to extract the "Age" and "Family History with Overweight_yes and we will store them in X_trian_mul (that belongs to the training set) and in X_test_mul (that belongs to the test set).

# In[67]:


#in order to extract our desired variables we use loc() function.


X_train_mul= X_train.loc[:,['Age','Family history with overweight_yes']]
X_test_mul= X_test.loc[:,['Age','Family history with overweight_yes']]


# <center><b><font style="color: skyblue" size="3"> CODING OUR ALGORITHM </font></b></center>

# In[68]:



regressor = LinearRegression()

#Train the model using the training set
regressor = regressor.fit(X_train_mul, Y_train_reg)

# Show the intercept
print(regressor.intercept_)

# Show the coeffients
print(regressor.coef_)

#Predict using the test set
y_pred_mlr = regressor.predict(X_test_mul)

# Compute the MSE
mse_mlr=metrics.mean_squared_error(Y_test_reg, y_pred_mlr)
mse_results.append(mse_mlr)

# Compute the R2
r_squared_mlr=metrics.r2_score(Y_test_reg, y_pred_mlr)
r2_results.append(r_squared_mlr)

print(f'R Squared: {r_squared_mlr} \n Mean Squared Error:{mse_mlr}')


# <center><b><font style="color: skyblue" size="3"> PLOTTING THE RESULTS </font></b></center

# In[69]:


x_coef=np.arange(regressor.coef_.shape[0])
print(x_coef)
plt.plot( x_coef, abs(regressor.coef_) ,color='r', marker="o")
plt.xticks(np.arange(regressor.coef_.shape[0]), X_train_mul.columns, rotation=90,  fontsize=12)  # Set text labels and properties.
plt.grid()


# <center><b><font style="color: skyblue" size="3"> CONCLUSIONS </font></b></center>
# 
# <blank></blank>
# 
# <u>**Conclusion:**</u> 
# The model is able to explain **26.52%** of the variability observed in the dependent variable (BMI).
# 
# For each unit increase in the number of hits, the BMI value increases by an average of **46.44** units.
# 
# <blank></blank>
# 
# <blank></blank>
# 
# ## <center><b><font style="color: darkblue" size="5"> 2.4- NON LINEAR REGRESSION </font></b></center>
# 
# 
# At this moment we are going to address the non-linear regression. 
# That is that the relationship of our variables are not going to follow a "line", they are going to follow a non-linear equation, as for instance polynomial equation.
# 
# We have decided therefore to use **non-linear polynomial equation.**
# 
# The first thing that we must perform is the cross validation in order to know which degree of the polynomio is going to be selected.
# 
# <blank></blank>
# 
# <center><b><font style="color: skyblue" size="3"> CROSS VALIDATION </font></b></center
# 
# 

# In[70]:


X_train_pol= X_train.loc[:,['Age','Family history with overweight_yes']]
X_test_pol= X_test.loc[:,['Age','Family history with overweight_yes']]


cv_degree_scores=[]
d_values = range(2,7,1)#degrees 2-7
for d in d_values:
    poly_reg = PolynomialFeatures(degree=d)
    X_train_poly = poly_reg.fit_transform(X_train_pol)
    X_test_poly = poly_reg.transform(X_test_pol)
    pol_reg = LinearRegression()
    scores= cross_val_score(pol_reg, X_train_poly, Y_train_reg, cv=3, scoring='neg_mean_squared_error')
    cv_degree_scores.append(scores.mean())
    
plt.plot(d_values, cv_degree_scores)
plt.xlabel('degree')
plt.ylabel('CV mse')
plt.show()

# Select the maximum because we are considering accuracuy
print("The degree of the polynomio will be:")
print(np.array(d_values)[cv_degree_scores.index(np.array(cv_degree_scores).max())])


# <center><b><font style="color: skyblue" size="3"> CODE </font></b></center>
# 
# 
# 
# 
# Now we are going to perform the polynomial regression.

# In[71]:


# Train and predict with the best degree
poly_reg = PolynomialFeatures(degree=np.array(d_values)[cv_degree_scores.index(np.array(cv_degree_scores).max())])
X_train_poly = poly_reg.fit_transform(X_train_pol)
X_test_poly = poly_reg.transform(X_test_pol)

# Train
pol_reg = LinearRegression()
pol_reg.fit(X_train_poly, Y_train_reg)
y_pred_pol=pol_reg.predict(X_test_poly)

# Compute the MSE
mse_pol=metrics.mean_squared_error(Y_test_reg, y_pred_pol)
mse_results.append(mse_pol)

# Compute the R2
r_squared_pol=metrics.r2_score(Y_test_reg, y_pred_pol)
r2_results.append(r_squared_pol)

# Print results
print(f'R Squared: {r_squared_pol} \n Mean Squared Error:{mse_pol}')


# <center><b><font style="color: skyblue" size="3"> CONCLUSIONS </font></b></center>
# 
# <blank></blank>
# 
# <u>**Conclusion:**</u> 
# The model is able to explain **37.13%** of the variability observed in the dependent variable (BMI).
# 
# For each unit increase in the number of hits, the BMI value increases by an average of **39.74** units.
# 
# 
# 
# ## <center><b><font style="color: darkblue" size="5"> 2.5- LINEAR REGRESSION WITH REGULARIZATION </font></b></center>
# 
# <blank></blank>
# 
# And lastly we are going to perform the linear regression with regularisation
# 
# Regularization is most used for avoiding over-fitting phenomenon.
# This technique imply the addition of  a  penalty  term  to  the  cost function.
# We have 3 approaches: Ridge, Lasso, Elastic Net. But we are going to perform only 2:
# 
# <blank></blank>
# 
# <center><b><font style="color: skyblue" size="3"> RIDGE </font></b></center>
# 
# 
# 

# In[72]:


X_train_reg= X_train.loc[:,['Age','Family history with overweight_yes']]
X_test_reg= X_test.loc[:,['Age','Family history with overweight_yes']]

# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

# Values for alpha
parameters = {"alpha":[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20,30]}

# Grid search for ridge regression
ridge_regression = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)

# Train 
ridge_regression.fit(X_train_reg, Y_train_reg)

# Best parameters and best score
print(ridge_regression.best_params_)
print(ridge_regression.best_score_)


# Predict
y_pred_ridge = ridge_regression.predict(X_test_reg)

# Compute the MSE
mse_ridge=metrics.mean_squared_error(Y_test_reg, y_pred_ridge)
mse_results.append(mse_ridge)


# Compute the R2
r_squared_ridge=metrics.r2_score(Y_test_reg, y_pred_ridge)
r2_results.append(r_squared_ridge)

print(f'R Squared: {r_squared_ridge} \n Mean Squared Error:{mse_ridge}')


# 
# 
# <u>**Conclusion:**</u> 
# The model is able to explain **26.54%** of the variability observed in the dependent variable (BMI).
# 
# For each unit increase in the number of hits, the BMI value increases by an average of **46.44** units.
# 
# <blank></blank>
# 
# <center><b><font style="color: skyblue" size="3"> LASSO </font></b></center
# 
# 

# In[73]:


X_train_reg= X_train.loc[:,['Age','Family history with overweight_yes']]
X_test_reg= X_test.loc[:,['Age','Family history with overweight_yes']]


lasso = Lasso()

# Values for alpha
parameters = {"alpha":[1e-4, 1e-2, 1, 5, 10, 20]}

# Grid search for lasso regression
lasso_regression = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)

# Train 
lasso_train=lasso_regression.fit(X_train_reg, Y_train_reg)

print(lasso_regression.best_params_)
print(lasso_regression.best_score_)

# Predict
y_pred_lasso = lasso_regression.predict(X_test_reg)

# Compute the MSE
mse_lasso=metrics.mean_squared_error(Y_test_reg, y_pred_lasso)
mse_results.append(mse_lasso)




# Compute the R2
r_squared_lasso=metrics.r2_score(Y_test_reg, y_pred_lasso)
r2_results.append(r_squared_lasso)

#Print results
print(f'R Squared: {r_squared_lasso} \n Mean Squared Error:{mse_lasso}')
 
# Calculate Mean Squared Error
mean_squared_error = np.mean((y_pred_lasso - Y_test_reg)**2)
print("Mean squared error on test set", mean_squared_error)



# <u>**Conclusion:**</u> 
# The model is able to explain **26.54%** of the variability observed in the dependent variable (BMI).
# 
# For each unit increase in the number of hits, the BMI value increases by an average of **46.44** units.
# 

# 
# ## <center><b><font style="color: darkblue" size="5"> 2.6- COMPARISON OF THE RESULTS OBTAINED WITH THE DIFFERENT MODELS </font></b></center>
# 
# 
# <center><b><font style="color: skyblue" size="3"> COMPARATIVE TABLE </font></b></center>
# 
# 
# 
# We are going to summarize all the previous results in order to get conclusions:

# In[74]:


results = pd.DataFrame()

results["R2 in the test set"] = r2_results
results["MSE in the test set"] = mse_results

#results
results["Models"] = ["Linear Regression (LR) ", "LR Normalized", "Multiple LR", "Non-LR", "Rigde", "Lasso"]
results.set_index("Models", inplace = True)


# In[75]:


results


# <center><b><font style="color: skyblue" size="3">R2 LITTLE EXPLANATION </font></b></center>
# 
# <blank></blank>
# 
# 
# R squared indicates how much percentage of the variability of the dependent variable our model can explain. In other words,  R2 represents the dispersion around the regression line.
# 
# Concretely R squared takes values between 0 and 1.
# 
# Usually the larger R square, the better the model
# 
# <blank></blank>
# 
# <center><b><font style="color: skyblue" size="3">CONCLUSIONS </font></b></center>
# 
# <blank></blank>
# 
# We can clearly see that the **SIMPLE LINEAR REGRESSION** using BMI as dependent variable and Age as the independent one provide us **poor results** both for normalized data an not.
# 
# By adding the categorical independent variable "Family history with overweight" in the **MULTIPLE LINEAR REGRESION**, that a priori was more correlated than Age to BMI, significantly increase the performance of our algorithm (although it is still not so good). Therefore incrementing the explanation of the variability of the BMI by 4th times the ones explained in the simple LR model. Moreover we also have decreased the MSE which is also good.
# 
# Here, in the **NON-LINEAR REGRESSION** we reach our maximum value of R squared (0.3713). Hereby we checked that although we weren't so sure of obtaining good results with this approach, until we did not perform it, we would not know which the results would be. Here again we have decreased the MSE. So, **this technique is the best approach that we have performed.**
# 
# And lastly the **REGRESSION WITH REGULARIZATION** approaches (Ridge and Lasso) provide us better results that the ones obtained with the linear regression, but not as good as the ones provided by the Non-Linear
# 

# 
# 
# <center><b><font style="color: skyblue" size="8">___________________________</font></b></center>
# &ensp;
# 
# # <center><b><font style="color: skyblue" size="6">3- LAB3 PARAMETRIC CLASSIFICATION </font></b></center>
# 
# 
#   <center><b><font style="color: skyblue" size="8">___________________________</font></b></center>
# &ensp;
# 
# 
# <center><b><font style="color: skyblue" size="3"> Selecting the target variable </font></b></center>
# 
# The **difference between parametric classification and parametric regression** lies on the **target variable.** In regression, the target variable is __numerical__, so that our aim is to estimate the numerical value of the target variable. On the other hand, in classification the target variable will be __categorical__: Our goal will consist of establishing a threshold that will separate the classes of the target variable, so that the algorithm will predict which class a subject belongs to.
# 
# 
# Taking this into consideration, we decided to select **'OUTCOME' as our target variable**.
# 
# 
# At first, the subjects fell under one of six different types: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. In order not to have a multi-class classification problem, we decided to create the binary variable Outcome, which groups obesity types into:
# 
#     - Obese (1): Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II.
#     - Not obese (0): Obesity Type I, Obesity Type II and Obesity Type III.
#     
# 
# 
# ## <center><b><font style="color: darkblue" size="5"> 3.1-LOGISTIC REGRESSION </font></b></center>
# 
# First of all, we create the following lists, which will be used to store the performace metrics of our model:

# In[76]:


accuracy=[]
sensitivity=[]
specificty=[]
auc=[]


# 
# <center><b><font style="color: skyblue" size="3"> Selecting the independent variable </font></b></center>
# 
# In order to select the independent variable to be used, we recall the pre-processing stage which gave us information about the effect different parameters have on weight. It is easy to see that the **Body Mass Index** would be a great independent variable:
# 
# 
# 

# In[77]:


x_obese = X_train.iloc[Y_train.to_numpy().nonzero()]
x_nobese = X_train[~X_train.index.isin(oby_tr.index)]


plt.figure(figsize=[10,8])

plt.hist(x_obese['BMI'].values, density = True,bins = 30, label='Obese subjects')
plt.hist(x_nobese['BMI'].values, density = True, bins = 30, label='Non obese subjects')

plt.title('BMI Distribution')

plt.legend()        
plt.tight_layout() 
plt.show()


# As we can observe by visually inspecting the graph above, the BMI is a clearly optimal parameter as our independent variable.
# 
# <blank></blank>
# 
# <center><b><font style="color: skyblue" size="3"> Coding </font></b></center>
# 
# Now that we have chosen our dependent and independent variables, we can start to code the algorithm.

# In[78]:


# Logist regressioin
regressor = LogisticRegression()


# Train the model using X_train
regressor = regressor.fit(np.array(X_train['BMI']).reshape(-1, 1), Y_train)

# Show the intercept
print("Intercept:")
print(regressor.intercept_)
print(" ")

# Show the coefficients
print("Coefficients:")
print(regressor.coef_)
print(" ")

# Predicted values in the test set
y_pred_cla = regressor.predict(np.array(X_test['BMI']).reshape(-1,1))
y_prob_pred= regressor.predict_proba(np.array(X_test['BMI']).reshape(-1,1))

print(f'Y_pred: {y_pred_cla} ')
print(" ")
print(f'y_prob_pred: {y_prob_pred} ')


# <center><b><font style="color: skyblue" size="3"> Evaluating our model </font></b></center>
# 
# Once that we have designing our classifier we are going to perform an evaluation of our algotithm
# 
# We have several approaches to evaluate our model:   
#    - Confusion matrix.
#    - Metric obtained from the confusion matrix.
#    - ROC.
#    - AUC.
# 
# <center><b><font style="color: BLACK" size="4"> _______________</font></b></center>
# 
# <center><b><font style="color: BLACK" size="3"> Confusion matrix </font></b></center>
# <center><b><font style="color: BLACK" size="4"> _______________</font></b></center>

# In[79]:


#  Compute the confusion matrix

cm = confusion_matrix(Y_test, y_pred_cla)

#in order to a better visualization we transform data into table view

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_cla).ravel()
users = {'Non-obese': [tn, fp],'Obese': [fn, tp]}

confusion_table = pd.DataFrame(users, index = ['Non-obese', 'Obese'])

#print the results
print("The confusion matrix of our classifer is : ")
print(" ")
print(confusion_table)


# Keeping in mind the configuration of the confusion matrix:
# 
# <blank></blank>
# 
# <blank></blank>
#  
# <div>
# <img src="https://www.nbshare.io/static/snapshots/cm_colored_1-min.png" width="500"/>
# </div>
# 
# <blank></blank>
# 
# <blank></blank>
# 
# To further assess the performarce of our model, we will hereby compute some more metrices.
# 
# <blank></blank>
# 
# <center><b><font style="color: BLACK" size="4"> ___________________________</font></b></center>
# 
# <center><b><font style="color: BLACK" size="3"> Accuracy and Sensibility </font></b></center>
# <center><b><font style="color: BLACK" size="4"> ___________________________</font></b></center>

# In[80]:


# Compute the accuracy
Accuracy=accuracy_score(Y_test, y_pred_cla)
print(f'Accuracy: {Accuracy}')

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_cla, target_names=target_names))
              
#Sensibility
Sensibility=recall_score(Y_test, y_pred_cla)
print(f'Sensibility: {Sensibility}')


# In[81]:


Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_cla)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# <center><b><font style="color: BLACK" size="4"> _______________</font></b></center>
# 
# <center><b><font style="color: BLACK" size="3"> ROC and AUC </font></b></center>
# <center><b><font style="color: BLACK" size="4"> _______________</font></b></center>

# In[82]:


# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_prob_pred[:,1])
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# 
# ## <center><b><font style="color: darkblue" size="5"> 3.2-MULTIVARIATE LOGISTIC REGRESSION </font></b></center>
# 
# 
# <center><div class="alert alert-block alert-info">
# BMI, Family history, NCP
# </div></center>
# 
# 

# In[83]:


x_in = X_train[['BMI','Family history with overweight_yes','NCP: Number of main meals']]

regressor = LogisticRegression(solver='lbfgs', max_iter=10000)

# Train the model using the training data
regressor_lrnn = regressor.fit(x_in, Y_train)

print(regressor_lrnn.intercept_)
print(regressor_lrnn.coef_)

y_pred_lrnn = regressor_lrnn.predict(X_test[['BMI','Family history with overweight_yes','NCP: Number of main meals']])

cm = confusion_matrix(Y_test, y_pred_lrnn)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_lrnn).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}\n')

Accuracy=accuracy_score(Y_test, y_pred_lrnn)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_lrnn, target_names=target_names))
              
Sensibility=recall_score(Y_test, y_pred_lrnn)

# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_lrnn)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_lrnn)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# <center><div class="alert alert-block alert-info">
# BMI, Family history, FAF
# </div></center>

# In[84]:


x_in = X_train[['BMI','Family history with overweight_yes','FAF: Physical activity frequency']]

regressor = LogisticRegression(solver='lbfgs', max_iter=10000)

# Train the model using the training data
regressor_lrnn = regressor.fit(x_in, Y_train)

print(regressor_lrnn.intercept_)
print(regressor_lrnn.coef_)

y_pred_lrnn = regressor_lrnn.predict(X_test[['BMI','Family history with overweight_yes','FAF: Physical activity frequency']])

cm = confusion_matrix(Y_test, y_pred_lrnn)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_lrnn).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}\n')

Accuracy=accuracy_score(Y_test, y_pred_lrnn)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_lrnn, target_names=target_names))
              
Sensibility=recall_score(Y_test, y_pred_lrnn)

# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_lrnn)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_lrnn)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# <center><div class="alert alert-block alert-info">
# BMI, Family history, SCC
# </div></center>

# In[85]:


x_in = X_train[['BMI','Family history with overweight_yes','SCC: Calories consumption monitoring_yes']]

regressor = LogisticRegression(solver='lbfgs', max_iter=10000)

# Train the model using the training data
regressor_lrnn = regressor.fit(x_in, Y_train)

print(regressor_lrnn.intercept_)
print(regressor_lrnn.coef_)

y_pred_lrnn = regressor_lrnn.predict(X_test[['BMI','Family history with overweight_yes','SCC: Calories consumption monitoring_yes']])

cm = confusion_matrix(Y_test, y_pred_lrnn)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_lrnn).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}\n')

Accuracy=accuracy_score(Y_test, y_pred_lrnn)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_lrnn, target_names=target_names))
              
Sensibility=recall_score(Y_test, y_pred_lrnn)

# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_lrnn)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_lrnn)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# <center><div class="alert alert-block alert-info">
# BMI, Family history, TUE
# </div></center>

# In[86]:


x_in = X_train[['BMI','Family history with overweight_yes','TUE: Time using technology devices']]

regressor = LogisticRegression(solver='lbfgs', max_iter=10000)

# Train the model using the training data
regressor_lrnn = regressor.fit(x_in, Y_train)

print(regressor_lrnn.intercept_)
print(regressor_lrnn.coef_)

y_pred_lrnn = regressor_lrnn.predict(X_test[['BMI','Family history with overweight_yes','TUE: Time using technology devices']])

cm = confusion_matrix(Y_test, y_pred_lrnn)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_lrnn).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}\n')

Accuracy=accuracy_score(Y_test, y_pred_lrnn)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_lrnn, target_names=target_names))
              
Sensibility=recall_score(Y_test, y_pred_lrnn)

# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_lrnn)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_lrnn)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# <center><div class="alert alert-block alert-info">
# BMI, Family history, FAVC
# </div></center>

# In[87]:


x_in = X_train[['BMI','Family history with overweight_yes',
                'FAVC: Frequent consumption of high caloric food_yes']]

regressor = LogisticRegression(solver='lbfgs', max_iter=10000)

# Train the model using the training data
regressor_lrnn = regressor.fit(x_in, Y_train)

print(regressor_lrnn.intercept_)
print(regressor_lrnn.coef_)

y_pred_lrnn = regressor_lrnn.predict(X_test[['BMI','Family history with overweight_yes',
                'FAVC: Frequent consumption of high caloric food_yes']])

cm = confusion_matrix(Y_test, y_pred_lrnn)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_lrnn).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}\n')

Accuracy=accuracy_score(Y_test, y_pred_lrnn)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_lrnn, target_names=target_names))
              
Sensibility=recall_score(Y_test, y_pred_lrnn)

# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_lrnn)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_lrnn)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# <center><div class="alert alert-block alert-info">
# BMI, Family history, NCP, SCC, FAVC
# </div></center>

# In[88]:


x_in = X_train[['BMI','Family history with overweight_yes','NCP: Number of main meals',
                'SCC: Calories consumption monitoring_yes',
                'FAVC: Frequent consumption of high caloric food_yes']]

regressor = LogisticRegression(solver='lbfgs', max_iter=10000)

# Train the model using the training data
regressor_lrnn = regressor.fit(x_in, Y_train)

print(regressor_lrnn.intercept_)
print(regressor_lrnn.coef_)

y_pred_lrnn = regressor_lrnn.predict(X_test[['BMI','Family history with overweight_yes','NCP: Number of main meals',
                'SCC: Calories consumption monitoring_yes',
                'FAVC: Frequent consumption of high caloric food_yes']])

cm = confusion_matrix(Y_test, y_pred_lrnn)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_lrnn).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}\n')

Accuracy=accuracy_score(Y_test, y_pred_lrnn)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_lrnn, target_names=target_names))
              
Sensibility=recall_score(Y_test, y_pred_lrnn)

# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_lrnn)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_lrnn)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# <center><div class="alert alert-block alert-info">
# BMI, Family history, FAF, TUE
# </div></center>

# In[89]:


x_in = X_train[['BMI','Family history with overweight_yes','FAF: Physical activity frequency',
                'TUE: Time using technology devices']]

regressor = LogisticRegression(solver='lbfgs', max_iter=10000)

# Train the model using the training data
regressor_lrnn = regressor.fit(x_in, Y_train)

print(regressor_lrnn.intercept_)
print(regressor_lrnn.coef_)

y_pred_lrnn = regressor_lrnn.predict(X_test[['BMI','Family history with overweight_yes','FAF: Physical activity frequency',
                'TUE: Time using technology devices']])

cm = confusion_matrix(Y_test, y_pred_lrnn)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_lrnn).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}\n')

Accuracy=accuracy_score(Y_test, y_pred_lrnn)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_lrnn, target_names=target_names))
              
Sensibility=recall_score(Y_test, y_pred_lrnn)

# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_lrnn)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_lrnn)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# <center><div class="alert alert-block alert-info">
# BMI, Family history, NCP, FAF, SCC, TUE, FAVC
# </div></center>

# In[90]:


x_in = X_train[['BMI','Family history with overweight_yes','NCP: Number of main meals',
                'FAF: Physical activity frequency','SCC: Calories consumption monitoring_yes',
                'TUE: Time using technology devices','FAVC: Frequent consumption of high caloric food_yes']]

regressor = LogisticRegression(solver='lbfgs', max_iter=10000)

# Train the model using the training data
regressor_lrnn = regressor.fit(x_in, Y_train)

print(regressor_lrnn.intercept_)
print(regressor_lrnn.coef_)

y_pred_lrnn = regressor_lrnn.predict(X_test[['BMI','Family history with overweight_yes','NCP: Number of main meals',
                'FAF: Physical activity frequency','SCC: Calories consumption monitoring_yes',
                'TUE: Time using technology devices','FAVC: Frequent consumption of high caloric food_yes']])

cm = confusion_matrix(Y_test, y_pred_lrnn)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_lrnn).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}\n')

Accuracy=accuracy_score(Y_test, y_pred_lrnn)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_lrnn, target_names=target_names))
              
Sensibility=recall_score(Y_test, y_pred_lrnn)

# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_lrnn)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_lrnn)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# <center><div class="alert alert-block alert-info">
# Whole set of data
# </div></center>

# In[91]:


regressor = LogisticRegression(solver='lbfgs', max_iter=10000)

# Train the model using the training data
regressor_lrnn = regressor.fit(X_train, Y_train)

print(regressor_lrnn.intercept_)
print(regressor_lrnn.coef_)

y_pred_lrnn = regressor_lrnn.predict(X_test)

cm = confusion_matrix(Y_test, y_pred_lrnn)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_lrnn).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}\n')

Accuracy=accuracy_score(Y_test, y_pred_lrnn)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_lrnn, target_names=target_names))
              
Sensibility=recall_score(Y_test, y_pred_lrnn)

# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_lrnn)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_lrnn)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# <center><div class="alert alert-block alert-info">
# Normalized data
# </div></center>

# In[92]:



# Logistic regression
regressor = LogisticRegression()

# Train the model using the training data
regressor_lrm = regressor.fit(X_train_norm, Y_train)

# Show the intercept
print(regressor_lrm.intercept_)

# Show the coefficients
print(regressor_lrm.coef_)

# Compute the predicted value in the test set
y_pred_lrm = regressor_lrm.predict(X_test_norm)

#  Compute the confusion matrix
cm = confusion_matrix(Y_test, y_pred_lrm)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_lrm).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}\n')

# Compute the accuracy 
Accuracy=accuracy_score(Y_test, y_pred_lrm)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_lrm, target_names=target_names))
              
#Sensibility
Sensibility=recall_score(Y_test, y_pred_lrm)

# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_lrm)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_lrm)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# Up until now, our models' **results have been close to perfect**:
# 
# - All figures of merit, that is accuracy, sensibility, specificity, precision and F1 score, have been very close to 1, meaning that our predictions have close to no errors. 
# - The AUC, Area Under the ROC (Receiver Operating Characteristic) Curve has also been 1 for most cases, representing a perfect predictive characteristic.
# 
# 
# The predictions have been so good that one would think they are wrong; in the next section we will design a model just in case our models have been overfit. However, we will now try to use other variables as the independent ones: From the very first moment, we chose the BMI as the independent variable, due to its relation with the obesity degree of patients. Below we will not use BMI to predict obesity of the subjects, let us see how the results turn to be.

# <center><div class="alert alert-block alert-info">
# Age, Family history, FAF, TUE
# </div></center>

# In[93]:


x_in = X_train[['Age','Family history with overweight_yes','FAF: Physical activity frequency',
                'TUE: Time using technology devices']]

regressor = LogisticRegression(solver='lbfgs', max_iter=10000)

# Train the model using the training data
regressor_lrnn = regressor.fit(x_in, Y_train)

print(regressor_lrnn.intercept_)
print(regressor_lrnn.coef_)

y_pred_lrnn = regressor_lrnn.predict(X_test[['Age','Family history with overweight_yes','FAF: Physical activity frequency',
                'TUE: Time using technology devices']])

cm = confusion_matrix(Y_test, y_pred_lrnn)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_lrnn).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}\n')

Accuracy=accuracy_score(Y_test, y_pred_lrnn)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_lrnn, target_names=target_names))
              
Sensibility=recall_score(Y_test, y_pred_lrnn)

# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_lrnn)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_lrnn)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# As suspected, **not using the BMI to make predictions greatly affects our model's performance**. Therefore, the good results obtained so far might be solely due to the great correlation between BMI and the obesity classification of patients. Recall the BMI distribution, discriminating between obese and non-obese patients:

# In[94]:


x_obese = X_train.iloc[Y_train.to_numpy().nonzero()]
x_nobese = X_train[~X_train.index.isin(oby_tr.index)]


plt.figure(figsize=[5,3])

plt.hist(x_obese['BMI'].values, density = True,bins = 30, label='Obese subjects')
plt.hist(x_nobese['BMI'].values, density = True, bins = 30, label='Non obese subjects')

plt.title('BMI Distribution')

plt.legend()        
plt.tight_layout() 
plt.show()


# 
# ## <center><b><font style="color: darkblue" size="5"> 3.3-LOGISTIC REGRESSION WITH REGULARIZATION </font></b></center>
# 
# 
# Since we have a huge amount of data, it could be possible that the results obtained in the previous sections were wrong due to overfitting. With this in mind, in this section we will design a model that will predict while using regularization, which brings in a penalization. By using regularization we aim to reduce overfitting of results, i.e. keep our model from knowing our data and its noise so that the predictions are perfect (as long as the data is not changed). 
# 
# <blank></blank>
# 
# <center><b><font style="color: skyblue" size="3"> Lasso and Ridge </font></b></center>
# 
# The penalization used in this case will be based on both Lasso and Ridge methods.
# 

# In[95]:


# C values and penalty
parameters = grid = {"C":np.linspace(1e-4,10,100), "penalty":["l1","l2"]} # l1 lasso l2 ridge

logreg=LogisticRegression(solver='liblinear')
logreg_cv=GridSearchCV(logreg,parameters,cv=5)
logreg_cv.fit(X_train_norm,Y_train)

# Grid search
log_regression = GridSearchCV(logreg, parameters, scoring='accuracy', cv=5)

# Train using X_train
log_regression.fit(X_train_norm, Y_train)

# We show the best value of the parameter and the score
print(log_regression.best_params_)
print(log_regression.best_score_)


# The predicted output is obtained
y_pred_reg = log_regression.predict(X_test_norm)


# In[96]:


#  Compute the confusion matrix
cm = confusion_matrix(Y_test, y_pred_reg)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_reg).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}')

# Compute the accuracy_score
Accuracy=accuracy_score(Y_test, y_pred_reg)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_reg, target_names=target_names))
              
#Sensibility
Sensibility=recall_score(Y_test, y_pred_reg)


# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_reg)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_reg)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# <center><b><font style="color: skyblue" size="3"> Lasso </font></b></center>
# 
# Now, we will just apply regularization with Lasso.
# 

# In[97]:


lasso = Lasso()

logreg=LogisticRegression(solver='liblinear', penalty='l1')

# Alpha values
parameters = {"C":[1e-4, 1e-2, 1, 5, 10, 20]}

# Grid search 
lasso_regression = GridSearchCV(logreg, parameters, scoring='accuracy', cv=5)

# Train using the Xtrain
lasso_train=lasso_regression.fit(X_train_norm, Y_train)

print(lasso_regression.best_params_)
print(lasso_regression.best_score_)

# Compute the predicted output
y_pred_lasso = lasso_regression.predict(X_test_norm)


# In[98]:


#  Compute the confusion matrix
cm = confusion_matrix(Y_test, y_pred_lasso)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_lasso).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}')

# Compute the accuracy_score
Accuracy=accuracy_score(Y_test, y_pred_lasso)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_lasso, target_names=target_names))
              
#Sensibility
Sensibility=recall_score(Y_test, y_pred_lasso)


# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_lasso)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_lasso)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# <center><b><font style="color: skyblue" size="3"> Elastic Net </font></b></center>
# 
# 
# Another method that is used when designing regression with regularization models is Elastic Net.
# 

# In[99]:


logreg = LogisticRegression(solver='saga', penalty='elasticnet',max_iter=10000,l1_ratio=0.5)

# Alpha values
parameters = {"C":[1e-4, 1e-2, 1, 5, 10, 20]}

# Grid search 
elastic_regression = GridSearchCV(logreg, parameters, scoring='accuracy', cv=5)

# Train using the Xtrain
elastic_train=elastic_regression.fit(X_train_norm, Y_train)

print(elastic_regression.best_params_)
print(elastic_regression.best_score_)

# Compute the predicted output
y_pred_elastic = elastic_regression.predict(X_test_norm)

#  Compute the confusion matrix
cm = confusion_matrix(Y_test, y_pred_elastic)
print(cm)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_elastic).ravel()
print(f'tn: {tn}, fp:{fp}, fn:{fn}, tp:{tp}')

# Compute the accuracy_score
Accuracy=accuracy_score(Y_test, y_pred_elastic)

target_names = ['Non-obese', 'Obese']
print(classification_report(Y_test, y_pred_elastic, target_names=target_names))
              
#Sensibility
Sensibility=recall_score(Y_test, y_pred_elastic)


# Compute the AUC and the ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_elastic)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

Specificity = tn/(tn+fp)
Precision = tp/(tp+fp) 
F1_score = f1_score(Y_test, y_pred_elastic)

print('''Summing up:
- Accuracy: {}
- Sensibility/recall: {}
- Specificity: {}
- Precision: {}
- F1 score: {}
'''.format(Accuracy,Sensibility,Specificity,Precision,F1_score))


# 
# ## <center><b><font style="color: darkblue" size="5"> 3.4-COMPARISON OF THE RESULTS OBTAINED WITH THE DIFFERENT MODELS </font></b></center>
# 
# 
# As discussed before, the results obtained along the parametric classification analysis have been very good. After data and predictions inspection, and various reviews of the coding content, we can conclude that our database was perfect to be analyzed by classification. Furthermore, calculating the BMI was a great decision because subjects can be almost perfectly sorted into obese vs. non-obese by using just the BMI.
# 
# There was not a big difference between the different classification methods, but it is safe to say that either Multivariate Logistic Regression or the Logistic Regression (with both Lasso and Ridge) with Regularization would be the most suitable models to predict, since the first learns from a wider range of information, and the latter penalizes possible overfit predictions.

# In[ ]:




