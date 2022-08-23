#!/usr/bin/env python
# coding: utf-8

# # 1. Clean the Data

# In[1]:


import pandas as pd

hd_df_uncleaned = pd.read_csv ('heart.csv')
print(hd_df_uncleaned)


# In[2]:


#Update target variable such that (0=no heart disease, 1= heart disease). It is currently inversed.

#Set 0 to 2 temporarily
hd_df_uncleaned['target'] = hd_df_uncleaned['target'].replace([0],2)
#Set 1 to 0 (no disease)
hd_df_uncleaned['target'] = hd_df_uncleaned['target'].replace([1],0)
#Set 2 to 1 (disease))
hd_df_uncleaned['target'] = hd_df_uncleaned['target'].replace([2],1)


# In[3]:


#Examine the data for null values
hd_df = hd_df_uncleaned.drop_duplicates()
#hd_df=hd_df_uncleaned
hd_df.info()


# # 1. EDA

# ### age
# 
# Age defines how old a person is.

# In[4]:


hd_df['age'].describe()


# 

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc = {'figure.figsize':(15,8)})


# In[6]:


sns.boxplot(x='target', y='age', data=hd_df)
plt.title("age vs. target",fontsize=20)
plt.ylabel('age (years)')
plt.show()


# Based on this, it seems like there really isn't any distinction between ages where one would be expected to have a heart attack. However, the older you are, you are slightly more likely to have the disease.

# ### Sex
# 
# * Value 0: Female
# * Value 1: Male

# In[7]:


sns.countplot(x='sex', hue='target', data=hd_df)
plt.title("sex vs. target",fontsize=20)
plt.show()


# While there are more males in this dataset, the data suggests that that males are more likely to have heart disease comapred to females.

# ### cp: chest pain type
# 
# This is a variable representing chest pain (CP). There are four levels of chest pain defined as follows:
# 
# * Value 0: asymptomatic 
# * Value 1: atypical angina
# * Value 2: non-anginal pain
# * Value 3: typical angina

# In[8]:


sns.countplot(x='cp', hue='target', data=hd_df)
plt.title("chest pain type vs. target",fontsize=20)
plt.show()


# NOTE: Angina pectoris or typical angina is the discomfort that is noted when the heart does not get enough blood or oxygen. 

# In[9]:


#
hd_df = hd_df.replace(to_replace={'cp': {0: 4}}, value=None)
hd_df = hd_df.replace(to_replace={'cp': {1: 0}}, value=None)
hd_df = hd_df.replace(to_replace={'cp': {2: 0}}, value=None)
hd_df = hd_df.replace(to_replace={'cp': {3: 0}}, value=None)
hd_df = hd_df.replace(to_replace={'cp': {4: 1}}, value=None)


# In[10]:


sns.countplot(x='cp', hue='target', data=hd_df)
plt.title("chest pain type vs. target",fontsize=20)
plt.show()


# ### Trestbps
# 
# Resting blood pressure (in mm Hg on admission to the hospital) 

# In[11]:


sns.boxplot(x='target', y='trestbps', data=hd_df)
plt.title("resting blood pressure vs. target",fontsize=20)
plt.ylabel('resting blood pressure (mm Hg)')
plt.show()


# ### Chol
# 
# serum cholestoral in mg/dl 

# In[12]:


sns.boxplot(x='target', y='chol', data=hd_df)
plt.title("serum cholestoral vs. target",fontsize=20)
plt.ylabel('serum cholestoral (mg/dl)')
plt.show()


# ### fbs
# 
# fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 

# In[13]:


sns.countplot(x='fbs', hue='target', data=hd_df)
plt.title("fasting blood sugar (> 120 mg/dl) vs. target",fontsize=20)
plt.show()


# There does not appear to be any strong indicators of wheter or not a higher fasting blood sugar means a greater likelihood of heart disease.

# ### restecg
# 
# resting electrocardiographic (ECG) results:
# 
# * Value 0: showing probable or definite left ventricular hypertrophy by Estes’ criteria
# * Value 1: normal
# * Value 2: having ST-T wave abnormality 

# In[14]:


sns.countplot(x='restecg', hue='target', data=hd_df)
plt.title("resting ECG vs. target",fontsize=20)
plt.show()


# Based on the data description, we can see we should set 0=normal and combine both the Estes' criteri and ST-T wave abornamlity together (i.e. abnormal ECG results).

# In[15]:


hd_df = hd_df.replace(to_replace={'restecg': {0: 2}}, value=None)
hd_df = hd_df.replace(to_replace={'restecg': {1: 0}}, value=None)
hd_df = hd_df.replace(to_replace={'restecg': {2: 1}}, value=None)


# In[16]:


sns.countplot(x='restecg', hue='target', data=hd_df)
plt.title("resting ECG vs. target",fontsize=20)
plt.show()


# We now have 0 representing normal ECG data and 1 representing abnormal ECG data.

# ### thalach
# 
# maximum heart rate achieved 

# In[17]:


sns.boxplot(x='target', y='thalach', data=hd_df)
plt.title("maximum heart rate (thalach) vs. target",fontsize=20)
plt.show()


# It seems like lower heart rates appear to be correlated to a greater likelihood of heart disease.

# ### exang
# 
# exercise induced angina (1 = yes; 0 = no)

# In[18]:


sns.countplot(x='exang', hue='target', data=hd_df)
plt.title("exercise induced angina (exang) vs. target",fontsize=20)
plt.show()


# If seems that if one has exercised induced agina, they have a higher likelihood of heart disease. 

# ### oldpeak
# 
# ST depression induced by exercise relative to rest
# 
# Note: ST depression refers to a finding on an electrocardiogram, wherein the trace in the ST segment is abnormally low below the baseline (https://en.wikipedia.org/wiki/ST_depression)

# In[19]:


hd_df['oldpeak'].describe()


# In[20]:


sns.boxplot(x='target', y='oldpeak', data=hd_df)
plt.title("oldpeak vs. target",fontsize=20)
plt.ylabel('ST Depression (mm)')
plt.show()


# Higher ST depressions appears to show a higher chance of heart disease.

# ### slope
# 
# slope: the slope of the peak exercise ST segment
# 
# * 0: downsloping; 
# * 1: flat; 
# * 2: upsloping

# In[21]:


sns.countplot(x='slope', hue='target', data=hd_df)
plt.title("slope vs. target",fontsize=20)
plt.show()


# Those with downsloping in their exercise ST segment appear to have a higher probability of heart disease. According to ECG information, upsloping ST segments are the normal segments seen during physical exercise. Horizontal (flat) or downsloping are abnormal and caused by ischemia, a condition in which the blood flow (and thus oxygen) is restricted or reduced in a part of the body.
# 
# We should group the abnormal ST segment data together and make that a distinct group from the normal (upsloping) ST segments. We define 0=upsloping, and 1=flat or downsloping.

# In[22]:


hd_df = hd_df.replace(to_replace={'slope': {2: 3}}, value=None)
hd_df = hd_df.replace(to_replace={'slope': {0: 1}}, value=None)
hd_df = hd_df.replace(to_replace={'slope': {3: 0}}, value=None)


# In[23]:


#Replot
sns.countplot(x='slope', hue='target', data=hd_df)
plt.title("slope vs. target",fontsize=20)
plt.show()


# From a marginal effects perspective, this makes much more sense.

# ### ca
# 
# number of major vessels (0-3) colored by flourosopy. Fluoroscopy is a medical procedure that makes a real-time video of the movements inside a part of the body by passing x-rays through the body over a period of time.
# 
# NOTE: ca values of 4 are NULLS (remove them)

# In[24]:


sns.countplot(x='ca', hue='target', data=hd_df)
plt.show()


# Let's remove the invalid value 4.

# In[25]:


#Remove invalid ca values
hd_df = hd_df[hd_df.ca != 4]

sns.countplot(x='ca', hue='target', data=hd_df)
plt.title("ca vs. target",fontsize=20)
plt.show()


# It would seem like being unable to see any major blood vessals via floursopy suggests lower rates of heart disease.

# In[26]:


hd_df = hd_df.replace(to_replace={'ca': {2: 1}}, value=None)
hd_df = hd_df.replace(to_replace={'ca': {3: 1}}, value=None)

sns.countplot(x='ca', hue='target', data=hd_df)
plt.title("ca vs. target",fontsize=20)
plt.show()


# Since these values do mean something on an ordinal scale, we will not make any adjustments.

# ### thal
# 
# * Value 0: NULL (dropped from the dataset previously)
# * Value 1: fixed defect (no blood flow in some part of the heart)
# * Value 2: normal blood flow
# * Value 3: reversible defect (a blood flow is observed but it is not normal)

# In[27]:


sns.countplot(x='thal', hue='target', data=hd_df)
plt.show()


# Let's first drop the invalid values.

# In[28]:


#Remove invalid thal values
hd_df = hd_df[hd_df.thal != 0]

sns.countplot(x='thal', hue='target', data=hd_df)
plt.show()


# Again, we have to deal with the fact that there is not an ordinal scale to go from normal blood flow to fixed defect to reversible defect. again, we have normal as a separate category

# In[29]:


hd_df = hd_df.replace(to_replace={'thal': {2: 0}}, value=None) #Normal
hd_df = hd_df.replace(to_replace={'thal': {3: 1}}, value=None)

sns.countplot(x='thal', hue='target', data=hd_df)
plt.title("thal vs. target",fontsize=20)
plt.show()


# It appears that by not having normal blood flow means higher risks for heart disease.

# Our dataset is now properly prepared for use.

# # 3. Logistic Regression & Bootstrapping

# ### Create Logit Model

# In[30]:


#For creating a train-test split
from sklearn.model_selection import train_test_split

#For the logitic regression algorithm
import statsmodels.api as sm
from statsmodels.formula.api import logit


# In[31]:


import random 

#Set random seed for reproducibility
random.seed(10)


# In[32]:


#Creating a training data set and a test data set from the overall heart disease dataset
train_data, test_data = train_test_split(hd_df, test_size=0.20, random_state=42)


# In[33]:


#All variable model
hd_formula = ('target ~ age + sex + cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal')
logit_model = logit(formula = hd_formula, data=train_data).fit()


# In[34]:


logit_model.summary()


# We now create another logistic regression model with certain variables removed

# In[35]:


hd_formula_B = ('target ~ sex + cp + trestbps + fbs + slope + ca + thal')
logit_model_B = logit(formula = hd_formula_B, data=train_data).fit()


# In[36]:


logit_model_B.summary()


# In[37]:


hd_formula_C = ('target ~ sex + cp + trestbps + slope + ca + thal')
logit_model_C = logit(formula = hd_formula_C, data=train_data).fit()

logit_model_C.summary()


# In[38]:


hd_formula_D = ('target ~ sex + cp + slope + ca + thal')
logit_model_D = logit(formula = hd_formula_D, data=train_data).fit()

logit_model_D.summary()


# ### Check Accuracy

# In[39]:


import numpy as np
from sklearn.metrics import accuracy_score

#Create a function to calculate the accuracy for a model using the test data
def accuracy_check(model, test_data):
    prediction = model.predict(exog = test_data)
    y_prediction = np.where(prediction > 0.5,1,0)
    y_actual = test_data['target']
    
    return(accuracy_score(y_actual, y_prediction))


# In[40]:


print('Accuracy of All Variable Models: ', accuracy_check(logit_model, test_data))
print('Accuracy of Model B: ', accuracy_check(logit_model_B, test_data))
print('Accuracy of Model C: ', accuracy_check(logit_model_C, test_data))
print('Accuracy of Model D: ', accuracy_check(logit_model_D, test_data))


# ### Base Odds Ratio

# In[41]:


odds_ratio_base = np.exp(logit_model.params)
odds_ratio_model_D = np.exp(logit_model_D.params)

print("\n odds ratios for all variable model:")
print(odds_ratio_base)
print("\n odds ratios for reduce variable model:")
print(odds_ratio_model_D)


# ### Base Marginal Effects

# In[42]:


#Calculate average marginal effects
AME_full = logit_model.get_margeff(at='overall', method='dydx')
AME_part = logit_model_D.get_margeff(at='overall', method='dydx')

print(AME_full.summary())
print(AME_part.summary())


# ### Bootstrap Phase

# In[43]:


#Create empty dataframes for odds ratios
odds_df_full = pd.DataFrame(columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                                  'exang', 'oldpeak', 'slope', 'ca', 'thal'])
odds_df_part = pd.DataFrame(columns = ['sex', 'cp', 'slope', 'ca', 'thal'])

#Create empty dataframes for marginal effects
margeff_df_full = pd.DataFrame(columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                                  'exang', 'oldpeak', 'slope', 'ca', 'thal'])
margeff_df_part = pd.DataFrame(columns = ['sex', 'cp', 'slope', 'ca', 'thal'])


# In[44]:


#Function to generate bootstrap samples and 
def bootstrap_full(logit_formula, stats_df, full=0, stats_type=0, n=2000):
    for i in range(n):
        #Generate the bootstrap sample
        random.seed(123)
        sample = hd_df.sample(n = len(hd_df), replace=True)
        #Recreate the logit model for this new sample, but we skip the testing/training
        logit_model = logit(formula = logit_formula, data=sample).fit(disp=0)
        #Calculate the statitsics
        temp = None
        #For odds ratio or average marginal effects
        if (stats_type==0):
            temp = np.exp(logit_model.params)
            new_row = {'age': temp[1], 'sex': temp[2], 'cp': temp[3], 'trestbps': temp[4],'chol': temp[5], 
                'fbs': temp[6], 'restecg': temp[7], 'thalach': temp[8],'exang': temp[9], 
                'oldpeak': temp[10], 'slope': temp[11], 'ca': temp[12], 'thal': temp[13]}
            stats_df = stats_df.append(new_row, ignore_index = True)
        else:
            bts_AME = logit_model.get_margeff(at='overall', method='dydx')
            temp = bts_AME.summary_frame()['dy/dx']
            new_row = {'age': temp[0], 'sex': temp[1], 'cp': temp[2], 'trestbps': temp[3],'chol': temp[4], 
                'fbs': temp[5], 'restecg': temp[6], 'thalach': temp[7],'exang': temp[8], 
                'oldpeak': temp[9], 'slope': temp[10], 'ca': temp[11], 'thal': temp[12]}
            stats_df = stats_df.append(new_row, ignore_index = True)
            
    return(stats_df)


# In[45]:


def bootstrap_part(logit_formula, stats_df, stats_type=0, n=2000):
    for i in range(n):
        #Generate the bootstrap sample
        random.seed(456)
        sample = hd_df.sample(n = len(hd_df), replace=True)
        #Recreate the logit model for this new sample, but we skip the testing/training
        logit_model = logit(formula = logit_formula, data=sample).fit(disp=0)
        #Calculate the statitsics
        temp = None
        #For odds ratio or average marginal effects
        if (stats_type==0):
            temp = np.exp(logit_model.params)
            new_row = {'sex': temp[1], 'cp': temp[2], 'slope': temp[3], 'ca': temp[4], 'thal': temp[5]}
            stats_df = stats_df.append(new_row, ignore_index = True)
        else:
            bts_AME = logit_model.get_margeff(at='overall', method='dydx')
            temp = bts_AME.summary_frame()['dy/dx']
            new_row = {'sex': temp[0], 'cp': temp[1], 'slope': temp[2], 'ca': temp[3], 'thal': temp[4]}
            stats_df = stats_df.append(new_row, ignore_index = True)
            
    return(stats_df)


# In[46]:


full_formula = ('target ~ age + sex + cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal')
or_full = bootstrap_full(full_formula, odds_df_full, stats_type=0)
margeff_full = bootstrap_full(full_formula, margeff_df_full, stats_type=1)


# In[47]:


part_formula = ('target ~ sex + cp + slope + ca + thal')
or_part = bootstrap_part(part_formula, odds_df_part, stats_type=0)
margeff_part = bootstrap_part(part_formula, margeff_df_part, stats_type=1)


# Having completed running the bootstrap to generate new odds-ratio and average marginal effects, we should now analyze these samples. We first analyze the odds-ratio data.

# In[48]:


import scipy.stats as st
full_vars = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
part_vars = ['sex', 'cp', 'slope', 'ca', 'thal']


# In[49]:


print('odds ratios (all variables)')
for j in range(len(full_vars)):
    print("\nBootstrap Mean for " + full_vars[j]  + " : ", str.format('{0:.3f}', np.mean(or_full.iloc[:, j])))
    ci = st.norm.interval(alpha=0.95, loc=np.mean(or_full.iloc[:, j]), scale=st.sem(or_full.iloc[:, j]))
    print("95% BCI for " + full_vars[j]  + " : (" + str.format('{0:.3f}', ci[0]) + ", " + str.format('{0:.3f}', ci[1]) + ")")


# In[50]:


print('odds ratios (partial variables)')
for j in range(len(part_vars)):
    print("\nBootstrap Mean for " + part_vars[j]  + " : ", str.format('{0:.3f}', np.mean(or_part.iloc[:, j])))
    ci = st.norm.interval(alpha=0.95, loc=np.mean(or_part.iloc[:, j]), scale=st.sem(or_part.iloc[:, j]))
    #print("95% BCI for " + hd_vars[j]  + " : ", ci)
    print("95% BCI for " + part_vars[j]  + " : (" + str.format('{0:.3f}', ci[0]) + ", " + str.format('{0:.3f}', ci[1]) + ")")


# In[51]:


print('marginal effects (all variables)')
for j in range(len(full_vars)):
    print("\nBootstrap Mean for " + full_vars[j]  + " : ", str.format('{0:.3f}', np.mean(margeff_full.iloc[:, j])))
    ci = st.norm.interval(alpha=0.95, loc=np.mean(margeff_full.iloc[:, j]), scale=st.sem(margeff_full.iloc[:, j]))
    #print("95% BCI for " + hd_vars[j]  + " : ", ci)
    print("95% BCI for " + full_vars[j]  + " : (" + str.format('{0:.3f}', ci[0]) + ", " + str.format('{0:.3f}', ci[1]) + ")")


# In[52]:


print('marginal effects (partial variables)')
for j in range(len(part_vars)):
    print("\nBootstrap Mean for " + part_vars[j]  + " : ", str.format('{0:.3f}', np.mean(margeff_part.iloc[:, j])))
    ci = st.norm.interval(alpha=0.95, loc=np.mean(margeff_part.iloc[:, j]), scale=st.sem(margeff_part.iloc[:, j]))
    #print("95% BCI for " + hd_vars[j]  + " : ", ci)
    print("95% BCI for " + part_vars[j]  + " : (" + str.format('{0:.3f}', ci[0]) + ", " + str.format('{0:.3f}', ci[1]) + ")")

