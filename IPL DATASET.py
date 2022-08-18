#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ipl_auction_df=pd.read_csv('https://raw.githubusercontent.com/Foridur3210/IPL-Dataset-Player-price-prediction/master/IPL%20IMB381IPL2013.csv')


# In[3]:


ipl_auction_df


# In[4]:


type(ipl_auction_df)


# In[5]:


# That is,ipl_auction_df is of 'type DATAFRAME'


# In[6]:


# we will only print maximum of 7 columns as the total width exceeds the page width and display is distorted.


# In[7]:


pd.set_option('display.max_columns',7)


# In[8]:


ipl_auction_df.head(5)


# In[9]:


list(ipl_auction_df.columns)


# In[10]:


# Other way to print daatframe with large no of columns is to TARNSPOSE dataframe and display row indexes as columns and column
# names as row indexes 


# In[13]:


ipl_auction_df.head(5).transpose()


# In[14]:


ipl_auction_df.shape


# In[15]:


# 130 records and 26 columns.


# In[31]:


ipl_auction_df.info()


# In[32]:


# slicing and indexing 
# Dataframes can be sliced or accessed by index or names. 
# The row or column index always start with 0.
# Indexing by default always starts with 0


# In[33]:


ipl_auction_df[0:5]


# In[34]:


ipl_auction_df[5:10]


# In[35]:


# Negative indexing is an excellent feature in pythin used to select records from the bottom of the dataframe.


# In[36]:


ipl_auction_df[-5:]


# In[37]:


# Specific columns of a dataframe can also be selected or sliced by column names.


# In[38]:


ipl_auction_df['PLAYER NAME'][0:10]


# In[39]:


# To selct two column, pass a list of column names to the dataframe.


# In[40]:


ipl_auction_df[['PLAYER NAME','COUNTRY']][0:5]


# In[41]:


# Specific rows and columns can also be selected using row and column index.
# Using ILOC ( index location) method of dataframe.
# remember here that inside iloc ,it takes row index as first parameter and column ranges as second parameter.


# In[42]:


ipl_auction_df.iloc[4:9,1:4]


# In[43]:


# Value_counts() provides occurence of each unique value in a column.
# Note that Value_counts() should primarily be used for categorical variables.
# Question hare is how many players from different countries have played in ipl.


# In[44]:


ipl_auction_df.COUNTRY.value_counts()


# In[45]:


# Passing parametr normalize=True to value_counts() will calculate the percentage of occurence of each unique value.


# In[46]:


ipl_auction_df.COUNTRY.value_counts(normalize=True)*100


# In[47]:


# crosstab() ( CROSS-TABULATION) features will help in find occurence for the combination of values for two columns.


# In[48]:


pd.crosstab(ipl_auction_df['AGE'],ipl_auction_df['PLAYING ROLE'])


# In[49]:


ipl_auction_df[['PLAYER NAME','SOLD PRICE']].sort_values('SOLD PRICE')[0:5]


# In[50]:


# False to ascending parameter will  sort data in descending order.
# By default it sort in ascending order.


# In[51]:


ipl_auction_df[['PLAYER NAME','SOLD PRICE']].sort_values('SOLD PRICE',ascending=False)[0:5]


# In[52]:


ipl_auction_df[['PLAYER NAME','SOLD PRICE']].sort_values('SOLD PRICE')[-5:]


# In[53]:


ipl_auction_df[['PLAYER NAME','SOLD PRICE']].sort_values('SOLD PRICE',ascending=False)[-5:]


# In[54]:


# Creating new columns.


# In[55]:


ipl_auction_df['PREMIUM']=ipl_auction_df['SOLD PRICE']-ipl_auction_df['BASE PRICE']


# In[56]:


ipl_auction_df[['PLAYER NAME','BASE PRICE','SOLD PRICE','PREMIUM']][0:5]


# In[57]:


# Find which players got the maximum premium offering on their base price.


# In[58]:


ipl_auction_df[['PLAYER NAME','BASE PRICE','SOLD PRICE','PREMIUM']].sort_values('PREMIUM',ascending=False )[0:5]


# In[59]:


# Grouping and aggregating.
#group all records by age and then find average sold price column.


# In[60]:


soldprice_by_age=ipl_auction_df.groupby('AGE')['SOLD PRICE'].mean()
soldprice_by_age


# In[61]:


# Above operation returns a' pd.series data 'structure.To convert into dataframe ,use reset_index() as shown below.


# In[62]:


soldprice_by_age=ipl_auction_df.groupby('AGE')['SOLD PRICE'].mean().reset_index()
soldprice_by_age


# In[63]:


# Multiple column can be passed to 'Groupby'.


# In[64]:


soldprice_by_age_role=ipl_auction_df.groupby(['AGE','PLAYING ROLE'])['SOLD PRICE'].mean().reset_index()
soldprice_by_age_role


# In[65]:


#Joining dataframes using 'merge()'.
# such case both dataframes must have a common.
# Join types can have inner,outer,left,right specified in how parameter.


# In[66]:


soldprice_comparision=soldprice_by_age_role.merge(soldprice_by_age,on='AGE',how='outer')
soldprice_comparision


# In[67]:


# Because column name sold price is same in both the dataframe,it automatically renames them to _x and _y
# sold_price_x comes from left table(soldprice_by_age_role) and sold_price_y comes from right table(soldprice_by_age)


# In[68]:


soldprice_comparision=soldprice_by_age_role.merge(soldprice_by_age,on='AGE',how='inner')
soldprice_comparision


# In[69]:


# Renaming multiple columns uses dictionary as a parameter where key is existing column name and value is new names assigned.


# In[70]:


soldprice_comparision.rename(columns={'SOLD PRICE_x':'SOLD_PRICE_AGE_ROLE','SOLD PRICE_y':'SOLD_PRICE_AGE'},inplace=True)
soldprice_comparision.head(5)


# In[71]:


# Applying operations on multiple columns using function apply() along any axix.
#Find whether players carry a premium if they belong to specific AGE and PLAYING ROLE.Premium here is percentage change .


# In[72]:


soldprice_comparision['CHANGE']=soldprice_comparision.apply(lambda x:(x.SOLD_PRICE_AGE_ROLE-x.SOLD_PRICE_AGE)/x.SOLD_PRICE_AGE,axis=1)
soldprice_comparision


# In[73]:


# Filtering records based on conditions.
#Find no of players hitting more than 80 sixes.


# In[74]:


ipl_auction_df[ipl_auction_df.SIXERS>80]


# In[75]:


ipl_auction_df[ipl_auction_df['SIXERS']>80]


# In[76]:


ipl_auction_df[ipl_auction_df['SIXERS']>80][['PLAYER NAME','SIXERS']]


# In[77]:


import seaborn as sns


# In[78]:


# Bar chart is a frequency for qualitative variable(or categorical variable).It is to assess most and the least occuring 
#categories within dataset.It uses barplot() function.


# In[79]:


sns.barplot(x='AGE',y='SOLD PRICE',data=soldprice_by_age)


# In[80]:


sns.barplot(x='AGE',y='SOLD PRICE',data=soldprice_by_age_role,hue='PLAYING ROLE')


# In[81]:


sns.barplot(x='AGE',y='SOLD_PRICE_AGE_ROLE',data=soldprice_comparision,hue='PLAYING ROLE')


# In[82]:


#Histogram is a plot shows frequency distribution of a set of continuous variables.It uses hist() method of matplotlib.
# Histogram gives an insight into the underlying distribution(normal distribution)of variable,outliers,skewness etc.
# Skewness is assymetry in a sttistical distribution , in which curve appears distorted or skewed either to right or left. 
#  By default it creates 10 bins in histogram.


# In[83]:


plt.hist(ipl_auction_df['SOLD PRICE'],bins=20)


# In[84]:


# insights of above histogram-
#Sold price is right skewed.
# most players are auctioned at low price range of 250000 and 500000,very few players are paid highly more than 1 million dollar.


# In[85]:


# Distribution or density plot depicts distribution of data over a continuous interval. 
# It gives insights into what might be the distribution of population.
# we use distplot() of seaborn.


# In[86]:


sns.distplot(ipl_auction_df['SOLD PRICE'])


# In[87]:


# Box plot ( whisker plot) is a graphical repressentation of numerical data that can be understand the variability of data and 
#existence of outliers
# Box plot is constructed using minimum,maximum,IQR values
# Length of box is equivalent to IQR,which is distance between 1st quartile(25 percentile) and 3rd quartile(75 percentile).


# In[ ]:





# In[88]:


box=sns.boxplot(ipl_auction_df['SOLD PRICE'])


# In[89]:


# Above plot indicates there are few outliers .


# In[90]:


box=plt.boxplot(ipl_auction_df['SOLD PRICE'])


# In[91]:


# Caps key in box variable returns max and min values of distribution.
# Whiskers key in box variable returns values of distribution at 25 and 75 quantiles.


# In[92]:


[item.get_ydata()[0] for item in box['whiskers']]


# In[93]:


[item.get_ydata()[0] for item in box['caps']]


# In[94]:


# So inter quartile range is 700000-225000=450000


# In[95]:


[item.get_ydata()[0] for item in box['medians']]


# In[96]:


# Let us find out names of outliers using condition.


# In[97]:


ipl_auction_df[ipl_auction_df['SOLD PRICE']>1350000][['PLAYER NAME','PLAYING ROLE','SOLD PRICE']]


# In[98]:


## COMPARING DISTRIBUTIONS.


# In[99]:


box=sns.boxplot(x='PLAYING ROLE',y='SOLD PRICE',data=ipl_auction_df)


# In[100]:


# observations from above plot:
# The median SOLD PRICE for allrounders and batsmans are higher than bowlers and wicketkeepers.
# Alrounders paid more than 1350000 USD are not considered outlier.Allrounder have high variance.
# There are outliers in batsman and w keeper category


# In[101]:


import seaborn as sns


# In[102]:


sns.distplot(ipl_auction_df[ipl_auction_df['CAPTAINCY EXP']==1]['SOLD PRICE'],color='y',label='Captaincy Experience')
sns.distplot(ipl_auction_df[ipl_auction_df['CAPTAINCY EXP']==0]['SOLD PRICE'],color='r',label='No Captaincy Experience')
plt.legend()


# In[103]:


# SCATTER PLOT, in this 2 variables are plotted along two axes and resulting pattern can reveal relation present btw the two,if any.
# It can be linear or non linear relation btw the two.
# useful for assessing strength of relationship and to find if there is any outlier in the data
# Used during regression model building to decide on the initial model, that is to include variable in a regression model or not.
# It uses matplotlib.


# In[104]:


ipl_batsman_df=ipl_auction_df[ipl_auction_df['PLAYING ROLE']=='Batsman']
ipl_batsman_df


# In[105]:


ipl_batsman_df.SIXERS.reset_index()


# In[106]:


plt.scatter(x=ipl_batsman_df['SIXERS'],
           y=ipl_batsman_df['SOLD PRICE'])


# In[107]:


# regplot() of seaborn can be used to draw direction of relationship btw the two variables.
#regplot must be 1D.


# In[108]:


sns.regplot(x='SIXERS',
           y='SOLD PRICE',
           data=ipl_batsman_df)


# In[109]:


# above plot tells there is a positive correlation between number of sixes hit by a batsman and sold price.


# In[ ]:





# In[110]:


sns.pairplot(ipl_auction_df[['SR-B','AVE','SIXERS','SOLD PRICE']])


# In[111]:


# Correlation is used for measuring strength and direction of linear relationship btw two continuous random variables .
# It is statistical measure that indicates extent to which two variables change .
# correlation value lies btw -1 and 1.
# 1 indicates perfect positive correlation.positive correlation indicates both variables increase or decrease together.
#-1 indicates perfect  neagtive correlation .negative correlation indicates if one variable increase then other increases.
#  corr() method of dataframe computes correlation values.


# In[112]:


ipl_auction_df[['SR-B','AVE','SIXERS','SOLD PRICE']].corr()


# In[113]:


# Using heatmap() from seaborn gives color map scale.
# annot attribute to true prints correlation values in each box of heatmap and improves readability of heatmap.


# In[114]:


sns.heatmap(ipl_auction_df[['SR-B','AVE','SIXERS','SOLD PRICE']].corr(),annot=True)


# In[115]:


# insights from above plot :
# ave and sixers are strong positive correlation
# sr-b and sold price are not strongly correlated.


# In[116]:


# Using multiple linear regression model.
# Multiple linear regression model is a supervised learning algorithm for finding the existence  of a association relationship
#between a dependnt variable(outcome variable) and several independent variables(feature or predictor variable)


# In[117]:


#Loading the dataset


# In[118]:


ipl_auction_df=pd.read_csv('https://raw.githubusercontent.com/Foridur3210/IPL-Dataset-Player-price-prediction/master/IPL%20IMB381IPL2013.csv')


# In[119]:


ipl_auction_df.info()


# In[120]:


ipl_auction_df.head(5)


# In[121]:


# df.iloc() is used for displaying a subset of the dataset
ipl_auction_df.iloc[0:5,0:10]


# In[122]:


# We can build a model to understand what features of players are influencing their SOLD PRICE  or predict players auction
#prices in future.
# S.no is not any feature of the player hence cant be cont be considered for building model.We are building model using only
#players statistics.So Base price can also be removed.
# We will create a variable X_features which will contain list of features that we will finally use for building model and 
#ignore rest of the columns.


# In[123]:


X_features = ipl_auction_df.columns
X_features


# In[124]:


X_features=['AGE', 'COUNTRY', 'PLAYING ROLE',
       'T-RUNS', 'T-WKTS', 'ODI-RUNS-S', 'ODI-SR-B', 'ODI-WKTS', 'ODI-SR-BL',
       'CAPTAINCY EXP', 'RUNS-S', 'HS', 'AVE', 'SR-B', 'SIXERS', 'RUNS-C',
       'WKTS', 'AVE-BL', 'ECON', 'SR-BL']


# In[125]:


# In the above features there are some qualitative or categorical features and hence need to be encoded before building
#the model.
# Categorical variables cant be directly included in the regression model, and hence must be encoded using dummy variables
#before incorporating in model building.


# In[126]:


# USING DUMMY VARIABLES
# If a categorical variable has n categories then we will need n-1 dummy variables.
# Examople: PLAYING ROLE has 4 vcategories then dummy variables required will be 3.
# Set the variable value to 1 to indicate the role of the player.Use pd.get_dummies for this.


# In[127]:


ipl_auction_df['PLAYING ROLE'].unique()


# In[128]:


pd.get_dummies(ipl_auction_df['PLAYING ROLE'])[0:5]


# In[129]:


# We create only (n-1) dummy variables is that inclusion of dummy variables for all categories and the constant in the 
#regression equation will create multi-collinearity .
# To drop one category , the parameter drop_first should be set to True.


# In[130]:


categorical_features=['AGE','COUNTRY','PLAYING ROLE','CAPTAINCY EXP']


# In[131]:


ipl_auction_encoded_df=pd.get_dummies(ipl_auction_df)
ipl_auction_encoded_df


# In[132]:


ipl_auction_encoded_df=pd.get_dummies(ipl_auction_df[X_features],columns=categorical_features,drop_first=True)
ipl_auction_encoded_df


# In[133]:


#The dataset contains new dummy variables that have been created. We can reassign new features to variable X_features to keep
#record of all features that will be used to build model finally.


# In[134]:


X_features=ipl_auction_encoded_df.columns
X_features


# In[135]:


# SPLITTING DATASET INTO TRAIN AND VALIDATION SETS


# In[136]:


# Before building model,split dataset into 80:20 ratio.The split function allows using a parameter "random_state",which is seed
#function for reproducibility of randomness.Setting parameter to a fixed number will make sure records that go into training 
#and test set remain unchanged and can be reproduced.Here we are using random state as 42 for reproducibility of results.
#Using different random seed may give different training and test data and hence different results.


# In[137]:


# statsmodel library is used in python for building statistical model
# OLS API available in statsmodel.api is used for estimation of simple linear regression model. The OLS() model takes two 
#parameters X and Y.OLS API estimates coefficient of only X parametr. To estimate regression coefficient ( β) ,a constant term 1
#needs to be added as a seperate column.As value of columns remains same across all samples,parameter set for this column
#will be intercept term.


# In[138]:


import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# In[139]:


X=sm.add_constant(ipl_auction_encoded_df)
X


# In[140]:


X=sm.add_constant(ipl_auction_encoded_df)
Y=ipl_auction_df['SOLD PRICE']

train_X, test_X, train_Y, test_Y = train_test_split(X,Y,train_size=0.8, random_state=42)


# In[141]:


# 0.8 as train size implies 80% of data is used for training model and 20% used for validating data


# In[142]:


# Building model on training dataset.


# In[143]:


ipl_model_1= sm.OLS(train_Y,train_X).fit()
ipl_model_1


# In[144]:


ipl_model_1.summary()


# In[145]:


# Insights of above MLR model output:
#1.As per p-value(<0.05),only features(HS,AGE_2,AVE and COUNTRY_ENG ) have come out significant .The model says that none other
#features are influencing SOLD PRICE( at significance value of 0.05).This is not very intutive and could be a result of 
#multi-collinearity effect of variables.


# In[146]:


# When dataset has large no of independent variables (feeatures) , it is possible that few of these independent variables
#may be highly correlated. The existence of highly correlation btw independent variables is called multi-collinearity.It can
#destabilize the MLR model.Thus,necessary to take corrective measures.


# In[147]:


# Multi-collinearity can have following impact on model:
#1.The standard error of estimate ,() is inflated.
#The standard error of the regression (S), also known as the standard error of the estimate,
#represents the average distance that the observed values fall from the regression line.

#2.A statistically significant explanatory variable may be labelled as statistically insignificant due to the large p-value.
#This is because when standard error of estimate inflated,it results in an underestimation of t-static value.
#In statistics, the t-statistic is the ratio of the departure of the estimated value of a parameter from its
#hypothesized value to its standard error.

#3.The sign of regression coefficient may be different , instead of positive value we have negative value for regression
#cofficient and vice versa.

#4. Adding /removing a variable or even an observation may result in large variation in regression coefficient estimates.


# In[148]:


# Variance inflation Factor is a measure used for identifying  the existence of multi collinearity .
# VIF= 1/(1-(sqr of R-squared value)) of the model.VIF value >4 requires further investigation to assess of multiple
#collinearity.
# One method to eliminate multi collinearity is to remove one of variable from model bilding.
# under root of VIF value by which t-static value is deflated.
# variance_inflation_factor() method available in statsmodels.stats.outliers_influence package.


# In[149]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
def get_vif_factors(X):
    X_matrix = X.to_numpy()
    vif=[variance_inflation_factor(X_matrix,i) for i in range(X_matrix.shape[1])]
    vif_factors = pd.DataFrame()
    vif_factors['column'] = X.columns
    vif_factors['VIF'] = vif
    return vif_factors


# In[150]:


#to_numpy() function is used to return a NumPy ndarray representing the values in given Series or Index.
#This function will explain how we can convert the pandas Series to numpy Array.


# In[151]:


vif_factors=get_vif_factors(X[X_features])
vif_factors


# In[152]:


# To check correlation of columns with large VIFs
#WE can generate heatmap correlation to understand correlation btw the independent variables which can be used to decide which 
#features to include in the model.We will first select features that have VIF value of more than 4.


# In[153]:


columns_with_large_vif = vif_factors[vif_factors.VIF>4].column
columns_with_large_vif


# In[154]:


plt.figure(figsize=(14,10))
sns.heatmap(X[columns_with_large_vif].corr(),annot=True)
plt.title('Figure 4.5 Heatmap depicting the correlation between features.')


# In[155]:


# Insights of bapve graph
#1.T-runs and odi runs, t wickets and odi wickets, are highly corrrelated.
#2. batsman features like hs, run-s, sixers, average are highly correlated
#3. bowlind features like ave-bl, econ, sr-bl are highly correlated.


# In[156]:


columns_to_be_removed=['T-RUNS','T-WKTS','HS','AVE', 'SR-B','AVE-BL','ECON', 'SR-BL', 'AGE_2',
                          'ODI-RUNS-S','ODI-SR-B','RUNS-C','RUNS-S']


# In[157]:


X_new_features= list(set(X_features) - set(columns_to_be_removed))


# In[158]:


get_vif_factors(X[X_new_features])


# In[159]:


# THe VIFs on the final set of variable indicate that there is no multicollinearity as VIFs greater than 4 are not present any
#more.We can proceed to build model with these sets of variables. 


# In[160]:


train_X= train_X[X_new_features]
ipl_model_2=sm.OLS(train_Y,train_X).fit()
ipl_model_2.summary()


# In[161]:


# based on p values only variables COUNTRY_ENG,SIXERS,COUNTRY_IND,CAPTAINCY EXP_1 are statistically significant
#as their p value is less than 0.05. So,these feature sdecide the sold price.


# In[162]:


significant_vars=(['COUNTRY_ENG','SIXERS','COUNTRY_IND','CAPTAINCY EXP_1'])
train_X=train_X[significant_vars]
ipl_model_3=sm.OLS(train_Y,train_X).fit()
ipl_model_3.summary2()


# In[163]:


# Insights of above model:
#1.All variables are statistically significant as p value is less than 0.05%
#2.The overall model is also signififcant as F-statistics is also less than 0.05%.
#3. The model can explain 71.5% of the variance in SOLD PRICE as the R-squared value is o.715.
#4. Adjusted R-squared value is 0.704. It is measure that is calculated after normalizing SSE and SST with corresponding 
#degrees of freedom.


# In[164]:


# Using adjusted R-squared over R-squared may be favored because of its ability to make a more accurate view of the
#correlation between one variable and another. Adjusted R-squared does this by taking into account how many independent
#variables are added to a particular model against which the stock index is measured.


# In[165]:


#SSE is the sum of squares due to error and SST is the total sum of squares.


# In[166]:


# RESIDUAL ANALYSIS IN MULTIPLE LINEAR REGRESSION:
# 1.Test for Normalality of Residuals(P-P plot)-
# The most important assumptions of regression is that the residuals shoild be normally distributed.Can be verified by P-P plot.
#We will develop draw_pp_plot() which takes the model output(residuals) and draws the P-P plot.


# In[167]:


probplot = sm.ProbPlot(ipl_model_3.resid)
plt.figure(figsize=(8,8))
probplot.ppplot(line='45')
plt.title("Figure 4.6 - Normal P-P Plot of Regression Standardized Residuals")
plt.show()


# In[168]:


def draw_pp_plot(model,title):
    probplot = sm.ProbPlot(model.resid);
    plt.figure(figsize=(8,8));
    probplot.ppplot(line='45');
    plt.title(title);
    plt.show();


# In[169]:


draw_pp_plot(ipl_model_3,"Figure 4.6 - Normal P-P Plot of Regression Standardized Residuals");


# In[170]:


# The pp plot above shows that the residuals follow an approximate normal distribution.


# In[171]:


# RESIDUAL PLOT FOR HOMOSCEDASTICITY AND MODEL SPECIFICATION:
#1.Residual plot is a plot between standardized fitted values and residuals.The residuals should n0t have any patterns.
#2.Residual plot with shape such as funnel may indicate existence of heteroscedasticicty.
#3. Any pattern in residual plot indicate use of incorrect functional form in regression model development.


# In[172]:


# Data standardization is a preprocessing step that you will be performing before running your model.
#Indeed, in some projects, you may improve the performance of your models by standardizing some of your features.
#Standardization is used on the data values that are normally distributed. Further, by applying standardization,
#we tend to make the mean of the dataset as 0 and the standard deviation equivalent to 1.


# In[173]:


def get_standardized_values(vals):
    return (vals-vals.mean())/vals.std()


# In[174]:


def plot_resid_fitted(fitted,resid,title):
    plt.scatter(get_standardized_values(fitted),
                get_standardized_values(resid))
    plt.title(title)
    plt.xlabel("Standardized predicted values")
    plt.ylabel("Stanadrdized residual values")
    plt.show()


# In[175]:


plot_resid_fitted(ipl_model_3.fittedvalues,ipl_model_3.resid,"Figure 4.7 - Residual Plot")


# In[176]:


# In above plot,the residuals do not show any sign of heteroscedasiticity( no funnellike pattern)


# In[177]:


# DETECTING INFLUENCERS:
#In OLS estimate,we assume that each record in data has equal influence on model parameters(regression coefficient)is not true.
#We use influence_plot() to identify hifhly influential observations.
#Leverage values of more than 3(k+1)/n are treated as highly influencial observations where k is no of variables and
# n is no of observations.


# In[178]:


k=train_X.shape[1]
n=train_X.shape[0]
print('Number of variables:',k,'and number of observations:',n)


# In[179]:


leverage_cutoff=3*((k+1)/n)
print('Cut off leverage value:',round(leverage_cutoff,3))


# In[180]:


#A leverage point is an observation that has an unusual predictor value (very different from the bulk of the observations).
#An influence point is an observation whose removal from the data set would cause a large change in the estimated
#reggression model coefficients.


# In[181]:


from statsmodels.graphics.regressionplots import influence_plot
fig,ax= plt.subplots(figsize=(8,6))
influence_plot(ipl_model_3, ax=ax)
plt.title('Figure 4.8- Leverage Values Vs Residuals')
plt.show()


# In[182]:


# In above plot there are three observations 23,83,58 that have comparitively high leverage with residuals.
#We can filter out these influenctial observations.


# In[183]:


ipl_auction_df[ipl_auction_df.index.isin( [23,83,58] )]


# In[185]:


train_X_new = train_X.drop([23,83,58],axis=0)
train_Y_new = train_Y.drop([23,83,58],axis=0)


# In[187]:


# TRANSFORMATION RESPONSIBLE VARIABLE:
#Transformation is process of deriviving new dependent and /independent variable to identify the correct functional form of the
#regression model.
#For ex- the dependent variable Y may be replaced in model with ln(Y),1/Y and  


# In[188]:


# Transformation in MLR is used to address the following issues:
#1.Poor fit (low R-squared value)
#2.Pattern in residual analysis indicating non linear releationship between the dependent and independent variables.
#3.Residuals do not follow a normal distribution.
#4. Residuals are not homoscedatic. 


# In[189]:


train_Y = np.sqrt(train_Y)


# In[190]:


ipl_model_4=sm.OLS(train_Y,train_X).fit()
ipl_model_4.summary2()


# In[191]:


# In above model , the R-squared value of the model is increased to 0.751.


# In[193]:


draw_pp_plot(ipl_model_4,"Figure 4.9 - Normal P-P Plot of Regression Standardized Residuals ")


# In[195]:


# MAKING PREDICTION ON VALIDATION SET:
#After final model is built as per our requirements and model has passed all diagnostic test, we can apply model on the
#validation test data to predict the SOLD PRICE.As model we have built predicts the sqrt of SOLD PRICE ,we need to square
#the predicted values to get actual SOLD PRICE.


# In[202]:


pred_Y=np.power(ipl_model_4.predict(test_X[train_X.columns]),2)


# In[210]:


from sklearn import metrics
np.sqrt(metrics.mean_squared_error(pred_Y,test_Y))


# In[218]:


np.round(metrics.r2_score(pred_Y,test_Y),2)


# In[ ]:


# The accuracy (R-squared) value on validation set (0.44) is quite low compared reported by the model on training dataset(0.751)
# This could be a sign of model over-fitting .


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




