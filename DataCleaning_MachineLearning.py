
# coding: utf-8

# In[22]:


#load modules and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


# In[8]:


#designate file paths
main_folder = 'C:/Users/128297/Documents/Test_Datasets_ML/bank'
file_1 = main_folder +'/'+ 'bank-full.csv'


# In[10]:


try:
    dataframe1 = pd.read_csv(file_1, sep = ';', na_values = '')
except IOError:
    print('file not found, provide correct filename')

#get info from dataframe
dataframe1.info()
    
categorical_columnnames = dataframe1.select_dtypes(include = 'category').columns
categorical_columnnames
# In[12]:


def Cat_To_Dummy(data, column_name):
    
    dmy_var = pd.get_dummies(data[column_name],drop_first=False)
    data
    data.drop(column_name)
    
    
    data_dmy = pd.concat([data, dmy_var], axis=1)
    return data_dmy

for i in categorical_columnnames:
    print (i)
    print (dataframe1['age'])
    dataframe1_dmy = Cat_To_Dummy(dataframe1, i)
   

dataframe1_dmy

dmy_edu = pd.get_dummies(dataframe1['education'],drop_first=False)
dmy_job = pd.get_dummies(dataframe1['job'],drop_first=False)
dmy_loan = pd.get_dummies(dataframe1['loan'],drop_first=False)
dmy_hous = pd.get_dummies(dataframe1['housing'],drop_first=False)
dmy_con = pd.get_dummies(dataframe1['contact'],drop_first=False)
dmy_pout = pd.get_dummies(dataframe1['poutcome'],drop_first=False)

dataframe1.drop(['Sex', 'Embarked'],axis=1,inplace=True)


# In[13]:


#Example to convert categorical values to numeric ones
#dataframe1['column_name'].replace(['Yes','No'],[1,0],inplace=True)

#remove non explanatory variables like customer_ID
dataframe1.pop('default')
dataframe1.pop('previous')
# In[16]:


#Check correlations to decide which variables to include in model
def corr_visualize(data):
    
    
    corr = data.corr()
    print(corr)
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
    heat_map=plt.gcf()
    heat_map.set_size_inches(20,15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    return corr
try:
    corr = corr_visualize(dataframe1)
except NameError:
    print('define a dataframe first')

corr_visualize(dataframe1)

# In[17]:


#remove strongly correlated explanatory (independent) variables
dataframe1.pop('column_name_toberemoved')


# In[24]:


dependent = ''
test_size = ''

def traintest_split(data, test_size, dependent):
    from sklearn.model_selection import train_test_split
    #split the dataframe in train and test data
    try:
        train, test = train_test_split(dataframe1, test_size = test_size)
        #set model variable names
    
        
        train_y = train[dependent]
        test_y = test[dependent]

        train_x = train
        train_x = train_x.pop(dependent)
        test_x = test
        test_x = test_x.pop(dependent)
    
    except NameError:
        print('define dependent column and test_size first!')
        
    return train_x, train_y, test_x, test_y

try:
    train_x, train_y, test_x, test_y = traintest_split(dataframe1, test_size, dependent_column)
except NameError:
    print('Define function input first....')
    

# In[25]:





# In[27]:


#Initiate model and train
logisticRegr = LogisticRegression()
logisticRegr.fit(X=train_x, y=train_y)

#Model prediction
test_y_pred = logisticRegr.predict(test_x)

#Test performance
confusion_matrix = confusion_matrix(test_y, test_y_pred)
print('Accuracy score for the model on the test set: {:.2f}'.format(logisticRegr.score(test_x, test_y)))
print(classification_report(test_y, test_y_pred))


# In[1]:


#Plot performance 
confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No churn', 'Churn'), ('No churn', 'Churn'))
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)
plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label', fontsize = 14)


# In[2]:


#Test distribution dependent variable

dataframe1[dependent].value_counts()
 
#In case the dependent variable is unequally distributed, upsampling is needed

def upsampling(data):
    ###Before using this function, define dependent variable column###
    from sklearn.utils import resample
    try:
        data_majority = data[data[dependent]==0]
        data_minority = data[data[dependent]==1]
        upsampled_data = resample(data_minority, replace = True, n_samples = len(data_majority), random_state = 1)
    except:
        print("upsampling failed due to undefined dependent column name")
        
    
    
    return data_majority, upsampled_data

data_majority, upsampled_data = upsampling(dataframe1)


#Connect upsampled data with original majority data
data_upsampled = pd.concat([data_majority, upsampled_data])


# In[3]:


def make_decision_tree(train_x, train_y):
    from sklearn import tree
    import graphviz
    decisionTree = tree.DecisionTreeClassifier(max_depth = 4)
    decisionTree = decisionTree.fit(X=train_x, y=train_y)
    DecisionTree_export = tree.export_graphviz(decisionTree, out_file=None, 
                         feature_names = list(train_x.columns.values),  
                         class_names = ['No Churn', 'Churn'],
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = graphviz.Source(DecisionTree_export)
    graph.render('decision_tree.gv', view=True)
    return decisionTree


DecisionTree = make_decision_tree(train_x, train_y)
test_y_pred = DecisionTree.predict(test_x)

print('Accuracy of decision tree classifier on test set: {:.2f}'.format(DecisionTree.score(test_x, test_y)))


# In[ ]:


def make_randomForest(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    RandomForest = RandomForestClassifier()
    RandomForest.fit(train_x, train_y)
    print('Accuracy of random forest classifier on test set: {:.2f}'.format(RandomForest.score(test_x, test_y)))
    return RandomForest.score(test_x, test_y)
