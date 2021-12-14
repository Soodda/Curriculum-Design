import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('D:/lab/机器学习/课程设计/house-prices-advanced-regression-techniques/train.csv')

#输出前十行的数据
print(data.head(10))

#检查数据的维度，行和列的数量
print(data.shape)



# In[5]:


data.info()
#查看数据类型 和 统计非缺失值个数


# In[6]:


(data.isnull().sum()/len(data))
#查看每一列的缺失值比例  删除有大多数缺失值的列


# In[7]:


data.describe()##数量统计，均值，标准差，最小值，四分之一分位数，二分之一分位数，四分之三分位数，最大值
#我们将获得连续变量的基本描述性统计信息
#箱线图


# In[8]:


categorical=[]
continous=[]
for x in data.columns :
    if data[x].dtypes=='object':
        categorical.append(x)#将筛选出来的列增加到序列的后面
    else:
        continous.append(x)
#我们将数据划分为分类变量和连续变量
#categorical变量是固定值，我们需要从列表中选择值 continous变量是数值


# In[9]:


print(categorical)
print('Categorical columns is',len(categorical))
print(continous)
print('Continous columns is',len(continous))
#输出分类完毕的分类值和连续值的列名.


# In[10]:


for x in continous[1:]:
    sns.boxplot(data[x])
    plt.show()
#我们将得到异常值，它显示与其余数据相距甚远的数值数据
#125值线是上胡须
#框中的粗线为中位数


# In[11]:


for x in continous[1:]:
    sns.distplot(data[x])
    plt.show()
#分布图
#我们将知道偏度 - 方差的度量和峰度


# In[12]:


for x in categorical[2:]:
    sns.countplot(data[x])
    plt.show()
#列中每个类相对于所有其他列的计数


# In[13]:


fig, ax = plt.subplots(19,2,figsize=(20,60))
ax = ax.flatten()

for i, col in enumerate(continous[1:]):
    sns.scatterplot(data[col],data['SalePrice'], ax = ax[i])
    
plt.tight_layout()
plt.show()
#二元分析
#Saleprice与其他自变量之间的关系将完成
#我们将在这里使用散点图
#将对连续变量和连续变量进行绘图


# In[14]:


fig, ax = plt.subplots(21,2,figsize=(20,60))
ax = ax.flatten()

for i, col in enumerate(categorical[1:]):
    sns.barplot(data[col],data['SalePrice'], ax = ax[i])
    
plt.tight_layout()
plt.show()
#将在分类变量与连续变量之间进行绘制
#此处将使用条形图来了解行为
#特征选择将在这里完成 - 平均值的显着差异。


# In[15]:


(data.isnull().sum()/len(data))
#统计缺失值在每一列中的占比


# In[16]:


data.drop(['LotFrontage', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'],axis=1,inplace=True)
#删掉缺失值占比超过百分之五十的列


# In[17]:


data['FireplaceQu'].fillna('Not Available',inplace=True)
#用 "Not Available"来填补缺失值


# In[18]:


missing_columns_data=pd.DataFrame(data.isnull().sum()/len(data)).sort_values(by=0,ascending=False)#按照降序排列
missing_columns_data=missing_columns_data.reset_index()
missing_columns_data.columns=['Column Names','Missing']
missing_columns_data['dtypes']=list(data[list(missing_columns_data['Column Names'])].dtypes)

#在变量"missing_columns_data"中创建缺少的变量，并按降序对它们进行排序
#打印缺失值及其数据类型，用于处理缺失值


# In[19]:


missing_columns_data.head(10)
#w输出 missing_columns_data的前十行


# In[20]:


data['GarageType'].mode()#众数
data['GarageYrBlt'].median()#中位数
data['GarageFinish'].mode()
data['GarageQual'].mode()
data['GarageCond'].mode()
data['BsmtExposure'].mode()
data['BsmtFinType2'].mode()
data['BsmtFinType1'].mode()
data['BsmtQual'].mode()
data['BsmtCond'].mode()
data['MasVnrType'].mode()
data['MasVnrArea'].median()
data['Electrical'].mode()
#获取categorical变量的众数和continue变量的中位数
#获得的值将被填充上述值


# In[21]:


data['GarageType'].fillna('Attchd',inplace=True)
data['GarageYrBlt'].fillna('1984.5',inplace=True)
data['GarageFinish'].fillna('Unf',inplace=True)
data['GarageQual'].fillna('TA',inplace=True)
data['GarageCond'].fillna('TA',inplace=True)
data['BsmtExposure'].fillna('0',inplace=True)
data['BsmtFinType2'].fillna('Unf',inplace=True)
data['BsmtFinType1'].fillna('Unf',inplace=True)
data['BsmtQual'].fillna('TA',inplace=True)
data['BsmtCond'].fillna('TA',inplace=True)
data['MasVnrType'].fillna('None',inplace=True)
data['MasVnrArea'].fillna(103.68526170798899,inplace=True)
data['Electrical'].fillna('SBrkr',inplace=True)
#用查找出来的数据填充


# In[22]:


data.isnull().sum()/len(data)
#了解每个列标题的空值百分比，以确保没有隐藏列


# In[23]:


categorical=[]
continous=[]
for x in data.columns :
    if data[x].dtypes=='object':
        categorical.append(x)
    else:
        continous.append(x)
#我们将数据划分为分类变量和连续变量。
#分类变量是固定值，我们需要从列表中选择值
#连续变量是数值


# In[24]:


categorical.remove("GarageYrBlt")#删除


# In[25]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data[categorical]=data[categorical].apply(le.fit_transform)
#对数据进行标准化
#使用标签解码器进行解码，用于对分类特征的级别进行编码
#分类要素将被编码为数值


# In[26]:


data.head()
#获取训练集的前五行数据


# In[27]:


X=data.drop('SalePrice',axis=1)
y=data['SalePrice']
#用于构建基本模型，并指定为"销售价格"


# In[28]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.26,random_state=42)#test_size:样本占比，为测试集样本数目与原始样本数目之比
#将对测试数据和训练数据执行线性回归


# In[29]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()#线性回归
lr.fit(X_train,y_train)#（输入数据，标签）
print('{} train score:{}'.format(lr,lr.score(X_train,y_train)))#主要用于度量给定测试集的预测效果的好坏
print('{} test score:{}'.format(lr,lr.score(X_test,y_test)))

#将导入线性回归技术
#这将执行基于给定自变量预测因变量的任务
#此技术库发现输入和输出之间的线性关系


# In[30]:


test_data=pd.read_csv('D:/lab/机器学习/课程设计/house-prices-advanced-regression-techniques/test.csv')


# In[31]:


test_data.shape


# In[32]:


test_data.drop(['LotFrontage', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'],axis=1,inplace=True)


# In[33]:


test_data['FireplaceQu'].fillna('Not Available',inplace=True)


# In[34]:


test_data['GarageType'].fillna('Attchd',inplace=True)
test_data['GarageYrBlt'].fillna('1984.5',inplace=True)
test_data['GarageFinish'].fillna('Unf',inplace=True)
test_data['GarageQual'].fillna('TA',inplace=True)
test_data['GarageCond'].fillna('TA',inplace=True)
test_data['BsmtExposure'].fillna('0',inplace=True)
test_data['BsmtFinType2'].fillna('Unf',inplace=True)
test_data['BsmtFinType1'].fillna('Unf',inplace=True)
test_data['BsmtQual'].fillna('TA',inplace=True)
test_data['BsmtCond'].fillna('TA',inplace=True)
test_data['MasVnrType'].fillna('None',inplace=True)
test_data['MasVnrArea'].fillna(103.68526170798899,inplace=True)
test_data['Electrical'].fillna('SBrkr',inplace=True)
test_data['MSZoning'].fillna('RL',inplace=True)
test_data['Utilities'].fillna('AllPub',inplace=True)
test_data['Exterior1st'].fillna('VinylSd',inplace=True)
test_data['Exterior2nd'].fillna('VinylSd',inplace=True)
test_data['BsmtFinSF1'].fillna(443,inplace=True)
test_data['BsmtFinSF2'].fillna(46,inplace=True)
test_data['BsmtUnfSF'].fillna(567,inplace=True)
test_data['TotalBsmtSF'].fillna(1057,inplace=True)
test_data['BsmtFullBath'].fillna(0,inplace=True)
test_data['BsmtHalfBath'].fillna(0,inplace=True)
test_data['Functional'].fillna('Typ',inplace=True)
test_data['KitchenQual'].fillna('TA',inplace=True)
test_data['GarageCars'].fillna(2,inplace=True)
test_data['GarageArea'].fillna(0,inplace=True)
test_data['SaleType'].fillna('WA',inplace=True)


# In[35]:


test_data.isnull().sum()/len(test_data)


# In[36]:


categorical=[]
continous=[]
for x in test_data.columns :
    if test_data[x].dtypes=='object':
        categorical.append(x)
    else:
        continous.append(x)


# In[37]:


categorical.remove("GarageYrBlt")


# In[38]:


test_data[categorical]=test_data[categorical].apply(le.fit_transform)


# In[39]:


test_data['SalePrice']=lr.predict(test_data)


# In[40]:


final_submission=test_data[['Id','SalePrice']]


# In[41]:


final_submission.head()


# In[42]:


final_submission.to_csv('D:/lab/机器学习/课程设计/submission2.csv', index=False)

