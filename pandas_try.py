import numpy as np
import pandas as pd

labels = ['a','b','c']
my_list = [1,2,3]
arr = np.array(my_list)
d = {'a':10,'b':20,'c':30}

## Series
pd.Series(data=my_list) # creating a Series
pd.Series(data=my_list, index=labels) # creating a series w labels
pd.Series(arr,labels) # data can be python list or nparray
pd.Series(d) # data can be dictionary, labels are auto-added
pd.Series([sum,print,len]) # even functions
ser1 = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan'])
ser2 = pd.Series([1,2,5,4],index = ['USA', 'Germany','Italy', 'Japan'])       
ser1; ser2; 
ser1+ser2 # row match

## datafram
from numpy.random import randn
np.random.seed(101)
df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())
df[['W','Z']] # select columns with names
df.W # SQL syntax
type(df['W']); type(df[['W']])
df['new'] = df['W'] + df['Y']; df  # create new column
df.drop('new',axis=1); df # remove column; not inplace
df.drop('new',axis=1,inplace=True); df # remove column; inplace
df.drop('E',axis=0); df # remove row

df.loc[['A','B']] # select rows
df.iloc[range(2)] # select rows by indexes
df.loc['B','Y'] # select individual
df.loc[['A','B'],['W','Y']] # select block
df[df>0] # conditional selection
df[ (df['W']>0) & (df['X']>0) ] # conditional selection; cannot use and

newind = 'CA NY WY OR CO'.split()
df['States'] = newind; df
df.set_index('States')

df.reset_index(inplace=True); df # Reset to default 0,1...n index
df.set_index('index', inplace=True); df

## Multi-index and index hierarchy
# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index); hier_index

df = pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B']); df
df.loc['G1']
df.loc['G1'].loc[[1,2]]
df.index.names = ['Group','Num']; df
df.xs('G1'); df.loc['G1'] # same


## Missing data
df = pd.DataFrame({'A':[1,2,np.nan],
                   'B':[5,np.nan,np.nan],
                   'C':[1,2,3]}); df
df.dropna() # try inplace=True
df.dropna(axis=1)
df.dropna(axis=0)
df.dropna(thresh=2) # at least 2 non-NA
df.fillna(value=100)
df['A'].fillna(value=df['A'].mean(), inplace=True); df

## group by
# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}
df = pd.DataFrame(data)
by_comp = df.groupby("Company")

# Applying aggregation functions
df.groupby("Company").count()
df.groupby("Company").max()
df.groupby("Company").max().loc['FB']
df.groupby("Company").sum()
df.groupby("Company").mean() # error
df.groupby("Company").std() # error
df.groupby("Company").describe()
df.groupby("Company").describe().transpose()
df.groupby("Company").describe().transpose()['GOOG']

## Merging, Joining, Concatenating
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])
pd.concat([df1,df2,df3]) # rbind with index name
pd.concat([df1,df2,df3],axis=1) # cbind with index name; allows repeated column name


## Merging
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})    
pd.merge(left,right,how='inner',on='key'); pd.concat([left,right],axis=1) 

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})
pd.merge(left,right,how='inner',on=['key1','key2']); pd.concat([left,right],axis=1) 
pd.merge(left,right,how='outer',on=['key1','key2'])
pd.merge(left,right,how='left',on=['key1','key2'])
pd.merge(left,right,how='right',on=['key1','key2'])

## joining (rely on rownames, not rely on 'keys')
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])
pd.concat([left,right])
left.join(right)
left.join(right,how='outer')

## Operations
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df['col2'].unique()
len(df['col2'].unique())
df['col2'].nunique() # number of unique
df['col2'].value_counts() # table

#Select from DataFrame using criteria from multiple columns
newdf = df[(df['col1']>2) & (df['col2']==444)]; newdf

# apply
def times2(x):
    return x*2
  
df['col1'].apply(times2)
df['col3'].apply(len)
df['col1'].sum()
del df['col1']; df # Permanently Removing a Column


## other attributes and functions
df.columns
df.index
df.sort_values(by='col2'); df #inplace=False by default
df.isnull()
df.dropna()
df = pd.DataFrame({'col1':[1,2,3,np.nan],
                   'col2':[np.nan,555,666,444],
                   'col3':['abc','def','ghi','xyz']})
df.fillna('FILL')


data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}
df = pd.DataFrame(data)
df.pivot_table(values='D',index=['A', 'B'],columns=['C'])


## Data input and output
folder = 'Refactored_Py_DS_ML_Bootcamp-master\\03-Python-for-Data-Analysis-Pandas\\'
df = pd.read_csv(folder+'example')
df
df.to_csv('example.csv',index=False)
pd.read_excel(folder+'Excel_Sample.xlsx',sheetname='Sheet1')
df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')

## SQL
from sqlalchemy import create_engine
