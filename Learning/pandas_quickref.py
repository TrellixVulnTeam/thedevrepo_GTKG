
import pandas as pd

'''
Reading
'''

# Create a DataFrame
df = pd.DataFrame({'col_1':['item_1', 'item_2'], 'col_2':[1,2]}, index=['id1', 'id2'])

# Read a CSV as DataFrame
df_csv = pd.read_csv('path/to/file')


'''
Indexing and Slicing
'''

# Index based selection
df.iloc[:2, 0] # exclusive slicing | selection using indexes only

# Label based selection
df.loc[:2, 'col_1'] # inclusive slicing | allows label selections

# set index title
df.set_index('some title')

# conditional selection
df.loc[(df.col_1 == 'item_1') | (df.col_2 < 3)]
df.loc[df.col_1.isin(['item_1','item_2'])]
df.loc[df.col_1.notnull()]

'''
Summary/Aggregation functions
'''
df.col_1.unique()
df.col_1.value_counts()

df.col_2.mean()

df.groupby('col_1').col_2.count()
df.groupby('col_1').col_2.min()

df.groupby('col_1').col_2.apply(lambda m: m/2)
df.groupby('col_1').col_2.apply(lambda df: df.loc[df.col_2.idxmax()])

'''
Rename/Replace
'''
df.col_2.dtype
df.col_2.astype('float64')

df.col_1.replace('item_1',"#")
df.col_1.fillna('Unknown')

df.rename(columns={'col_1':'column_1'})
df.rename(index={0:'first'})

df.rename_axis("title_1", axis='rows')
df.rename_axis("title_2", axis='columns')

# df.concat([df1,df2])
# df.join(df3, lsuffix='_left', rsuffix='_right')

## df.merge is almost similar to df.join
