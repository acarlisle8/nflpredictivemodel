# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
#Initializing dataframe
df=pd.DataFrame()

# %%
df_2019 = pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2019.csv')
df_2021 = pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2021.csv')
df_2022 = pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2022.csv')
df_2018= pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2018.csv')
df_2017= pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2017.csv')
df_2016= pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2016.csv')
df_2015= pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2015.csv')
df_2014= pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2014.csv')
df_2013= pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2013.csv')
df_2012= pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2012.csv')
df_2011= pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2011.csv')
df_2010= pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2010.csv')
df_2020= pd.read_csv('/Users/aidancarlisle/Downloads/play_by_play_2020.csv')

df = pd.concat([df_2019, df_2020, df_2021, df_2022, df_2018, df_2017, df_2016, df_2015, df_2014, 
               df_2013, df_2012, df_2011, df_2010], ignore_index=True, sort=False)



# %%
#reset the index
df=df.reset_index(drop=True)
print(df.shape)

# %%
#QB dataframe build

# %%

df['explosive_plays'] = df['yards_gained'].apply(lambda x: 1 if x > 15 else 0)
qb_stats= ('season', 'passer_id', 'passer', 'pass', 'complete_pass', 'interception', 'sack', 'yards_gained', 'touchdown', 
           'explosive_plays','third_down_converted', 'rushing_yards')
groupby_qb_stats=('season', 'passer_id', 'passer',)
qb_df = df[list(qb_stats)].groupby(list(groupby_qb_stats), as_index=False).sum()




# %%
#correlates with touchdowns
for y in ['yards_gained', 'complete_pass', 'pass', 'interception', 'sack', 
          'explosive_plays', 'third_down_converted', 'rushing_yards']: 
    sns.regplot(data=qb_df, x='touchdown', y=y)
    plt.title(f'touchdowns and {y}')
    plt.show()

# %%
#make a copy
_df= qb_df.copy()

#add 1 to season
_df['season']= _df['season'].add(1)


# %%
# Merge with OG qb_df via left join
new_qb_df= (qb_df.merge(_df, on=['season', 'passer_id', 'passer'], suffixes=('', '_prev'), how='left'))

# %%
#correlation for next year touchdowns
for y in ['touchdown_prev', 'explosive_plays_prev', 'yards_gained_prev', 'complete_pass_prev', 
          'interception_prev', 'sack_prev', 'yards_gained_prev', 'third_down_converted_prev', 'rushing_yards_prev']:
    sns.regplot(data=new_qb_df, x='touchdown', y=y)
    plt.title(f"touchdowns and {y}")
    plt.show()


# %%

from sklearn.linear_model import Ridge

# %%
#features and targets
features= ['pass_prev', 'complete_pass_prev', 'yards_gained_prev', 'touchdown_prev', 'explosive_plays_prev',
          'third_down_converted_prev', 'rushing_yards_prev']
target='touchdown'

# %%
#eliminate nulls
model_data= (new_qb_df.dropna(subset= features+[target]))
print(model_data.shape)

# %%
#training data
train_data = model_data.loc[model_data['season'].between(2010, 2020), :]


# %%
#testing data
test_data = model_data.loc[model_data['season'].isin([2021, 2022])]



# %%
#Ridge regression model
alpha=4.0
model= Ridge(alpha=alpha)


# %%
#Fit
model.fit(train_data.loc[:, features], train_data[target])

# %%
#Predict on test data
preds=model.predict (test_data.loc[:, features])

preds=pd.Series (preds, index=test_data.index)

test_data['preds']= preds

# %%
#stats to check quality of model
rmse = mean_squared_error(test_data['touchdown'], test_data['preds']) ** 0.5
r2= pearsonr(test_data['touchdown'], test_data['preds'])[0]**2
print (f"rmse:{rmse}\nr2: {r2}")

# %%
sns.regplot(data=test_data,x='touchdown', y='preds')
plt.title('touchdowns and predictions')
plt.show()


