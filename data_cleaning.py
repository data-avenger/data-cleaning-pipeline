#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder



# Define constants
SELECT_COLS = ['Market/Regular ',  'actual_eta', 'Org_lat_lon', 'Des_lat_lon', 'ontime',
'delay','trip_start_date', 'trip_end_date', 'TRANSPORTATION_DISTANCE_IN_KM', 
'supplierNameCode', 'Material Shipped']
DATE_COLS = ['trip_start_date', 'trip_end_date']
FLOAT_COLS = ['origin_lat','origin_lon','des_lat','des_lon','days_taken']
CAT_COLS = ['market_regular','supplier_name_code','material_shipped']
REPLACE_REGEX_DICT = {r'(?!^)([A-Z]+)':r'_\1','/| ': '','__':'_','T_R':'TR','/':''}
DROP_COLS = ['ontime', 'delay','trip_start_date','trip_end_date','actual_eta','org_lat_lon','des_lat_lon']


# <a id = 'one'></a>
# ## Understand/View 
# 
# For this example we'll use a simple dataset: [Delivery truck trips data](https://www.kaggle.com/ramakrishnanthiyagu/delivery-truck-trips-data).
# 
# Some features we'll look at (of course we could examine more):
# - Market/Regular - Type of trip. Regular - Vendors with whom we will have contract. Market - Vendor with whom we will not have contract
# - Orglatlon - Latitude/Longitude of start place
# - Des lat lon - Latitude/Longitude of end place
# - actual eta - Time when the truck arrived Currlat - current latitude - changes each time when we receive GPS ping
# - TRANSPORTATIONDISTANCEINKM - Total KM of travel
# - supplierNameCode - Supplier who provides the vehicle
# - Material Shipped - Type of materials in delivery
# - ontime/delay - indicates if package was delivered as expected or later.
# 
# ![](https://media.giphy.com/media/xT9C25UNTwfZuk85WP/giphy.gif)
# 


df = pd.read_csv('logistics-data.csv')

# <a id = 'two'></a>
# ## Rename/Select
# 
# The column names are a bit of a mess. Better make them consistent with `snake_case` style (it's the Python standard, get it)!


df = df[SELECT_COLS]
for col_name, rename_col in REPLACE_REGEX_DICT.items():
    df.columns = df.columns.str.replace(col_name, rename_col)
df.columns = df.columns.str.lower()
df.head()


# <a id ='three'> </a>
# ## Create/Drop Columns
# Data cleaning should be for a purpose; usually data visualization, analysis or modelling.
# 
# You create or drop data columns depending on what you've interested in looking at and what you're trying to achieve. 
# ### Create new columns: 
# - `'ontime_delay'`: Convert to integer values with 0 ('ontime') and 1 ('delay').
# - `'origin_lat', 'origin_lon'`: Split latitude and longitude for place of origin into separate columns.
# - `'des_lat', 'des_lon'`: Split latitude and longitude for destination into separate columns
# - `'days_taken'`: Calculate `actual eta` - `trip starting date`.  
# ### Drow rows/columns:
# - Drop rows where `'days_taken'` is negative value (yeah, I'm not sure why?!)
# - Drop columns that are unnecessary, given by `DROP_COLS`.
# 
# Drop it like a boss!
# 
# ![](https://media.giphy.com/media/DfbpTbQ9TvSX6/giphy.gif)



df[['origin_lat', 'origin_lon']] = df['org_lat_lon'].str.split(',',expand=True) 
df[['des_lat', 'des_lon']] = df['des_lat_lon'].str.split(',',expand=True) 
df[DATE_COLS] = df[DATE_COLS].astype('datetime64')
days_taken = (df['trip_end_date']-df['trip_start_date'])/ np.timedelta64(1, 'D')
df.insert(0,'days_taken',days_taken)
df = df.drop(df[df['days_taken']<0].index, axis=0)
df = df.drop(DROP_COLS, axis=1)
df.head()


# <a id ='four'> </a>
# ## Convert Data Types
# - `df.info()`: Easy to check all data types.
# - `.astype()`: Convert data types easily; usually `float, int, category`.
#  
# ![](https://media.giphy.com/media/kDmiZp6eXOgGunaAEe/giphy.gif)


df[FLOAT_COLS] = df[FLOAT_COLS].astype("float")
df[CAT_COLS] = df[CAT_COLS].astype("category")
df2 = df.copy()


# <a id ='five'> </a>
# ## Handle Missing Values
# - `df.isnull().sum()`: Easy to check the number of missing values in each column.
# - We impute missing numeric values with the column mean. 
# 
# Since there are no missing values for categorical variables for this dataset, we commented out code for handling missing categorical values.



# Check for missing data
print(df.isnull().sum()) 
missing_values_df = df.loc[:, df.isnull().sum()>0]

# Handle missing numeric values only
numeric_features = missing_values_df.select_dtypes(include='float').columns
num_imputer = SimpleImputer(missing_values= np.nan, strategy='median')
df[numeric_features] = num_imputer.fit_transform(missing_values_df.select_dtypes(include='float'))


# <a id ='six'> </a>
# ## Scaling/Transforms
# Scaling/transforms is useful for:
# - Visualizing data on the same scale - less distortion or skewness.
# - Data analysis techniques such as PCA, clustering or outlier detection. 
# - Data modelling to prevent variables on a larger scale, dominating the model weights.
# 
# Basically, we want data columns on **similar scales**.
# 
# ![](https://media.giphy.com/media/IcStLavfAdhoQ/giphy.gif)
# 
# 


# Fill missing values for y data (do not scale as we want do not want to unscale for predictions)
y = df2["days_taken"]
num_imputer = SimpleImputer(missing_values= np.nan, strategy='median')
y_pre = num_imputer.fit_transform(y.to_numpy().reshape(len(y), 1)) 


# Preprocess features X
df.drop("days_taken",axis=1, inplace=True)
numeric_features =  df.select_dtypes('number').columns
numeric_transformer = Pipeline(
steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
categorical_features = df.select_dtypes(exclude='number').columns
categorical_transformer = Pipeline(
steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
]
)
preprocess = ColumnTransformer(
transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
]
)
X_pre = preprocess.fit_transform(df)
dataset = np.concatenate((y_pre, X_pre.toarray()), axis=1)
print(dataset)


# Any questions? Let me know!
# 
# Otherwise, you're good to go 🙂
