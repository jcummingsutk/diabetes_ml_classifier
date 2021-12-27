import pandas as pd
#import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

def clean_data(df) -> pd.DataFrame:
    """
    Lowercases the column headers, drops the skin thickness column and
    imputes missing data in the glucose, bloodpressure, insulin, and bmi column
    """
    df.columns = df.columns.str.lower()
    df['glucose'].replace(0, df[df['glucose']>0].loc[:,'glucose'].median(), inplace=True)
    df['bloodpressure'].replace(0, df[df['bloodpressure']>0].loc[:,'bloodpressure'].median(), inplace=True)
    df['insulin'].replace(0, df[df['insulin']>0].loc[:,'insulin'].median(), inplace=True)
    df['bmi'].replace(0, df[df['bmi']>0].loc[:,'bmi'].median(), inplace=True)
    df['skinthickness'].replace(0, df[df['skinthickness']>0].loc[:,'skinthickness'].median(), inplace=True)
    return df

def scale_data(df) -> pd.DataFrame:
    """
    Scales the data using a standard scaler
    """
    ss = StandardScaler()
    cols_to_scale = ['pregnancies', 'age', 'bmi', 'diabetespedigreefunction', 'bloodpressure', 'insulin', 'glucose']
    cols_not_scaled = ['outcome']
    df_scaled = pd.DataFrame(ss.fit_transform(df[cols_to_scale]), index=df.index, columns=cols_to_scale)
    df_scaled[cols_not_scaled] = df[cols_not_scaled]
    return df_scaled

def load_data() -> pd.DataFrame:
    """ Loads, cleans, and scales the diabetes data """
    logging.info("Loading data")
    file = "data/diabetes.csv"
    df = pd.read_csv(file)
    df = clean_data(df)
    df = scale_data(df)
    return df
