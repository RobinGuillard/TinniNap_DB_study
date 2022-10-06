import os
import pandas as pd
import config
import numpy as np

def split_database_col(df,col):
    """ Generates n smaller df from an initial df for each n different values of column col. Must be used for
    categorical col

        Parameters
        ----------
        df : pandas dataframe,
            Initial dataframe
        col: string,
            Name of the column of the split
        Returns
        dic_df : python dictionnary
        a dictionnary where the keys are the split col values and the values are the partial databases
        """
    dic_df = {}
    keys = df[col].unique()
    for elm in keys :
        dic_df[elm] = df[df[col] == elm]
    return dic_df

def get_mean_std_num_cols(dic_df, num_cols):
    """ Generates n smaller df from an initial df for each n different values of column col. Must be used for
        categorical col

            Parameters
            ----------
            dic_df : python dictionnary
                a dictionnary where the keys are the split col values and the values are the partial databases
            num_cols: list of string,
                Names of the numerical continuous columns of the dataset imported from config
            Returns

            """
    dic_cols = {}
    for col in num_cols:
        dic_cols[col] = { "order_vals":[],"means":[], "stds":[], "medians":[]}
        for X in list(dic_df.keys()):
            dic_cols[col]["order_vals"].append(X)
            dic_cols[col]["means"].append(np.mean(dic_df[X][col]))
            dic_cols[col]["stds"].append(np.std(dic_df[X][col]))
            dic_cols[col]["medians"].append(np.median(dic_df[X][col]))
    return dic_cols



if __name__ == "__main__":
    DIRECTORY = "D:\Documents\Thèse EDISCE\TinniNap_DB_study\data"
    FILENAME = os.path.join(DIRECTORY, "sarah_michiels_v2.csv")
    # Avoiding path issues related to different OS
    data = pd.read_csv(FILENAME, sep=",", encoding="latin1")
    dic_df = split_database_col(data, "InflNap")
    print(list(data.keys()))
    num_cols = config.SM_num_cols
    print(num_cols)
    dic_cols = get_mean_std_num_cols(dic_df, num_cols)
    for elm in dic_cols:
        print(elm)
        print(dic_cols[elm])