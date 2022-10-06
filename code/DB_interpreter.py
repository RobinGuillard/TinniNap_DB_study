import os
import pandas as pd

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




if __name__ == "__main__":
    DIRECTORY = "D:\Documents\Thèse EDISCE\TinniNap_DB_study\data"
    FILENAME = os.path.join(DIRECTORY, "sarah_michiels_v2.csv")
    # Avoiding path issues related to different OS
    data = pd.read_csv(FILENAME, sep=",", encoding="latin1")
    dic_df = split_database_col(data, "Female")
    print(list(data.keys()))
