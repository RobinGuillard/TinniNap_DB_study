import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, 'D:/Documents/Thèse EDISCE/TinniNap_DB_study/code')

from DB_interpreter import split_database_col, get_mean_std_num_cols

def test_split_database_col():

    dict1 = {"A": 1, "B": 12, "C": 14}
    dict2 = {"A": 1, "B": 17, "C": 12}
    dict3 = {"A": 2, "B": 11, "C": 14}
    dict4 = {"A": 2, "B": 11, "C": 14}
    dictList = [dict1, dict2, dict3, dict4]
    myDf = pd.DataFrame(dictList)

    dic_df = split_database_col(myDf, "A")
    assert len(list(dic_df.keys())) == 2
    assert list(dic_df.keys()) == [1,2]
    assert list(dic_df[1]["B"]) == [12,17]

def test_get_mean_std_num_cols():
    dict1 = {"A": 1, "B": 12, "C": 14}
    dict2 = {"A": 1, "B": 17, "C": 12}
    dict3 = {"A": 2, "B": 11, "C": 14}
    dict4 = {"A": 2, "B": 11, "C": 14}
    dictList = [dict1, dict2, dict3, dict4]
    myDf = pd.DataFrame(dictList)

    dic_df = split_database_col(myDf, "A")
    dic_cols = get_mean_std_num_cols(dic_df, ["B", "C"])

    assert list(dic_cols.keys()) == ["B", "C"]
    assert dic_cols[list(dic_cols.keys())[0]]["means"] == [14.5, 11]
    assert dic_cols[list(dic_cols.keys())[1]]["stds"] == [np.std([14, 12]), 0]

