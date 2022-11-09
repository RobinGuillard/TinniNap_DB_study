import os
import pandas as pd
import config
import numpy as np
import scipy.stats as stats
import pingouin
import csv



def merging_groups(df, col, values_to_replace):
    """ Overwrites values in a given col, here to fuse groups

                Parameters
                ----------
               df : pandas dataframe,
                    Initial dataframe
                col: string,
                    Name of the column of the split
                values_to_replace : dict
                a dict with keys the values to replace and as value the value of replacement
                Returns
                df : the same dataframe but with updated values in col
                """
    for index, row in df.iterrows():
        if list(values_to_replace.keys()).__contains__(str(df[col][index])):
            df[col][index] = values_to_replace[str(df[col][index])]
    return df

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

def get_anova_and_eta_squared(df, num_col, ref_col):
    return pingouin.anova(data=df, dv=num_col, between=ref_col, effsize="np2")["np2"][0]

def prepare_chi_squared(df, cat_cols, ref_col):
    crosstabs = []
    for col in cat_cols:
        crosstab = pd.crosstab(df[col], df[ref_col], margins=True)
        ind1 = len(df[col].unique())
        ind2 = len(df[ref_col].unique())
        crosstabs.append(crosstab)


        value = np.array([crosstab.iloc[i][0:ind2].values for i in range(ind1)]) #attention hardcoded
        #print(col)
        chi2 = stats.chi2_contingency(value)
        print(crosstab)
        #print("chi-squared test results (stat, pval, df) : " + str(chi2[0:3]))
        #print(len(df))
        cramerV = np.sqrt(chi2[0]/(chi2[2]*len(df)))   # source : https://www.statology.org/effect-size-chi-square/
        #print("cramerV " + str(cramerV))
        print()
    return 0

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
        dic_cols[col] = { "order_vals":[],"means":[], "stds":[], "medians":[], "F-stat" : -1, "pval" : -1, "mins" : [], "maxs" : [], "missing":[], "perc":[]}
        li_df=[]

        for X in list(dic_df.keys()):
            dic_cols[col]["order_vals"].append(X)
            dic_cols[col]["means"].append(np.mean(dic_df[X][col].dropna()))
            dic_cols[col]["stds"].append(np.std(dic_df[X][col].dropna()))
            dic_cols[col]["medians"].append(np.median(dic_df[X][col].dropna()))
            dic_cols[col]["mins"].append(np.min(dic_df[X][col].dropna()))
            dic_cols[col]["maxs"].append(np.max(dic_df[X][col].dropna()))
            missing = dic_df[X][col].isnull().sum()
            print(missing)
            dic_cols[col]["missing"].append(missing)
            dic_cols[col]["perc"].append((len(dic_df[X][col])-missing)*100/len(dic_df[X][col]))
            li_df.append(dic_df[X][col].dropna())

        if len(li_df) == 3: #Beware : hardcoded
            dic_cols[col]["F-stat"], dic_cols[col]["pval"] = stats.f_oneway(li_df[0], li_df[1], li_df[2])
        if len(li_df) == 2: #Beware : hardcoded
            dic_cols[col]["F-stat"], dic_cols[col]["pval"] = stats.f_oneway(li_df[0], li_df[1])

    return dic_cols

def create_feature_set_of_rows_num(col, dic_cols, np2):
    nb_split = len(dic_cols[col]["order_vals"])
    good_order = np.argsort(dic_cols[col]["order_vals"]) #should give the good order to parse the datas
    #print(good_order)
    set_of_rows = []
    first_row = [col]
    mean_line = ["Mean (SD)"]
    median_line =["Median [Min Max]"]
    missing_line=["Missing"]
    for i in good_order:
        first_row.append("")
        mean_line.append(str(round(dic_cols[col]["means"][i], 1)) + " (" + str(round(dic_cols[col]["stds"][i], 2)) + ")")
        median_line.append(str(dic_cols[col]["medians"][i]) + " [" + str(dic_cols[col]["mins"][i]) + ", " +
                       str(dic_cols[col]["maxs"][i]) + "]" )
        missing_line.append(str(dic_cols[col]["missing"][i]) + " (" + str(round(dic_cols[col]["perc"][i],1))+"%)")
    first_row.append("F(***) = " + str(round(dic_cols[col]["F-stat"],1)))
    if dic_cols[col]["pval"] < 0.05:
        first_row[-1] = first_row[-1] +"*"

    first_row.append(round(np2,3))
    for i in range(2):
        mean_line.append("")
        median_line.append("")
        missing_line.append("")
    set_of_rows.append(first_row)
    set_of_rows.append(mean_line)
    set_of_rows.append(median_line)
    set_of_rows.append(missing_line)
    print(set_of_rows)
    return set_of_rows


if __name__ == "__main__":
    DIRECTORY = "D:\Documents\Thèse EDISCE\TinniNap_DB_study\data"
    FILENAME = os.path.join(DIRECTORY, "sarah_michiels_v3_with_missing.csv")
    # Avoiding path issues related to different OS
    data = pd.read_csv(FILENAME, sep=",", encoding="latin1")

    #Regrouping naps groups in 3 groups : worsens, improves and nothing changes
    data = merging_groups(data, "InflNap", {"-2" : -1, "2" : 1})
    data = merging_groups(data, "InflLightEx", {"-2": -1, "2": 1})
    data = merging_groups(data, "InflModerateWorkout", {"-2": -1, "2": 1})
    data = merging_groups(data, "InflIntenseworkout", {"-2" : -1, "2" : 1})
    data = merging_groups(data, "InflBadSleep", {"-2": -1, "2": 1})
    data = merging_groups(data, "InflGoodSleep", {"-2": -1, "2": 1})
    data = merging_groups(data, "InflAnxiety", {"-2": -1, "2": 1})
    data = merging_groups(data, "InflStress", {"-2": -1, "2": 1})
    num_cols = config.SM_num_cols
    cat_cols = config.SM_already_categorical

    #
    #for elm in cat_cols:
    #    print(elm)
    #    for var in data[elm].unique():
    #        print()


    dic_df = split_database_col(data, "InflNap")

    dic_cols = get_mean_std_num_cols(dic_df, num_cols)
    #for i in range(3):
    #     for elm in dic_cols:
    #         print(dic_cols[elm]["order_vals"][i])
    #         print(str(round(dic_cols[elm]["means"][i], 1)) + " (" + str(round(dic_cols[elm]["stds"][i], 2)) + ")")
    #         print(str(dic_cols[elm]["medians"][i]) + " [" + str(dic_cols[elm]["mins"][i]) + ", " +
    #               str(dic_cols[elm]["maxs"][i]) + "]" )
    #


    #     print()
    #     print()
    #
    #
    # for elm in num_cols:
    #
    #     print( str(round(get_anova_and_eta_squared(data, elm, "InflNap")["np2"].values[0], 3)))
    #     print()
    #     print()
    #get_it = prepare_chi_squared(data, cat_cols, "InflNap")

    table=[["", "Worsens (N= 1404 )",	"Improves (N= 507 )",	"No effect (N= 4204 )",	"Statistic",	"Effect size"]]

    for col in num_cols:
        table.extend(create_feature_set_of_rows_num(col, dic_cols, get_anova_and_eta_squared(data, col, "InflNap")))
    print(table)

os.chdir("D:/Documents/Thèse EDISCE/TinniNap_DB_study/figures")
with open("table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(table)
