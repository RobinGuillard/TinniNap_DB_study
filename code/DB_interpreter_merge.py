import os
import pandas as pd
import config
import useful_func
import numpy as np
import scipy.stats as stats
import pingouin
import csv
from statsmodels.sandbox.stats.multicomp import multipletests


if __name__ == "__main__":
    split_variable = "InflNap"
    DIRECTORY = "D:\Documents\Thèse EDISCE\TinniNap_DB_study\data"
    FILENAME = os.path.join(DIRECTORY, "merge_final_v1.csv")
    # Avoiding path issues related to different OS
    data = pd.read_csv(FILENAME, sep=";", encoding="latin1")

    print(len(data))
    #first step : drop all lines that have nans in the split_variable column :
    data = useful_func.drop_na_split(data, split_variable, suppl_to_drop=[])
    print(len(data))


    #Regrouping naps groups in 3 groups : worsens, improves and nothing changes

    num_cols = config.merge_num
    cat_cols = config.merge_categ


    #extracting datas for numerical cols
    dic_df = useful_func.split_database_col(data, split_variable)
    print(dic_df.keys())
    dic_cols = useful_func.get_mean_std_num_cols(dic_df, num_cols)

    split_sizes = [len(dic_df[i-1]) for i in range(len(dic_df))]
    print(split_sizes)
    # extracting datas for cat cols
    dic_crosstab, dic_post_hoc = useful_func.prepare_chi_squared(data, cat_cols, split_variable, split_sizes)
    print(dic_crosstab)

    #Creating and completing table for csv export
    table=[["", "Worsens (N= "+str(len(dic_df[-1]))+ " )",	"No effect (N= "+str(len(dic_df[0]))+" )",
            "Improves (N= "+str(len(dic_df[1]))+" )",	"Statistic", "P-Value",	"Effect size","","Post-hoc test"]]

    li_pvals=[]
    col_p_vals = []
    for col in num_cols:
        next_line, p_val = useful_func.create_feature_set_of_rows_num(col, dic_cols, useful_func.get_anova_and_eta_squared(data, col, split_variable))
        table.extend(next_line)
        li_pvals.append(p_val)
        col_p_vals.append(col)
    for col in cat_cols:
        #print(col)
        next_line, p_val = useful_func.create_feature_set_of_rows_cat(col, dic_df, data, dic_crosstab)
        if next_line!=0:
            table.extend(next_line)
            li_pvals.append(p_val)
            col_p_vals.append(col)

    #Dealing with holm correction
    p_bool, p_adj, p_Sidak, p_alpha_adj = multipletests(li_pvals, alpha=0.05, method='holm')
    count_p=0
    for row in table:
        if row[-5]!="" and row[-5]!="Statistic":
            if p_bool[count_p]:
                row[-5]=row[-5]+"*"
            row[-4] = p_adj[count_p] #replaces the p-vals by the corrected p-values
            row[-2] = col_p_vals[count_p]
            count_p+=1

    table = useful_func.add_post_hoc(table, dic_cols, dic_post_hoc, num_cols, cat_cols)

    # dernière touche : arrondir les p_val des tests Kruskal-Wallis:
    for row in table:
        if row[-4] != "" and row[-4] != "P-Value":
            row[-4] = useful_func.write_p_val(row[-4])

    #saving csv
    os.chdir("D:/Documents/Thèse EDISCE/TinniNap_DB_study/figures")
    with open(split_variable+"_table_merged_DB.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table)
