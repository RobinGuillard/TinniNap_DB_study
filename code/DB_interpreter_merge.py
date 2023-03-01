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
        next_line, p_val = useful_func.create_feature_set_of_rows_num(col, dic_cols, dic_cols[col]["eta-kruskal"])
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
    recap_sign = {}
    for row in table:
        if row[-5]!="" and row[-5]!="Statistic":
            if p_bool[count_p]:
                recap_sign[row[0]] = [row[-4], row[-3]]
                row[-5]=row[-5]+"*"
            row[-4] = p_adj[count_p] #replaces the p-vals by the corrected p-values
            row[-2] = col_p_vals[count_p]
            count_p+=1

    table = useful_func.add_post_hoc(table, dic_cols, dic_post_hoc, num_cols, cat_cols)

    # dernière touche : arrondir les p_val des tests Kruskal-Wallis:
    for i in range(len(table)):
        row = table[i]
        if row[-4] != "" and row[-4] != "P-Value":
            row[-4] = useful_func.write_p_val(row[-4])
            if list(recap_sign.keys()).__contains__(row[0]):

                recap_sign[row[0]].extend([table[i+1][-2], table[i+1][-1]])
                recap_sign[row[0]].extend([table[i+2][-2], table[i+2][-1]])
                recap_sign[row[0]].extend([table[i + 3][-2], table[i + 3][-1]])
                recap_sign[row[0]].append(row[0])


    recap_sign2=[]
    for elm in recap_sign :
        recap_sign2.append([recap_sign[elm], recap_sign[elm][1]])
    recap_sign2.sort(key=lambda x: x[1])
    recap_sign2.reverse()
    print(recap_sign2)
    print("")
    for elm in recap_sign2:
        if num_cols.__contains__(elm[0][-1]):
            if elm[0][1]>0.01:
                print( "Significative column : " +str(elm[0][-1]))
                print("P-value : " +str(elm[0][0]))
                print("Effect size : " +str(elm[0][1]))
                print(elm[0][2] + " " + str(elm[0][3]))
                print(elm[0][4] + " " + str(elm[0][5]))
                print(elm[0][6] + " " + str(elm[0][7]))
                print("")
        if cat_cols.__contains__(elm[0][-1]):
            if elm[0][1]>0.04:
                print("Significative column : " + str(elm[0][-1]))
                print("P-value : " + str(elm[0][0]))
                print("Effect size : " + str(elm[0][1]))
                print(elm[0][2] + " " + str(elm[0][3]))
                print(elm[0][4] + " " + str(elm[0][5]))
                print(elm[0][6] + " " + str(elm[0][7]))
                print("")
    #saving csv
    os.chdir("D:/Documents/Thèse EDISCE/TinniNap_DB_study/figures")
    with open(split_variable+"_table_merged_DB.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table)
