import os
import pandas as pd
import config
import useful_func
import numpy as np
import scipy.stats as stats
import pingouin
import csv
from statsmodels.sandbox.stats.multicomp import multipletests
import scikit_posthocs as sp
import copy



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
    dic_post_hoc={}
    dic_output={}
    for col in cat_cols:
        crosstab = pd.crosstab(df[col], df[ref_col], margins=True)
        ind1 = len(df[col].unique())
        ind2 = len(df[ref_col].unique())

        missing = [0,0,0]

        if (col == "Female"):
            ind1 = ind1-1  #car il y a des np.nan dans unique()
            missing = [9, 19, 8]
        value = np.array([crosstab.iloc[i][0:ind2].values for i in range(ind1)]) #attention hardcoded



        chi2 = stats.chi2_contingency(value)
        if chi2[1] < 0.05 : #on conduit les post_hoc tests ATTENTION HARDCODED
            copy_cross = copy.copy(crosstab)
            copy_cross.drop(copy_cross.columns[1], axis=1, inplace=True)
            #print(copy_cross)
            #print(crosstab)

            value_12 = np.array([crosstab.iloc[i][0:2].values for i in range(ind1)])
            value_23 = np.array([crosstab.iloc[i][1:3].values for i in range(ind1)])
            value_13 = np.array([copy_cross.iloc[i][0:2].values for i in range(ind1) ])

            dic_post_hoc[col]= {}
            dic_post_hoc[col]["post-hoc"]=[[[-1,0],stats.chi2_contingency(value_12)[1]],
                                [[0, 1], stats.chi2_contingency(value_23)[1]],
                                [[-1, 1], stats.chi2_contingency(value_13)[1]]
                                ]
            print(dic_post_hoc)

        cramerV = np.sqrt(chi2[0]/(chi2[2]*len(df)))   # source : https://www.statology.org/effect-size-chi-square/
        dic_output[col] = [value[i][j] for i in range(len(value)) for j in range(len(value[0])) ]
        dic_output[col].extend(missing)
        dic_output[col].append(chi2[0])#Stat
        dic_output[col].append(chi2[1])#p-val
        dic_output[col].append(cramerV)

    return dic_output, dic_post_hoc

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
        dic_cols[col] = { "order_vals":[],"means":[], "stds":[], "medians":[], "F-stat" : -1, "pval" : -1, "mins" : [],
                          "maxs" : [], "missing":[], "perc":[], "post-hoc":-1}
        li_df=[]

        for X in list(dic_df.keys()):
            dic_cols[col]["order_vals"].append(X)
            dic_cols[col]["means"].append(np.mean(dic_df[X][col].dropna()))
            dic_cols[col]["stds"].append(np.std(dic_df[X][col].dropna()))
            dic_cols[col]["medians"].append(np.median(dic_df[X][col].dropna()))
            dic_cols[col]["mins"].append(np.min(dic_df[X][col].dropna()))
            dic_cols[col]["maxs"].append(np.max(dic_df[X][col].dropna()))
            missing = dic_df[X][col].isnull().sum()
            dic_cols[col]["missing"].append(missing)
            dic_cols[col]["perc"].append((len(dic_df[X][col])-missing)*100/len(dic_df[X][col]))
            li_df.append(dic_df[X][col].dropna())

        if len(li_df) == 3: #Beware : hardcoded
            dic_cols[col]["F-stat"], dic_cols[col]["pval"] = stats.f_oneway(li_df[0], li_df[1], li_df[2])
            if dic_cols[col]["pval"] < 0.05 :
                #print(col)
                dunn_test = sp.posthoc_dunn([list(li_df[0]), list(li_df[1]), list(li_df[2])])
                dic_cols[col]["post-hoc"] = [[[dic_cols[col]["order_vals"][0], dic_cols[col]["order_vals"][1]], dunn_test[1][2]],
                                             [[dic_cols[col]["order_vals"][1], dic_cols[col]["order_vals"][2]],
                                              dunn_test[2][3]],
                                             [[dic_cols[col]["order_vals"][0], dic_cols[col]["order_vals"][2]],
                                              dunn_test[1][3]]]

                #print(dunn_test)
                #print(dic_cols[col]["order_vals"])
                #print(dic_cols[col]["post-hoc"])
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
    first_row.append(dic_cols[col]["pval"])
    first_row.append(round(np2,3))
    first_row.append("")
    first_row.append("")
    for i in range(5):
        mean_line.append("")
        median_line.append("")
        missing_line.append("")
    set_of_rows.append(first_row)
    set_of_rows.append(mean_line)
    set_of_rows.append(median_line)
    set_of_rows.append(missing_line)
    #print(set_of_rows)
    return set_of_rows, dic_cols[col]["pval"]

def create_feature_set_of_rows_cat(col, dic_df, data, dic_crosstab):

    nb_split = len(list(dic_df.keys()))
    if len(data[col].unique()) == 2 or col =="Female": #attention hardcoded mauvaise gestion des nan
        set_of_rows = []
        first_row = [col]
        Yes_line = ["Yes"]
        No_line =["No"]
        missing_line=["Missing"]
        for i in range(nb_split):
            first_row.append("")
            Yes_line.append(str(dic_crosstab[col][i+nb_split]) + " (" + str(round(dic_crosstab[col][i+nb_split]*100/(dic_crosstab[col][i+nb_split]+dic_crosstab[col][i]), 1)) + "%)")
            No_line.append(str(dic_crosstab[col][i]) + " (" + str(round(dic_crosstab[col][i]*100/(dic_crosstab[col][i+nb_split]+dic_crosstab[col][i]), 1)) + "%)")
            #print(dic_crosstab[col])
            perc = dic_crosstab[col][i+2*nb_split]*100/(dic_crosstab[col][i+2*nb_split]+ dic_crosstab[col][i] + dic_crosstab[col][i+nb_split])
            #print(perc)
            missing_line.append(str(dic_crosstab[col][i+2*nb_split]) + " (" + str(round(perc,1))+"%)")
        first_row.append("Chi2(***) = " + str(round(dic_crosstab[col][-3],1)))

        if dic_crosstab[col][-2] < 0.05:
            first_row[-1] = first_row[-1] +"*"
        first_row.append(dic_crosstab[col][-2])
        first_row.append(round(dic_crosstab[col][-1],3))
        first_row.append("")
        first_row.append("")

        for i in range(5):
            Yes_line.append("")
            No_line.append("")
            missing_line.append("")
        set_of_rows.append(first_row)
        set_of_rows.append(Yes_line)
        set_of_rows.append(No_line)
        set_of_rows.append(missing_line)
        #print(set_of_rows)
        return set_of_rows, dic_crosstab[col][-2]

    else: #case for the non-binary columns
        print(col)
        choices = sorted(list(data[col].unique()))
        print(choices)
        print()


        set_of_rows = []
        first_row = [col]
        print(int((len(dic_crosstab[col]) - 9 )/ 3))
        many_line = [[str(choices[i])] for i in range(int((len(dic_crosstab[col]) - 9 )/ 3))]
        missing_line = ["Missing", "0 (100%)","0 (100%)","0 (100%)"]



        for i in range(nb_split):
            first_row.append("")
            denoms=0
            for var in range(int((len(dic_crosstab[col]) - 9 )/ 3)):
                denoms+=dic_crosstab[col][i+var*3]
            for var in range(int((len(dic_crosstab[col]) - 9 )/ 3)):
                many_line[var].append(str(dic_crosstab[col][i+var*3]) + " (" + str(
                    round(dic_crosstab[col][i+var*3] * 100 / (denoms),
                          1)) + "%)")


        first_row.append("Chi2(***) = " + str(round(dic_crosstab[col][-3], 1)))
        if dic_crosstab[col][-2] < 0.05:
            first_row[-1] = first_row[-1] + "*"
        first_row.append(dic_crosstab[col][-2])
        first_row.append(round(dic_crosstab[col][-1], 3))
        first_row.append("")
        first_row.append("")
        for i in range(5):
            for var in range(int((len(dic_crosstab[col]) - 9) / 3)):
                many_line[var].append("")

            missing_line.append("")
            missing_line.append("")
            missing_line.append("")
        set_of_rows.append(first_row)
        set_of_rows.extend(many_line)
        set_of_rows.append(missing_line)
        print(set_of_rows)
        return set_of_rows, dic_crosstab[col][-2]

def add_post_hoc(table, dic_cols, dic_cols2, num_cols, cat_cols):
    correc_pvals=[]
    compt_ps=0
    #first step : aggregating p_vals for correction
    for i in range(len(table)):
        row = table[i]
        if row[-5]!="" and row[-5]!="Statistic":
            col = row[-2]
            if row[-4] < 0.05 :
                if num_cols.__contains__(col):
                    correc_pvals.append(dic_cols[col]["post-hoc"][0][1])
                    dic_cols[col]["post-hoc"][0][1] = len(correc_pvals)-1 # on écrit dans le dic l'indice de la p-val
                    correc_pvals.append(dic_cols[col]["post-hoc"][2][1])
                    dic_cols[col]["post-hoc"][2][1] = len(correc_pvals) - 1  # on écrit dans le dic l'indice de la p-val
                    correc_pvals.append(dic_cols[col]["post-hoc"][1][1])
                    dic_cols[col]["post-hoc"][1][1] = len(correc_pvals) - 1  # on écrit dans le dic l'indice de la p-val
                    compt_ps+=3

                if cat_cols.__contains__(col):
                    correc_pvals.append(dic_cols2[col]["post-hoc"][0][1])
                    dic_cols2[col]["post-hoc"][0][1] = len(correc_pvals) - 1  # on écrit dans le dic l'indice de la p-val
                    correc_pvals.append(dic_cols2[col]["post-hoc"][1][1])
                    dic_cols2[col]["post-hoc"][1][1] = len(correc_pvals) - 1  # on écrit dans le dic l'indice de la p-val
                    correc_pvals.append(dic_cols2[col]["post-hoc"][2][1])
                    dic_cols2[col]["post-hoc"][2][1] = len(correc_pvals) - 1  # on écrit dans le dic l'indice de la p-val
                    compt_ps += 3

    p_bool, p_adj, p_Sidak, p_alpha_adj = multipletests(correc_pvals, alpha=0.05, method='holm')

    #p_adj contient les p-valeurs corrigées

    #second step : writing p_vals in table
    for i in range(len(table)):
        row = table[i]
        if row[-5]!="" and row[-5]!="Statistic":
            col = row[-2]
            row[-2] = ""
            if row[-4] < 0.05 :
                if num_cols.__contains__(col):

                    table[i+1][-2]="Worsens VS No effect"
                    table[i + 1][-1] = write_p_val(p_adj[dic_cols[col]["post-hoc"][0][1]])
                    table[i + 2][-2] = "Improves VS No effect"
                    table[i + 2][-1] = write_p_val(p_adj[dic_cols[col]["post-hoc"][2][1]]) #attention: indices inversés, surveiller dic_col
                    table[i + 3][-2] = "Worsens VS Improves"
                    table[i + 3][-1] = write_p_val(p_adj[dic_cols[col]["post-hoc"][1][1]]) #attention: indices inversés, surveiller dic_col

                if cat_cols.__contains__(col):

                    table[i+1][-2]="Worsens VS No effect"
                    table[i + 1][-1] = write_p_val(p_adj[dic_cols2[col]["post-hoc"][0][1]])
                    table[i + 2][-2] = "Improves VS No effect"
                    table[i + 2][-1] = write_p_val(p_adj[dic_cols2[col]["post-hoc"][1][1]])
                    table[i + 3][-2] = "Worsens VS Improves"
                    table[i + 3][-1] = write_p_val(p_adj[dic_cols2[col]["post-hoc"][2][1]])
    return table

def write_p_val(pval):
    output = 0
    if pval<0.001:
        output=" p < 0.001"
    else :
        output=round(pval, 3)
    return(output)


if __name__ == "__main__":
    split_variable = "InflNap"
    DIRECTORY = "D:\Documents\Thèse EDISCE\TinniNap_DB_study\data"
    FILENAME = os.path.join(DIRECTORY, "sarah_michiels_v3_with_missing_with_true_on-off.csv")
    # Avoiding path issues related to different OS
    data = pd.read_csv(FILENAME, sep=",", encoding="latin1")

    #Regrouping naps groups in 3 groups : worsens, improves and nothing changes
    data = useful_func.merging_groups(data, "InflNap", {"-2" : -1, "2" : 1})
    data = useful_func.merging_groups(data, "InflLightEx", {"-2": -1, "2": 1})
    data = useful_func.merging_groups(data, "InflModerateWorkout", {"-2": -1, "2": 1})
    data = useful_func.merging_groups(data, "InflIntenseworkout", {"-2" : -1, "2" : 1})
    data = useful_func.merging_groups(data, "InflBadSleep", {"-2": -1, "2": 1})
    data = useful_func.merging_groups(data, "InflGoodSleep", {"-2": -1, "2": 1})
    data = useful_func.merging_groups(data, "InflAnxiety", {"-2": -1, "2": 1})
    data = useful_func.merging_groups(data, "InflStress", {"-2": -1, "2": 1})
    data = useful_func.merging_groups(data, "Female", {"2.0": np.nan})
    num_cols = config.SM_num_cols
    cat_cols = config.SM_already_categorical


    #extracting datas for numerical cols
    dic_df = useful_func.split_database_col(data, split_variable)
    dic_cols = useful_func.get_mean_std_num_cols(dic_df, num_cols)

    split_sizes = [len(dic_df[i-1]) for i in range(len(dic_df))]
    print(split_sizes)
    # extracting datas for cat cols
    dic_crosstab, dic_post_hoc = useful_func.prepare_chi_squared(data, cat_cols, split_variable, split_sizes)

    #Creating and completing table for csv export
    table=[["", "Worsens (N= "+str(len(dic_df[-1]))+ " )",	"No effect (N= "+str(len(dic_df[0]))+" )",
            "Improves (N= "+str(len(dic_df[1]))+" )",	"Statistic", "P-Value",	"Effect size", "","Post-hoc test",
            "Post-hoc effect size"]]
    li_pvals=[]
    col_p_vals=[]
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
        if row[-6]!="" and row[-6]!="Statistic":
            if p_bool[count_p]:
                row[-6]=row[-6]+"*"
                recap_sign[row[0]] = [row[-5], row[-4]]
            row[-5] = p_adj[count_p] #replaces the p-vals by the corrected p-values
            row[-3] = col_p_vals[count_p]
            count_p+=1

    table = useful_func.add_post_hoc(table, dic_cols, dic_post_hoc, num_cols, cat_cols)

    # dernière touche : arrondir les p_val des tests Kruskal-Wallis:
    for i in range(len(table)):
        row = table[i]
        if row[-5] != "" and row[-5] != "P-Value":
            row[-5] = useful_func.write_p_val(row[-5])
            if list(recap_sign.keys()).__contains__(row[0]):
                recap_sign[row[0]].extend([table[i + 1][-3], table[i + 1][-2]])
                recap_sign[row[0]].extend([table[i + 2][-3], table[i + 2][-2]])
                recap_sign[row[0]].extend([table[i + 3][-3], table[i + 3][-2]])
                recap_sign[row[0]].append(row[0])

    recap_sign2 = []
    for elm in recap_sign:
        recap_sign2.append([recap_sign[elm], recap_sign[elm][1]])
    recap_sign2.sort(key=lambda x: x[1])
    recap_sign2.reverse()
    print(recap_sign2)
    print("")
    for elm in recap_sign2:
        if num_cols.__contains__(elm[0][-1]):
            if elm[0][1] > 0.01:
                print("Significative column : " + str(elm[0][-1]))
                print("P-value : " + str(elm[0][0]))
                print("Effect size : " + str(elm[0][1]))
                print(elm[0][2] + " " + str(elm[0][3]))
                print(elm[0][4] + " " + str(elm[0][5]))
                print(elm[0][6] + " " + str(elm[0][7]))
                print("")
        if cat_cols.__contains__(elm[0][-1]):
            if elm[0][1] > 0.04:
                print("Significative column : " + str(elm[0][-1]))
                print("P-value : " + str(elm[0][0]))
                print("Effect size : " + str(elm[0][1]))
                print(elm[0][2] + " " + str(elm[0][3]))
                print(elm[0][4] + " " + str(elm[0][5]))
                print(elm[0][6] + " " + str(elm[0][7]))
                print("")

    table_short = [table[0]]

    flag = 0
    for i in range(len(table)):
        if i > 0:
            if not table[i][-5] == "":
                if table[i][-5] == " p < 0.001":
                    table_short.append(table[i])
                    for j in range(len(table)):  #:parcourir le table à partir de i jusqu'au prochain indice -5 rempli
                        if i + j + 1 < len(table):
                            if table[i + j + 1][-5] == "":
                                table_short.append(table[i + j + 1])
                            else:
                                break

                else:
                    if float(table[i][-5]) < 0.05:
                        table_short.append(table[i])
                        for j in range(
                                len(table)):  #:parcourir le table à partir de i jusqu'au prochain indice -5 rempli
                            if i + j + 1 < len(table):
                                if table[i + j + 1][-5] == "":
                                    table_short.append(table[i + j + 1])
                                else:
                                    break

    print(table_short)

    # saving csv
    os.chdir("D:/Documents/Thèse EDISCE/TinniNap_DB_study/figures")

    with open(split_variable + "_table_short.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table_short)

    with open(split_variable + "_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table)


