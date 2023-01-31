import os
import pandas as pd
import config
import numpy as np
import scipy.stats as stats
import pingouin
import csv
from statsmodels.sandbox.stats.multicomp import multipletests



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

def split_database_col(df,col, exclude=[]):
    """ Generates n smaller df from an initial df for each n different values of column col. Must be used for
    categorical col

        Parameters
        ----------
        df : pandas dataframe,
            Initial dataframe
        col: string,
            Name of the column of the split
        exclude : optional
            Values to exclude from the split
        Returns
        dic_df : python dictionnary
        a dictionnary where the keys are the split col values and the values are the partial databases
        """
    dic_df = {}
    keys = list(df[col].unique())
    if len(exclude)>0:
        for excl in exclude:
            keys.remove(excl)
    for elm in keys :
        dic_df[elm] = df[df[col] == elm]
    return dic_df

def get_anova_and_eta_squared(df, num_col, ref_col):
    return pingouin.anova(data=df, dv=num_col, between=ref_col, effsize="np2")["np2"][0]

def prepare_chi_squared(df, cat_cols, ref_col, split_sizes):

    dic_output={}
    for col in cat_cols:
        crosstab = pd.crosstab(df[col], df[ref_col], margins=True)
        print(crosstab)
        ind1 = len(df[col].unique())
        uni = list(df[col].unique())
        for elm in uni:
            if str(elm)=="nan":
                ind1 = ind1-1

        ind2 = len(df[ref_col].unique())

        missing = []

        if (col == "Female"):
            ind1 = ind1-1  #car il y a des np.nan dans unique()
            missing = [9, 19, 8]
        value = np.array([crosstab.iloc[i][0:ind2].values for i in range(ind1)]) #attention hardcoded
        print("here comes value")
        print(value)
        head_count = np.array(crosstab.iloc[ind1][0:ind2].values )  # attention hardcoded
        print("here comes head_count")
        print(head_count)
        if len(head_count)!= len(split_sizes):
            print("head_count et split_sizes n'ont pas la même taille, il y a une erreur")
            return  0
        for i in range(len(head_count)):
            missing.append(split_sizes[i]-head_count[i])

        for i in range(len(head_count)): #on rajoute les pourcentages
            missing.append(100*(split_sizes[i]-head_count[i])/split_sizes[i])


        chi2 = stats.chi2_contingency(value)
        #print(chi2)

        cramerV = np.sqrt(chi2[0]/(chi2[2]*len(df)))   # source : https://www.statology.org/effect-size-chi-square/
        dic_output[col] = [value[i][j] for i in range(len(value)) for j in range(len(value[0])) ]
        dic_output[col].extend(missing)
        dic_output[col].append(chi2[0])#Stat
        dic_output[col].append(chi2[1])#p-val
        dic_output[col].append(cramerV)

    return dic_output

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
        keys = list(dic_df.keys())
        for X in list(keys):
            dic_cols[col]["order_vals"].append(X)
            dic_cols[col]["means"].append(np.mean(dic_df[X][col].dropna()))
            dic_cols[col]["stds"].append(np.std(dic_df[X][col].dropna()))
            dic_cols[col]["medians"].append(np.median(dic_df[X][col].dropna()))
            dic_cols[col]["mins"].append(np.min(dic_df[X][col].dropna()))
            dic_cols[col]["maxs"].append(np.max(dic_df[X][col].dropna()))
            missing = dic_df[X][col].isnull().sum()
            dic_cols[col]["missing"].append(missing)
            dic_cols[col]["perc"].append((missing)*100/len(dic_df[X][col]))
            li_df.append(dic_df[X][col].dropna())
        if len(li_df) == 3: #Beware : hardcoded
            dic_cols[col]["F-stat"], dic_cols[col]["pval"] = stats.kruskal(li_df[0], li_df[1], li_df[2])
        if len(li_df) == 2: #Beware : hardcoded
            dic_cols[col]["F-stat"], dic_cols[col]["pval"] = stats.kruskal(li_df[0], li_df[1])

    return dic_cols

def create_feature_set_of_rows_num(col, dic_cols, np2):
    nb_split = len(dic_cols[col]["order_vals"])

    good_order = list(np.argsort(dic_cols[col]["order_vals"])) #should give the good order to parse the datas
    #print(good_order)
    set_of_rows = []
    first_row = [col]
    mean_line = ["Mean (SD)"]
    median_line =["Median [Min Max]"]
    missing_line=["Missing"]
    for i in good_order:
        first_row.append("")
        mean_line.append(str(round(dic_cols[col]["means"][i], 1)) + " (" + str(round(dic_cols[col]["stds"][i], 2)) + ")")
        median_line.append(str(round(dic_cols[col]["medians"][i],1)) + " [" + str(round(dic_cols[col]["mins"][i], 1)) + ", " +
                       str(round(dic_cols[col]["maxs"][i], 1)) + "]" )
        missing_line.append(str(dic_cols[col]["missing"][i]) + " (" + str(round(dic_cols[col]["perc"][i],1))+"%)")
    first_row.append("F(***) = " + str(round(dic_cols[col]["F-stat"],1)))
    if dic_cols[col]["pval"] < 0.05:
        first_row[-1] = first_row[-1] +"*"
    first_row.append(dic_cols[col]["pval"])
    first_row.append(round(np2,3))
    for i in range(3):
        mean_line.append("")
        median_line.append("")
        missing_line.append("")
    set_of_rows.append(first_row)
    set_of_rows.append(mean_line)
    set_of_rows.append(median_line)
    set_of_rows.append(missing_line)
    #print(set_of_rows)
    return set_of_rows, dic_cols[col]["pval"]

def create_feature_set_of_rows_cat(col, dic_df, data, dic_crosstab): #se méfier des missing values !! #supprimer les nan comme catégorie

    nb_split = len(list(dic_df.keys()))
    print(nb_split)
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

        for i in range(3):
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
        choices = list(data[col].unique())
        choices_2 =[]
        for elm in choices:
            if str(elm)!="nan":
                choices_2.append(elm)

        print("here are the choices :")
        choices_2 = sorted(choices_2)
        print(choices_2)
        print()
        print(dic_crosstab[col])
        print(len(dic_crosstab[col]))

        set_of_rows = []
        first_row = [col]
        print(int((len(dic_crosstab[col]) - 9 )/ 3))
        many_line = [[str(choices_2[i])] for i in range(int((len(dic_crosstab[col]) - 9 )/ 3))]
        missing_line = ["Missing"]



        for i in range(nb_split):
            first_row.append("")
            denoms=0
            for var in range(int((len(dic_crosstab[col]) - 9 )/ 3)):
                denoms+=dic_crosstab[col][i+var*3]
            for var in range(int((len(dic_crosstab[col]) - 9 )/ 3)):
                many_line[var].append(str(dic_crosstab[col][i+var*3]) + " (" + str(
                    round(dic_crosstab[col][i+var*3] * 100 / (denoms),
                          1)) + "%)")
            missing_line.append(
                str(dic_crosstab[col][i-9]) + " (" + str(round(dic_crosstab[col][i-6], 1)) + "%)")

        first_row.append("Chi2(***) = " + str(round(dic_crosstab[col][-3], 1)))
        if dic_crosstab[col][-2] < 0.05:
            first_row[-1] = first_row[-1] + "*"
        first_row.append(dic_crosstab[col][-2])
        first_row.append(round(dic_crosstab[col][-1], 3))
        for i in range(3):
            for var in range(int((len(dic_crosstab[col]) - 9) / 3)):
                many_line[var].append("")

            missing_line.append("")
        set_of_rows.append(first_row)
        set_of_rows.extend(many_line)
        set_of_rows.append(missing_line)
        print(set_of_rows)
        return set_of_rows, dic_crosstab[col][-2]

def drop_na_split(data, split_variable, suppl_to_drop=[]):
    data.dropna(subset=[split_variable], inplace=True)
    if len(suppl_to_drop)>0:
        print("here")
        for val in suppl_to_drop:
            print(val)
            data = data[data[split_variable] != val]
    return data


if __name__ == "__main__":
    split_variable = "InflNap"
    DIRECTORY = "D:\Documents\Thèse EDISCE\TinniNap_DB_study\data"
    FILENAME = os.path.join(DIRECTORY, "merge_final_v1.csv")
    # Avoiding path issues related to different OS
    data = pd.read_csv(FILENAME, sep=";", encoding="latin1")

    print(len(data))
    #first step : drop all lines that have nans in the split_variable column :
    data = drop_na_split(data, split_variable, suppl_to_drop=[])
    print(len(data))


    #Regrouping naps groups in 3 groups : worsens, improves and nothing changes

    num_cols = config.merge_num
    cat_cols = config.merge_categ


    #extracting datas for numerical cols
    dic_df = split_database_col(data, split_variable)
    print(dic_df.keys())
    dic_cols = get_mean_std_num_cols(dic_df, num_cols)

    split_sizes = [len(dic_df[i-1]) for i in range(len(dic_df))]
    print(split_sizes)
    # extracting datas for cat cols
    dic_crosstab = prepare_chi_squared(data, cat_cols, split_variable, split_sizes)
    print(dic_crosstab)

    #Creating and completing table for csv export
    table=[["", "Worsens (N= "+str(len(dic_df[-1]))+ " )",	"No effect (N= "+str(len(dic_df[0]))+" )", "Improves (N= "+str(len(dic_df[1]))+" )",	"Statistic", "P-Value",	"Effect size"]]

    li_pvals=[]
    for col in num_cols:
        next_line, p_val = create_feature_set_of_rows_num(col, dic_cols, get_anova_and_eta_squared(data, col, split_variable))
        table.extend(next_line)
        li_pvals.append(p_val)
    for col in cat_cols:
        #print(col)
        next_line, p_val=create_feature_set_of_rows_cat(col, dic_df, data, dic_crosstab)
        if next_line!=0:
            table.extend(next_line)
            li_pvals.append(p_val)


    #Dealing with holm correction
    p_bool, p_adj, p_Sidak, p_alpha_adj = multipletests(li_pvals, alpha=0.05, method='holm')
    count_p=0
    for row in table:
        if row[-3]!="" and row[-3]!="Statistic":
            if p_bool[count_p]:
                row[-3]=row[-3]+"*"
            row[-2] = p_adj[count_p] #replaces the p-vals by the corrected p-values
            count_p+=1

    #saving csv
    os.chdir("D:/Documents/Thèse EDISCE/TinniNap_DB_study/figures")
    with open(split_variable+"_table_merged_DB.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table)
