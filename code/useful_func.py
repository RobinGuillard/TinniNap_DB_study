
import pandas as pd
import numpy as np
import scipy.stats as stats
import pingouin
import copy
from statsmodels.sandbox.stats.multicomp import multipletests
import scikit_posthocs as sp



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
    dic_post_hoc = {}
    dic_output={}
    for col in cat_cols:
        print(col)
        crosstab = pd.crosstab(df[col], df[ref_col], margins=True)
        ind1 = len(df[col].unique())
        uni = list(df[col].unique())
        for elm in uni:
            if str(elm)=="nan":
                ind1 = ind1-1

        ind2 = len(df[ref_col].unique())

        missing = []

        if (col == "Female"):
            missing = [9, 19, 8]
        value = np.array([crosstab.iloc[i][0:ind2].values for i in range(ind1)]) #attention hardcoded

        head_count = np.array(crosstab.iloc[ind1][0:ind2].values )  # attention hardcoded

        if len(head_count)!= len(split_sizes):
            print("head_count et split_sizes n'ont pas la même taille, il y a une erreur")
            return  0
        for i in range(len(head_count)):
            missing.append(split_sizes[i]-head_count[i])

        for i in range(len(head_count)): #on rajoute les pourcentages
            missing.append(100*(split_sizes[i]-head_count[i])/split_sizes[i])


        chi2 = stats.chi2_contingency(value)
        #print(chi2)

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
            if dic_cols[col]["pval"] < 0.05:
                # print(col)
                dunn_test = sp.posthoc_dunn([list(li_df[0]), list(li_df[1]), list(li_df[2])])
                dic_cols[col]["post-hoc"] = [
                    [[dic_cols[col]["order_vals"][0], dic_cols[col]["order_vals"][1]], dunn_test[1][2]],
                    [[dic_cols[col]["order_vals"][1], dic_cols[col]["order_vals"][2]],
                     dunn_test[2][3]],
                    [[dic_cols[col]["order_vals"][0], dic_cols[col]["order_vals"][2]],
                     dunn_test[1][3]]]
        if len(li_df) == 2: #Beware : hardcoded, would only be useful if loche database
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

def create_feature_set_of_rows_cat(col, dic_df, data, dic_crosstab): #se méfier des missing values !! #supprimer les nan comme catégorie

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
        choices = list(data[col].unique())
        choices_2 =[]
        for elm in choices:
            if str(elm)!="nan":
                choices_2.append(elm)


        choices_2 = sorted(choices_2)


        set_of_rows = []
        first_row = [col]
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
            print("relecture crosstab")
            print(dic_crosstab[col])
            missing_line.append(
                str(dic_crosstab[col][i-9]) + " (" + str(round(dic_crosstab[col][i-6], 1)) + "%)")

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
        set_of_rows.append(first_row)
        set_of_rows.extend(many_line)
        set_of_rows.append(missing_line)
        return set_of_rows, dic_crosstab[col][-2]

def drop_na_split(data, split_variable, suppl_to_drop=[]):
    data.dropna(subset=[split_variable], inplace=True)
    if len(suppl_to_drop)>0:
        for val in suppl_to_drop:

            data = data[data[split_variable] != val]
    return data

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

