
import pandas as pd
import numpy as np
import scipy.stats as stats
import pingouin
import copy
from statsmodels.sandbox.stats.multicomp import multipletests
import scikit_posthocs as sp
import config

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return abs((np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof))



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
        print(crosstab)
        ind1 = len(df[col].unique())
        uni = list(df[col].unique())
        for elm in uni:
            if str(elm)=="nan":
                ind1 = ind1-1

        ind2 = len(df[ref_col].unique())

        missing = []


        value = np.array([crosstab.iloc[i][0:ind2].values for i in range(ind1)]) #attention hardcoded
        print(value)

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
            dic_post_hoc[col]["post-hoc_eff"] = [[[-1, 0],
                                                  write_effect_size(round(np.sqrt(stats.chi2_contingency(value_12)[0]/
                                                                                  (stats.chi2_contingency(value_12)[2]*len(df))),3),
                                                                    "CramerV",df=stats.chi2_contingency(value_12)[2])
                                                  ],
                                             [[0, 1], write_effect_size(round(np.sqrt(stats.chi2_contingency(value_23)[0]/(stats.chi2_contingency(value_23)[2]*len(df))),3),
                                                                        "CramerV",df=stats.chi2_contingency(value_23)[2])
                                              ],
                                             [[-1, 1], write_effect_size(round(np.sqrt(stats.chi2_contingency(value_13)[0]/(stats.chi2_contingency(value_13)[2]*len(df))),3),
                                                                         "CramerV",df=stats.chi2_contingency(value_13)[2])]
                                             ]
            #print(dic_post_hoc)

        cramerV = np.sqrt(chi2[0]/(chi2[2]*len(df)))   # source : https://www.statology.org/effect-size-chi-square/
        dic_output[col] = [value[i][j] for i in range(len(value)) for j in range(len(value[0])) ]
        dic_output[col].extend(missing)
        dic_output[col].append(chi2[0])#Stat
        dic_output[col].append(chi2[1])#p-val
        dic_output[col].append( write_effect_size(round(cramerV,3), "CramerV", df=chi2[2]))
    print(dic_output)
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
                          "maxs" : [], "missing":[], "perc":[], "post-hoc":-1, "post-hoc_eff":-1}
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
            dic_cols[col]["eta-kruskal"] = (dic_cols[col]["F-stat"]- 3 +1)/(len(li_df[0]) + len(li_df[1]) + len(li_df[2]) - 3)
            if dic_cols[col]["pval"] < 0.05:
                # print(col)

                #print("")
                #print(dic_cols[col]["order_vals"])
                #print(dic_cols[col]["means"])
                dunn_test = sp.posthoc_dunn([list(li_df[0]), list(li_df[1]), list(li_df[2])])
                #print(dunn_test)
                dic_cols[col]["post-hoc"] = [
                    [[dic_cols[col]["order_vals"][0], dic_cols[col]["order_vals"][1]], dunn_test[1][2]],
                    [[dic_cols[col]["order_vals"][1], dic_cols[col]["order_vals"][2]],
                     dunn_test[2][3]],
                    [[dic_cols[col]["order_vals"][0], dic_cols[col]["order_vals"][2]],
                     dunn_test[1][3]]]


                # a modifier : ici mettre à la place des 0 l'effect size de mann-whitney
                dic_cols[col]["post-hoc_eff"] = [
                    [[dic_cols[col]["order_vals"][0], dic_cols[col]["order_vals"][1]], round(cohen_d(list(li_df[0]), list(li_df[1])),3)],
                    [[dic_cols[col]["order_vals"][1], dic_cols[col]["order_vals"][2]], round(cohen_d(list(li_df[1]), list(li_df[2])),3)],
                    [[dic_cols[col]["order_vals"][0], dic_cols[col]["order_vals"][2]], round(cohen_d(list(li_df[0]), list(li_df[2])),3)]]


                #print(dic_cols[col]["post-hoc"])
        if len(li_df) == 2: #Beware : hardcoded, would only be useful if loche database
            dic_cols[col]["F-stat"], dic_cols[col]["pval"] = stats.kruskal(li_df[0], li_df[1])

    return dic_cols

def create_feature_set_of_rows_num(col, dic_cols, np2, db=""):
    nb_split = len(dic_cols[col]["order_vals"])

    good_order = list(np.argsort(dic_cols[col]["order_vals"])) #should give the good order to parse the datas
    #print(good_order)
    set_of_rows = []
    if db == "SM":
        first_row = [config.SM_str_fields[col][0]]
    elif db == "JS" :
        first_row = [config.JS_str_fields[col][0]]
    else:
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
    first_row.append("H = " + str(round(dic_cols[col]["F-stat"],1)))
    if dic_cols[col]["pval"] < 0.05:
        first_row[-1] = first_row[-1] +"*"
    first_row.append(dic_cols[col]["pval"])
    first_row.append(write_effect_size(round(np2,3), "kruskal"))
    first_row.append("")
    first_row.append("")
    first_row.append("")
    for i in range(6):
        mean_line.append("")
        median_line.append("")
        missing_line.append("")
    set_of_rows.append(first_row)
    set_of_rows.append(mean_line)
    set_of_rows.append(median_line)
    set_of_rows.append(missing_line)
    #print(set_of_rows)
    return set_of_rows, dic_cols[col]["pval"]

def create_feature_set_of_rows_cat(col, dic_df, data, dic_crosstab, db=""): #se méfier des missing values !! #supprimer les nan comme catégorie

    nb_split = len(list(dic_df.keys()))
    choices = list(data[col].unique())
    choices_2 =[]
    for elm in choices:
        if str(elm)!="nan":
            choices_2.append(elm)
    choices_2 = sorted(choices_2)
    set_of_rows = []
    if db == "SM":

        first_row = [config.SM_str_fields[col][0]]
        many_line = [[config.SM_str_fields[col][1][int(choices_2[i])]] for i in range(int((len(dic_crosstab[col]) - 9) / 3))]
    elif db == "JS" :
        print(col)
        print(config.JS_str_fields[col][1])
        first_row = [config.JS_str_fields[col][0]]
        many_line = [[config.JS_str_fields[col][1][int(choices_2[i])]] for i in range(int((len(dic_crosstab[col]) - 9) / 3))]
    else:
        first_row = [col]
        many_line = [[choices_2[i]] for i in range(int((len(dic_crosstab[col]) - 9) / 3))]

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
        #print("relecture crosstab")
        #print(dic_crosstab[col])
        missing_line.append(
            str(dic_crosstab[col][i-9]) + " (" + str(round(dic_crosstab[col][i-6], 1)) + "%)")

    first_row.append("Chi2 = " + str(round(dic_crosstab[col][-3], 1)))
    if dic_crosstab[col][-2] < 0.05:
        first_row[-1] = first_row[-1] + "*"
    first_row.append(dic_crosstab[col][-2])
    first_row.append(dic_crosstab[col][-1])
    first_row.append("")
    first_row.append("")
    first_row.append("")
    for i in range(6):
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
        if row[-6]!="" and row[-6]!="Statistic":
            col = row[-3]
            if row[-5] < 0.05 :

                if num_cols.__contains__(col):
                    ex_col = col
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

    #pour SM ['Worsens VS No effect', 'Improves VS No effect', 'Worsens VS Improves']
    #Pour JS ['Worsens VS No effect', 'Worsens VS Improves', 'Improves VS No effect']
    #second step : writing p_vals in table
    for i in range(len(table)):
        row = table[i]
        if row[-6]!="" and row[-6]!="Statistic":
            col = row[-3]
            row[-3] = ""
            if row[-5] < 0.05 :


                #ici rajouter l'effect size en bout de ligne, penser à rajouter un elm de plus à table et à décaler de 1 la valeur des indices dans table dans les appels harcoded
                if num_cols.__contains__(col):

                    table[i+1][-3]="Worsens VS No effect"
                    table[i + 1][-2] = write_p_val(p_adj[dic_cols[col]["post-hoc"][0][1]])
                    table[i + 1][-1] = write_effect_size(dic_cols[col]["post-hoc_eff"][0][1], "cohen")
                    table[i + 2][-3] = "Improves VS No effect"
                    table[i + 2][-2] = write_p_val(p_adj[dic_cols[col]["post-hoc"][1][1]]) #attention: indices inversés, surveiller dic_col
                    table[i + 2][-1] = write_effect_size(dic_cols[col]["post-hoc_eff"][1][1], "cohen")
                    table[i + 3][-3] = "Worsens VS Improves"
                    table[i + 3][-2] = write_p_val(p_adj[dic_cols[col]["post-hoc"][2][1]]) #attention: indices inversés, surveiller dic_col
                    table[i + 3][-1] = write_effect_size(dic_cols[col]["post-hoc_eff"][2][1], "cohen")

                if cat_cols.__contains__(col):

                    table[i+1][-3]="Worsens VS No effect"
                    table[i + 1][-2] = write_p_val(p_adj[dic_cols2[col]["post-hoc"][0][1]])
                    table[i + 1][-1] = dic_cols2[col]["post-hoc_eff"][0][1]
                    table[i + 2][-3] = "Improves VS No effect"
                    table[i + 2][-2] = write_p_val(p_adj[dic_cols2[col]["post-hoc"][1][1]])
                    table[i + 2][-1] = dic_cols2[col]["post-hoc_eff"][1][1]
                    table[i + 3][-3] = "Worsens VS Improves"
                    table[i + 3][-2] = write_p_val(p_adj[dic_cols2[col]["post-hoc"][2][1]])
                    table[i + 3][-1] = dic_cols2[col]["post-hoc_eff"][2][1]
    return table

def write_p_val(pval):
    output = 0
    if pval<0.001:
        output=" p < 0.001"
    else :
        output=round(pval, 3)
    return(output)

def write_effect_size(eff, stat_test,df=0):
    if stat_test=="cohen":
        if eff < 0.2 :
            eff = str(eff) + " (Negligible)"
        elif eff < 0.5:
            eff = str(eff) + " (Small)"
        elif eff < 0.8:
            eff = str(eff) + " (Medium)"
        elif eff >= 0.8:
            eff = str(eff) + " (Large)"

    if stat_test == "kruskal":
        if eff < 0.01 :
            eff = str(eff) + " (Negligible)"
        elif eff < 0.06:
            eff = str(eff) + " (Small)"
        elif eff < 0.14:
            eff = str(eff) + " (Medium)"
        elif eff >= 0.14:
            eff = str(eff) + " (Large)"

    if stat_test == "CramerV":
        if df==0:
            return eff
        else:
            if df==1:
                if eff < 0.1 :
                    eff = str(eff) + " (Negligible)"
                elif eff < 0.3:
                    eff = str(eff) + " (Small)"
                elif eff < 0.5:
                    eff = str(eff) + " (Medium)"
                elif eff >= 0.5:
                    eff = str(eff) + " (Large)"
            elif df==2:
                if eff < 0.07 :
                    eff = str(eff) + " (Negligible)"
                elif eff < 0.21:
                    eff = str(eff) + " (Small)"
                elif eff < 0.35:
                    eff = str(eff) + " (Medium)"
                elif eff >= 0.35:
                    eff = str(eff) + " (Large)"
            elif df==3:
                if eff < 0.06 :
                    eff = str(eff) + " (Negligible)"
                elif eff < 0.17:
                    eff = str(eff) + " (Small)"
                elif eff < 0.29:
                    eff = str(eff) + " (Medium)"
                elif eff >= 0.29:
                    eff = str(eff) + " (Large)"
            elif df==4:
                if eff < 0.05 :
                    eff = str(eff) + " (Negligible)"
                elif eff < 0.15:
                    eff = str(eff) + " (Small)"
                elif eff < 0.25:
                    eff = str(eff) + " (Medium)"
                elif eff >= 0.25:
                    eff = str(eff) + " (Large)"
            elif df >= 5:
                if eff < 0.04 :
                    eff = str(eff) + " (Negligible)"
                elif eff < 0.13:
                    eff = str(eff) + " (Small)"
                elif eff < 0.22:
                    eff = str(eff) + " (Medium)"
                elif eff >= 0.22:
                    eff = str(eff) + " (Large)"


    return eff
#rajouter une fonction qui prend en entrée 2 arg : valeur de l'effect size et type de l'effect size et qui renvoie en str
# le meme nb mais avec après marqué entre () (small), (negligible) etc...

