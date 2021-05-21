import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def dummies_city_relevent_experience(df):
    df1 = pd.get_dummies(df[['city', 'relevent_experience']])
    df1 = pd.DataFrame(df1, index=df.index)
    df = df.drop(['city', 'relevent_experience'], axis=1)
    df = pd.concat([df, df1], axis=1)
    return df

def impute_experience(df, cat_var):
    df['experience'] = df['experience'].replace(['<1'], 0)
    df['experience'] = df['experience'].replace(['>20'], 21)
    df1 = df
    df1 = df.drop(cat_var + list(['last_new_job']), axis=1)
    imputer = KNNImputer()
    df1_imputed = imputer.fit_transform(df1)
    df1_imputed = pd.DataFrame(df1_imputed, index=df1.index, columns=df1.columns)

    bins = np.linspace(0, 25, 6)
    labels = ['exp_one', 'exp_two', 'exp_three', 'exp_four', 'exp_five']
    df1_imputed['exp_bins'] = pd.cut(df1_imputed['experience'], bins=bins, labels=labels)
    df2 = pd.get_dummies(df1_imputed['exp_bins'])
    df = df.drop(['experience'], axis=1)
    df = pd.concat([df, df2], axis=1)
    return df

def impute_last_new_job(df, cat_var):
    df['last_new_job'] = df['last_new_job'].replace(['never'], 0)
    df['last_new_job'] = df['last_new_job'].replace(['>4'], 5)
    df1 = df
    df1 = df.drop(cat_var, axis=1)
    imputer = KNNImputer()
    df1_imputed = imputer.fit_transform(df1)
    df1_imputed = pd.DataFrame(df1_imputed, index=df1.index, columns=df1.columns)

    bins = np.linspace(-1, 5, 7)
    labels = ['lnj_zero', 'lnj_one', 'lnj_two', 'lnj_three', 'lnj_four', 'lnj_five']
    df1_imputed['lnj_bins'] = pd.cut(df1_imputed['last_new_job'], bins=bins, labels=labels)
    df2 = pd.get_dummies(df1_imputed['lnj_bins'])
    df = df.drop(['last_new_job'], axis=1)
    df = pd.concat([df, df2], axis=1)
    return df

def find_category_mappings(df, variable):
    return {k: i for i, k in enumerate(df[variable].dropna().unique(), 0)}

def integer_encode(df , variable, ordinal_mapping):
    df[variable] = df[variable].map(ordinal_mapping)

mm = MinMaxScaler()
mappin = dict()
def imputations(df1, cols):
    df = df1.copy()
    for variable in cols:
        mappings = find_category_mappings(df, variable)
        mappin[variable] = mappings

    for variable in cols:
        integer_encode(df, variable, mappin[variable])

    sca = mm.fit_transform(df)
    knn_imputer = KNNImputer()
    knn = knn_imputer.fit_transform(sca)
    df.iloc[:, :] = mm.inverse_transform(knn)
    for i in df.columns:
        df[i] = round(df[i]).astype('int')

    for i in cols:
        inv_map = {v: k for k, v in mappin[i].items()}
        df[i] = df[i].map(inv_map)
    return df

def impute_categorical_variable(df, cat_var):
    df_cat = df[cat_var]
    df_imputed = imputations(df_cat, cat_var)
    df = df.drop(cat_var, axis=1)
    df = pd.concat([df, df_imputed], axis=1)
    df1 = pd.get_dummies(df[cat_var])
    df1 = pd.DataFrame(df1, index=df.index)
    df = df.drop(cat_var, axis=1)
    df = pd.concat([df, df1], axis=1)
    return df

if __name__ == '__main__':
    df_train = pd.read_csv("aug_train.csv")
    df_test = pd.read_csv("aug_test.csv")
    df = pd.concat([df_test, df_train])
    df = df.set_index('enrollee_id')

    cat_var = ['gender', 'enrolled_university', 'education_level', 'major_discipline', 'company_size', 'company_type']
    df = dummies_city_relevent_experience(df)
    df = impute_experience(df, cat_var)
    df = impute_last_new_job(df, cat_var)
    df = impute_categorical_variable(df, cat_var)
    df_test1 = df[0:len(df_test)]
    df_train1 = df[len(df_test):]
    df_test1.to_csv("df_test.csv")
    df_train1.to_csv("df_train.csv")
