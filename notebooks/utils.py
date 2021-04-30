import numpy as np
import pandas as pd
import category_encoders as ce


def make_tempo_features(df, key="tempo"):

    tempo = df.copy()[key]

    # min, max
    tempo = tempo.str.split("-", expand=True)
    tempo.columns = ["tempo_min", "tempo_max"]
    tempo = tempo[["tempo_min", "tempo_max"]].astype(int)

    # sum
    tempo["tempo_sum"] = tempo["tempo_min"] + tempo["tempo_max"]
    # diff
    tempo["tempo_diff"] = tempo["tempo_max"] - tempo["tempo_min"]
    # mean
    tempo["tempo_mean"] = tempo["tempo_sum"] / 2

    return tempo


# def make_region_features(df, key="region"):

#     region = df.copy()[key]
#     region = pd.get_dummies(region)

#     _df = pd.concat([df, region], axis=1)
#     return _df


def make_popularity_features(df, key="popularity"):

    _popularity = df.copy()[key]
    _popularity = _popularity.astype(str).apply(lambda x : x.zfill(2))

    popularity = pd.DataFrame({
        "popularity01": _popularity.apply(lambda x: x[-1]),
        "popularity10": _popularity.apply(lambda x: x[-2]),

    })
    return popularity.astype(int)


def max_min(x):
    return max(x) - min(x)

def q75_q25(x):
    return x.quantile(0.75) - x.quantile(0.25)

def z_score(x):
    return 0.0

def _make_key_agg(df, key):

    # 前処理
    if "genre" in df.columns:
        df = df.drop(["genre"], axis=1)
    df_columns = df.columns

    # 集約特徴量作成
    agg_methods = ["min", "mean", "max", "median", "std", max_min, q75_q25, z_score]
    key_agg = df.groupby(key).agg(agg_methods)

    # 適切な列名を作成
    key_agg_columns = []
    for col in key_agg.columns.levels[0]:
        for stat in key_agg.columns.levels[1]:
            key_agg_columns.append(f"agg_{stat}_{col}_groupby_{key}")
    key_agg.columns = key_agg_columns

    return key_agg


def make_agg_region_features(df, key="region"):
    key_agg = _make_key_agg(df, key)

    # 出力するdfを作成
    _df = pd.DataFrame({})
    for i, k in enumerate(df[key]):
        if i == 0:
            _df = key_agg.loc[k]
        else:
            _df = pd.concat([_df, key_agg.loc[k]], axis=1)
    _df = _df.T.reset_index(drop=True)

    # z_scoreを追加
    i = 0
    for col in _df.columns:
        if "z_score" in col:
            feature_name = df.columns[i]
            _df[col] = (df[feature_name].values - _df[f"agg_mean_{feature_name}_groupby_{key}"].values) / (_df[f"agg_std_{feature_name}_groupby_{key}"].values + 1e-3)
            i += 1

    return _df


def make_ce_features(df):
    # count encodingした特徴量
    _df = pd.concat([df, make_popularity_features(df)], axis=1)
    cols = ["region", "popularity10"]
    encoder = ce.CountEncoder()
    return encoder.fit_transform(_df[cols]).add_prefix("CE_")


def make_oe_features(df):
    # ordinal encording (label encoding)した特徴量
    cols = ["region"]
    ce_encoder = ce.CountEncoder()
    return ce_encoder.fit_transform(df[cols]).add_prefix("CE_")


def make_numerical_features(df):
    # そのままの数値特徴
    cols = ['popularity',
            'duration_ms',
            'acousticness',
            'positiveness',
            'danceability',
            'loudness',
            'energy',
            'liveness',
            'speechiness',
            'instrumentalness']
    return df[cols].copy()


def preprocess(df, funcs):
    df_list = [func(df) for func in funcs]
    _df = pd.concat(df_list, axis=1)
    return _df


def get_train_data(train, test):
    # each_funcs: trainのみを対象とした処理
    each_funcs = [make_numerical_features,
                  make_tempo_features,
                  make_popularity_features]

    # whole_funcs: train+testの全体集合を対象とした処理
    whole_funcs = [make_ce_features,
                   make_oe_features,
                   make_agg_region_features]
    whole_df = pd.concat([train, test]).reset_index(drop=True)

    train_out = preprocess(train, each_funcs)
    print("------------------")
    whole_out = preprocess(whole_df, whole_funcs)

    X_train = pd.concat([train_out, whole_out.iloc[:len(train)]], axis=1)
    return X_train

def get_test_data(train, test):
    # each_funcs: testのみを対象とした処理
    each_funcs = [make_numerical_features,
                  make_tempo_features,
                  make_popularity_features]

    # whole_funcs: train+testの全体集合を対象とした処理
    whole_funcs = [make_ce_features,
                   make_oe_features,
                   make_agg_region_features]
    whole_df = pd.concat([train, test]).reset_index(drop=True)

    train_out = preprocess(test, each_funcs)
    whole_out = preprocess(whole_df, whole_funcs)

    X_test = pd.concat([test_out, whole_out.iloc[len(test):].reset_index(drop=True)], axis=1)
    return X_test