import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load the saved model
with open('catboost_tuned_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
    
#%%

df = pd.read_csv('Release Radar.csv')

def popularity_calculator(df):
    
    for col in ['Acousticness', 'Instrumentalness']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[~(df['Genres'].isna())]
    df = df[~(df['Instrumentalness'].isna())]
    df = df[~(df['Acousticness'].isna())]
    
    
    
    # Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
    
    def grab_col_names(dataframe, cat_th=10, car_th=20):
        """
    
        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    
        Parameters
        ------
            dataframe: dataframe
                    Değişken isimleri alınmak istenilen dataframe
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optinal
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri
    
        Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi
    
        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))
    
    
        Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.
            Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    
        """
    
        # cat_cols, cat_but_car
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                       dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]
    
        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')
        return cat_cols, num_cols, cat_but_car
    
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    
    df['pop'] = 0
    for index, each in enumerate(df['Genres']):
        if "pop" in str(each):
            df['pop'][index] = 1
        
    df['rock'] = 0
    for index, each in enumerate(df['Genres']):
        if "rock" in str(each):
            df['rock'][index] = 1
        
    
    df['r_b'] = 0
    for index, each in enumerate(df['Genres']):
        if "r_b" in str(each):
            df['r_b'][index] = 1
        
    df['hiphop'] = 0
    for index, each in enumerate(df['Genres']):
        if "hiphop" in str(each):
            df['hiphop'][index] = 1
    
    df['reggaeton'] = 0
    for index, each in enumerate(df['Genres']):
        if "reggaeton" in str(each):
            df['reggaeton'][index] = 1
            
    
    df['latin'] = 0
    for index, each in enumerate(df['Genres']):
        if "latin" in str(each):
            df['latin'][index] = 1
            
    df['rap'] = 0
    for index, each in enumerate(df['Genres']):
        if "rap" in str(each):
            df['rap'][index] = 1
            
    df['rock'] = 0
    for index, each in enumerate(df['Genres']):
        if "rock" in str(each):
            df['rock'][index] = 1
            
    df['trap'] = 0
    for index, each in enumerate(df['Genres']):
        if "trap" in str(each):
            df['trap'][index] = 1
    
    
    df["NEW_ENERGY_DANCEABILITY"] = df["Danceability"] * df["Energy"]
    df["NEW_SPEECHINESS_LIVENESS"] =  df["Speechiness"] * df["Liveness"]
    df["NEW_TEMPO_VALENCE"] = df["Tempo"] * df["Valence"]
    df["NEW_POPULARITY_ENERGY"] = df["Popularity"] * df["Energy"]
    df["NEW_VALENCE_TEMPO"] = df["Valence"] * df["Tempo"]
    df["NEW_ENERGY_TEMPO"] = df["Energy"] * df["Tempo"]
    df["NEW_VALENCE_ENERGY"] = df["Valence"] * df["Energy"]
    df["NEW_DANCEABILITY_TEMPO"] = df["Danceability"] * df["Tempo"]
    df["NEW_ACOUSTICNESS_LIVENESS_ENERGY"] = (df["Acousticness"] * df["Liveness"]) / df["Energy"]
    
    
    
    ## group or single
    df['single_or_group'] = 0
    for index, each in enumerate(df['Artist IDs']):
        if "," in each:
            df['single_or_group'][index] = 1
    
    
    df['one_or_multiple_genra'] = 0
    for index, each in enumerate(df['Genres']):
        if "," in each:
            df['single_or_group'][index] = 1
    
    
    
    
    df = df.drop(['Artist Name(s)',"Added By","Added At",'Genres',
                  "Album Name","Artist IDs","Release Date","Spotify ID","Track Name",], axis=1)
    
    def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
        quantile_one = dataframe[variable].quantile(low_quantile)
        quantile_three = dataframe[variable].quantile(up_quantile)
        interquantile_range = quantile_three - quantile_one
        up_limit = quantile_three + 1.5 * interquantile_range
        low_limit = quantile_one - 1.5 * interquantile_range
        return low_limit, up_limit
    
    def check_outlier(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
            return True
        else:
            return False
    
    for col in num_cols:
        if col != "Popularity":
          print(col, check_outlier(df, col))
    
    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    
    
    for col in num_cols:
        if col != "popularity":
            replace_with_thresholds(df,col)
    
    
    def onehot_encode(df, column, prefix):
        df = df.copy()
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
        return df
    
    
    
    ###### rare encoder ######
    def rare_encoder(dataframe, rare_perc):
        temp_df = dataframe.copy()
    
        rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                        and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    
        for var in rare_columns:
            tmp = temp_df[var].value_counts() / len(temp_df)
            rare_labels = tmp[tmp < rare_perc].index
            temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    
        return temp_df
    
    
    rare_encoder(df,0.01)
    
    
    df = df.dropna()
    ##### MODEL #####
    y = df.loc[:, 'Popularity']
    X = df.drop('Popularity', axis=1)
    
    from sklearn.preprocessing import MinMaxScaler
    
    # Assuming X is your feature matrix
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)


    X_test = X.copy()
    # Now, you can use the loaded_model for predictions
    predictions = loaded_model.predict(X_test)
    print(predictions)

    
#%%

popularity_calculator(df)


