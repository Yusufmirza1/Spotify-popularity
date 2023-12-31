# track_popularity (Parça Popülerliği): Parçanın çalınma sayısına dayanarak 0 ile 100 arasında ölçülen popülerlik.
# acousticness (Akustiklik): Parçanın akustik olup olmadığını 0.0 ile 1.0 arasında ölçen bir değer. 1.0, parçanın yüksek olasılıkla akustik olduğunu gösterir.
# danceability (Dans Edilebilirlik): Parçanın dans için uygunluğunu tempo, ritim stabilitesi, vuruş gücü ve genel düzenlilik gibi müziksel öğelerin bir kombinasyonuna dayanarak açıklayan bir değer. 0.0 en düşük dans edilebilirlik, 1.0 en yüksek dans edilebilirlik.
# duration_ms (Süre): Parçanın süresi milisaniye cinsinden.
# energy (Enerji): Algısal bir ölçü olan enerji, genellikle hızlı, yüksek sesli ve gürültülü hissedilen bir parçanın yoğunluğunu ve aktivitesini temsil eder. Değerler 0.0 ile 1.0 arasında.
# instrumentalness (Enstrümantalite): Bir parçanın vokal içerip içermediğini ölçen bir değer. Değer ne kadar yakınsa, parçanın vokal içerme olasılığı o kadar düşüktür.
# key (Ton): Parçanın tahmini genel tonu. -1 ise tonun belirlenemediği anlamına gelir.
# key (Ton):Anahtar, bir şarkının temelini oluşturan ton, notalar veya ölçüdür. 0 ile 11 arasında değişen 12 anahtar bulunmaktadır.
# liveness (Canlılık): Kayıtta bir izleyici varlığını algılar. Değer ne kadar yüksekse, parçanın canlı performans olma olasılığı o kadar yüksektir.
# loudness (Yükseklik): Parçanın genel yüksekliği desibel (dB) cinsinden.
# mode (Mod): Mode: Sayısal olarak, mod, bir parçanın modasını (majör veya minör) gösterir; melodi içeriğinin türetildiği perde türünü belirtir. Majör, 1 ile temsil edilirken, minör 0 ile temsil edilir.
# speechiness (Konuşma Benzerliği): Parçada konuşma benzeri sesleri algılar. 1.0'a ne kadar yakınsa, kaydın büyük olasılıkla tamamen konuşma içerdiği anlamına gelir.
# tempo (Tempo): Parçanın genel tahmini temposu dakikadaki vuruş sayısında (BPM).
# valence (Valans): Parçanın ilettiği müzikal olumluluk. Yüksek valanslı parçalar daha olumlu (mutlu, neşeli, coşkulu), düşük valanslı parçalar daha olumsuz (üzgün, bunalımlı, öfkeli) hissettirir.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
#%%

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)

# csv verisini okuyunuz.
df = pd.read_csv("spotify_all_data.csv")



#%%

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
#%%
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

#%%%

df["NEW_ENERGY_DANCEABILITY"] = df["Danceability"] * df["Energy"]

df["NEW_SPEECHINESS_LIVENESS"] =  df["Speechiness"] * df["Liveness"]

df["NEW_TEMPO_VALENCE"] = df["Tempo"] * df["Valence"]

df["NEW_POPULARITY_ENERGY"] = df["Popularity"] * df["Energy"]

# df["NEW_DURATION_LOUNDNESS"] = df["Duration (ms)"] * df["Loudness"]

df["NEW_VALENCE_TEMPO"] = df["Valence"] * df["Tempo"]

df["NEW_ENERGY_TEMPO"] = df["Energy"] * df["Tempo"]

df["NEW_VALENCE_ENERGY"] = df["Valence"] * df["Energy"]

df["NEW_DANCEABILITY_TEMPO"] = df["Danceability"] * df["Tempo"]

# df["NEW_DURATION_SPEECHINESS_ENERGY"] = df["Duration (ms)"] + df["Speechiness"] / df["Energy"]

df["NEW_ACOUSTICNESS_LIVENESS_ENERGY"] = (df["Acousticness"] * df["Liveness"]) / df["Energy"]

# df["NEW_DANCEABILITY_TEMPO_VALENCE"] = (df["Danceability"] + df["Tempo"]) / df["Valence"]



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
              "Album Name","Artist IDs","Release Date","Spotify ID","Track Name",'top50date','top50datesecond'], axis=1)


#%%

# koorelasyon matrisi
music_features = ['Popularity',"Key", 'Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo',"Duration (ms)"]
correlation_matrix = df[music_features + ['Time Signature',"Mode","top50"]].corr()

correlation_matrix

#%%

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
#%%

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "popularity":
        replace_with_thresholds(df,col)

#%%
#"Energy" korelasyon ardından kaldırılabilir.

#df = df.drop(["top50","Energy"], axis=1)
# df['Popularity'] = pd.qcut(df['Popularity'], q=2, labels=[0, 1])
def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

# df = onehot_encode(df, 'Genres', 'genre')

#%%

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
#%%
##### MODEL #####
y = df.loc[:, 'Popularity']
X = df.drop('Popularity', axis=1)

from sklearn.preprocessing import MinMaxScaler

# Assuming X is your feature matrix
scaler = MinMaxScaler()
X = scaler.fit_transform(X)



# X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

#%%

models = [('LR', LinearRegression()),
          #("Ridge", Ridge()),
          #("Lasso", Lasso()),
          #("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          #('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


############(hangi değişkenin ne kadar değer kattığı gösterir)######
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = XGBRegressor(objective='reg:squarederror')
model.fit(X, y)
plot_importance(model, X)

#df = df.drop(["top50","Energy"], axis=1) ile çalıştığında
#RMSE: 0.4827 (LightGBM)
#RMSE: 63196485811859.89 (LR)
#RMSE: 0.5478 (KNN)
#RMSE: 0.6499 (CART)
#RMSE: 0.4874 (RF)
#RMSE: 0.4831 (GBM)
#RMSE: 0.4876 (XGBoost)


#RMSE: 66516945723481.0 (LR)
#RMSE: 0.5454 (KNN)
#RMSE: 0.6491 (CART)
#RMSE: 0.4842 (RF)
#RMSE: 0.4813 (GBM)
#RMSE: 0.486 (XGBoost)
#RMSE: 0.4798 (LightGBM)
#RMSE: 0.4801 (CatBoost)


#%%

from sklearn.metrics import r2_score



def rmse_r2_calculater(X,y,alg):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)
    model = alg().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    RMSE_ = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE = f'{RMSE_:.5f}'[:-1]
    r2_= r2_score(y_test,y_pred)
    r2 = f'{r2_:.5f}'[:-1]
    model_name = alg.__name__
    print(model_name, "RMSE Score:",RMSE,'R^2:',r2)
    rmse.append(RMSE_)
    r_2.append(r2_)

    
rmse = [] 
r_2 = []

l_models = [
          Ridge,
          LinearRegression,
          Lasso,
          ElasticNet
         ]

for i in l_models:
    rmse_r2_calculater(X,y, i)
    
nl_models = [GradientBoostingRegressor, 
          RandomForestRegressor, 
          DecisionTreeRegressor,
          KNeighborsRegressor, 
          SVR,
          XGBRegressor,
          CatBoostRegressor,
          LGBMRegressor

         ]
for i in nl_models:
    rmse_r2_calculater(X,y, i)
    
    

df_rmse = pd.DataFrame (rmse, columns = ['RMSE Score'])
df_r2 = pd.DataFrame (r_2, columns = ['R^2 Score'])

model = ['Ridge',
          'LinearRegression',
          'Lasso',
          'ElasticNet',
          'GradientBoostingRegressor', 
          'RandomForestRegressor', 
          'DecisionTreeRegressor',
          'KNeighborsRegressor', 
          'SVR',
          'XGBRegressor',
          'CatBoostRegressor',
          'LGBMRegressor'
        ]
df_models = pd.DataFrame (model, columns = ['Models'])
df_rmse_r2= pd.concat([df_rmse,df_r2,df_models],axis=1)
df_rmse_r2


#%%


sns.barplot(x= 'RMSE Score', y = 'Models', data=df_rmse_r2, color= 'g')
plt.xlabel('RMSE Score')
plt.title('RMSE score of the models');


#%%

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)
model = CatBoostRegressor().fit(X_train, y_train)
y_pred = model.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
RMSE = f'{RMSE:.5f}'[:-1]
r2= r2_score(y_test,y_pred)
r2 = f'{r2:.5f}'[:-1]
print("model_name", "RMSE Score:",RMSE,'R^2:',r2)

#%%
feature_imp = pd.Series(model.feature_importances_,
                        ).sort_values(ascending=False)
#%%
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Importance Score of Variable')
plt.ylabel('Variables')
plt.title("Importance Level of Variable")
plt.show()

#%%
sns.scatterplot(y_test, y_pred)
plt.xlabel('Actual ')
plt.ylabel('Predicted ')
plt.plot([0, 100], [0, 100], color='red', lw=3) #Plot a diagonal length (Popularity variable can be between 0-100)
plt.show()


#%%
eror=y_test - y_pred
sns.scatterplot(range(len(eror)),eror)
plt.plot([0, 33], [0, 0], color='red', lw=3) #Plot a diagonal length
plt.xlabel('Test Variables')
plt.ylabel('Residual')
plt.title("Residual Table")


#%%

tuplee = (X_test,y_test)

model = CatBoostRegressor(iterations=10000,random_state=42).fit(X_train, y_train,plot=True,eval_set = tuplee)

#%%


catb_model_setted = CatBoostRegressor().fit(X_train, y_train,early_stopping_rounds=9990)
y_pred = catb_model_setted.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#%%
r2_score(y_test,y_pred)
#%%

catb_params = {'depth': [6,None],
              'learning_rate' : [0.01, 0.03,0.05, 0.1],
              # 'iterations'    : [None,1000, 5000, 10000]
                 }

catb_model = CatBoostRegressor(random_state=42).fit(X_train, y_train,eval_set = tuplee,early_stopping_rounds=9990)
#%%
catb_cv_model = GridSearchCV(catb_model, 
                           catb_params, 
                           cv = 5, 
                           n_jobs = -1, 
                           verbose = 2).fit(X_train, y_train)

#%%

best_params = catb_cv_model.best_params_
#%%
catb_tuned = CatBoostRegressor(depth = 6, learning_rate = 0.05).fit(X_train, y_train,early_stopping_rounds=9990)
#%%

y_pred_train_tuned = catb_tuned.predict(X_train)
rmse_train_tuned=np.sqrt(mean_squared_error(y_train, y_pred_train_tuned))


#%%
y_pred = catb_tuned.predict(X_test)
rmse_test=np.sqrt(mean_squared_error(y_test, y_pred))

r2_test = r2_score(y_test,y_pred)
#%%

import pickle

# Save the model to a file using pickle
with open('catboost_tuned_model.pkl', 'wb') as catb_tuned:
    pickle.dump(model, catb_tuned)

