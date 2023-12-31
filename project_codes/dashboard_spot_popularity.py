#Import Libraries
import pandas as pd
import dash
from dash import dcc
from dash import html
import plotly.express as px
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import webbrowser
import pickle

with open('catboost_tuned_model.pkl', 'rb') as file:
    model = pickle.load(file)

##Open The browser
webbrowser.open("http://127.0.0.1:8050/")


musical_features = [
    'Acousticness', 'Danceability', 'Duration_ms', 'Energy',
    'Instrumentalness', 'Key', 'Liveness', 'Loudness',
    'Mode', 'Tempo', 'Valence', "Speechiness "
]

##Create app 
app = dash.Dash(__name__ , external_stylesheets=[dbc.themes.DARKLY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])

app.layout = html.Div([
    # Sol Sütun
    html.Div([
        html.H2("SPOTIFY ŞARKI POPÜLERLİĞİ", style={'textAlign': 'center'}),
        html.Img(id='image', src='https://media-cdn.t24.com.tr/media/library/2021/03/1616921843361-1.jpg',
                 style={'width': '80%', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
        html.P("""Müzik dünyasında yeni bir sayfa açan uygulamamız, sanatçıların yaratıcılıklarını destekleyerek geleceklerini aydınlatıyor. 
                      Geliştirdiğimiz akıllı algoritma, Spotify üzerinde yeni yayınlanan şarkıların potansiyel popülerliğini analiz ediyor ve gelecek 
                      vaat eden hitlerin izini sürüyor. Bu teknoloji, müzik endüstrisindeki trendleri öngörerek sanatçıların kariyerlerini daha bilinçli 
                      şekilde yönlendirmelerine olanak tanıyor.""",
               style={'textAlign': 'center', 'fontWeight': 'bold'}),
        html.P("Nasıl Çalışır?", style={'textAlign': 'center', 'fontWeight': 'bold'}),
        html.P("""Uygulamamız, çeşitli müzikal özellikler ve geçmiş verileri temel alarak, bir şarkının Spotify'da ne kadar popüler olabileceğini 
                      tahmin ediyor. Bu süreç, şarkının ritmi, melodisi, sözlerinin yapısı ve sanatçının mevcut popülerliğini de içeriyor. Yapay zeka ve 
                      makine öğrenimi teknolojilerini kullanarak, her şarkının benzersiz özelliklerini detaylı bir şekilde analiz ediyoruz.""",
               style={'textAlign': 'center'}),
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'}),


    # Sağ Sütun
    html.Div([
        # Müzikal özellikler için girdi alanları
        html.Div(id='input-container-left', children=[
            html.Div([
                html.Label(feature.capitalize()),
                dcc.Input(id=f'input-{feature}', type='number', placeholder=feature.capitalize())
                if feature != 'mode' else
                dcc.RadioItems(
                    id='input-mode',
                    options=[{'label': 'Major (1)', 'value': 1}, {'label': 'Minor (0)', 'value': 0}],
                    value=1,
                    inline=True
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginBottom': '15px'})
            for feature in musical_features
        ], style={'textAlign': 'left'}),




        # Tek "Gönder" butonu
        dbc.Button('Gönder', id='submit-button', n_clicks=0, className='mt-3', style={'width': '100%'}),

        # Çıktı konteyneri
        html.Div(id='output-container')
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top',
              'boxSizing': 'border-box'}),

    # Modal yapısı
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Popülerlik Oranı")),
            dbc.ModalBody(html.Div(id='popularity-score')),
            dbc.ModalFooter(
                dbc.Button("Kapat", id="close-modal", className="ms-auto", n_clicks=0)
            ),
        ],
        id="modal-popularity",
        is_open=False,
    )
])

# Modalı açma ve kapama işlevi için callback

@app.callback(
    Output('modal-popularity', 'is_open'),
    [Input('submit-button', 'n_clicks'), Input('close-modal', 'n_clicks'),
    Input('modal-popularity', 'is_open')],
    prevent_initial_call=True
)


def toggle_modal(submit_n_clicks, close_n_clicks, is_open):
    if submit_n_clicks or close_n_clicks:
        return not is_open
    return is_open



@app.callback(
    Output('popularity-score', 'children'),
    [Input('submit-button', 'n_clicks')] +
    [Input(f'input-{feature}', 'value') for feature in musical_features],
    prevent_initial_call=True
)

def update_popularity_score(n_clicks, *feature_values):
    # Callback fonksiyonunuzun geri kalan kısmını buraya ekleyin
    # feature_values şimdi her bir müzik özelliği için gelen değerleri içerir
    # Örneğin: feature_values[0] birinci müzik özelliğinin değerini içerir
    #           feature_values[1] ikinci müzik özelliğinin değerini içerir
    #           ...
    popularity_calculator(input_df)

    # Model ile tahmin yap
    prediction = model.predict(input_df)

    # Tahmini geri döndür
    return f"Şarkının popülerlik skoru tahmini: {prediction[0]}"

if __name__ == '__main__':
    app.run_server(debug=False)


