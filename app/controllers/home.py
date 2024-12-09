from flask import render_template
from app.models.Dataset import *
from app.models.Detail import *
import pandas as pd

def index():
    # PT Bank Rakyat Indonesia (Persero)
    df = pd.read_csv('static/Dataset/BKRKF.csv', delimiter=';')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Mengonversi ke datetime
    df = df.dropna()

    labels = [label.strftime('%d/%m/%Y') for label in df['Date']]
    value  = df['Close'].values.tolist()

    # PT Bank Mandiri (Persero)
    df2 = pd.read_csv('static/Dataset/PPERY.csv', delimiter=';')
    df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')  # Mengonversi ke datetime
    df2 = df2.dropna()
    
    labels2 = [label.strftime('%Y-%m-%d') for label in df2['Date']]
    value2  = df2['Close'].values.tolist()

    # PT Bank Central Asia Tbk
    df3 = pd.read_csv('static/Dataset/PBCRF.csv', delimiter=';')
    df3['Date'] = pd.to_datetime(df3['Date'], errors='coerce')  # Mengonversi ke datetime
    df3 = df3.dropna()
    
    labels3 = [label.strftime('%Y-%m-%d') for label in df3['Date']]
    value3  = df3['Close'].values.tolist()

    # PT Aneka Tambang Tbk
    df4 = pd.read_csv('static/Dataset/PAEKY.csv', delimiter=';')
    df4['Date'] = pd.to_datetime(df4['Date'], errors='coerce')  # Mengonversi ke datetime
    df4 = df4.dropna()
    
    labels4 = [label.strftime('%Y-%m-%d') for label in df4['Date']]
    value4  = df4['Close'].values.tolist()

    # PT Adaro Energy Indonesia Tbk
    df5 = pd.read_csv('static/Dataset/ADOOY.csv', delimiter=';')
    df5['Date'] = pd.to_datetime(df5['Date'], errors='coerce')  # Mengonversi ke datetime
    df5 = df5.dropna()
    
    labels5 = [label.strftime('%Y-%m-%d') for label in df5['Date']]
    value5  = df5['Close'].values.tolist()

    # PT TIMAH Tbk
    df6 = pd.read_csv('static/Dataset/PTTMF.csv', delimiter=';')
    df6['Date'] = pd.to_datetime(df6['Date'], errors='coerce')  # Mengonversi ke datetime
    df6 = df6.dropna()
    
    labels6 = [label.strftime('%Y-%m-%d') for label in df6['Date']]
    value6  = df6['Close'].values.tolist()

    return render_template('pages/home.html', segment='home', labels=labels, value=value, labels2=labels2, value2=value2,
    labels3=labels3, value3=value3, labels4=labels4, value4=value4, labels5=labels5, value5=value5, labels6=labels6, value6=value6)