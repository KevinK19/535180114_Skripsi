from flask import render_template, request
from app.models.Dataset import *
from app.models.Evaluasi import *
from app.models.Kategori import *
import pandas as pd

def index():
    list_kategori = Kategori.get().serialize()
    show_data = 0
    kategori = ''
    df = pd.DataFrame([])

    if len(request.args) > 0:
        show_data = 1
        id_kategori = request.args['kategori']
        kategori = Kategori.where('id', int(id_kategori)).first().serialize()

        data = Evaluasi.join('dataset', 'dataset.id', '=', 'evaluasi.dataset_id')\
        .select('dataset.nama_data', 'dataset.kategori_id', 'evaluasi.*')\
        .where('dataset.kategori_id', int(id_kategori))\
        .get().serialize()
        df = pd.DataFrame(data)
        if len(df) > 0:
            # Mengurutkan berdasarkan kolom nama_data dan test_split
            df = df.sort_values(by=['nama_data', 'test_split'], ascending=[True, False])
            # print(df)
            df.to_excel('static/Hasil Evaluasi '+kategori['kategori']+'.xlsx')
    return render_template('pages/evaluasi.html', segment='evaluasi', df=df, kategori=kategori, 
    list_kategori=list_kategori, show_data=show_data)