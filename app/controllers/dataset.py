from flask import render_template, flash, redirect, url_for, request
from werkzeug.utils import secure_filename
from app.models.Dataset import *
from app.models.Detail import *
from app.models.Kategori import *
import os
import pandas as pd
import numpy as np

# konfigurasi file
conf = {
    'UPLOAD_EXTENSIONS': ['.csv', '.xlsx']
}

def index():
    list_kategori = Kategori.get().serialize()
    show_data = 0
    data = ''
    kategori = ''

    if len(request.args) > 0:
        show_data = 1
        id_kategori = request.args['kategori']
        kategori = Kategori.where('id', int(id_kategori)).first().serialize()
        data = Dataset.where('kategori_id', int(id_kategori)).get().serialize()
    return render_template('pages/dataset/index.html', data=data, segment='dataset', list_kategori=list_kategori, 
    show_data=show_data, kategori=kategori)

def create():
	return render_template('pages/dataset/create.html', segment='dataset')

def store_kategori(data):
    cekKategori = Kategori.get_by_kategori(data['kategori'])
    if cekKategori == None:
        saveData = Kategori()
        saveData.kategori = data['kategori']
        saveData.save()
        flash('Kategori Berhasil Ditambahkan.', 'success')
        return redirect(url_for("dataset_index"))
    else:
        flash('Kategori Sudah Terdaftar.', 'danger')
        return redirect(url_for('dataset_index'))

def store(request):
    post = request.form # Berisi data dari form HTML
    # Menyimpan nama_data kedalam database
    dataset = Dataset()
    dataset.kategori_id = int(post['kategori'])
    dataset.nama_data = post['nama_data']
    dataset.save()
    
    uploaded_file = request.files['file']
    filename      = secure_filename(uploaded_file.filename)

    file_ext = os.path.splitext(filename)[1]
    if file_ext not in conf['UPLOAD_EXTENSIONS']:
        flash('Tipe file tidak sesuai!', 'danger')
        return redirect(url_for('dataset_index'))

    # Upload file to static with new name
    uploaded_file.save("static/import_data" + file_ext)

    if file_ext == '.csv':
        # Read uploaded file
        # df = pd.read_csv("static/import_data.csv", delimiter=';')
        df = pd.read_csv("static/import_data.csv", delimiter=';')
        df = df.replace(np.nan, 'EMPTY')
    elif file_ext == '.xlsx':
        # Read uploaded file
        df = pd.read_excel("static/import_data.xlsx")
        df = df.replace(np.nan, 'EMPTY')

    # Mengonversi 'Date' ke datetime
    print(df)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Mengambil hanya bagian tanggal
    df['Date'] = df['Date'].dt.date

    for index, row in df.iterrows():
        print(row['Date'])
        print('Execute index ', index, end='... ')
        tmp_store = {
            'dataset_id' : dataset.serialize()['id'],
            'date'       : row['Date'],
            'open'       : row['Open'],
            'high'       : row['High'],
            'low'        : row['Low'],
            'close'      : row['Close'],
            'adj_close'  : row['Adj Close'],
            'volume'     : row['Volume'],
        }
        Detail.insert(tmp_store)
        print('Done.')
    flash('Data berhasil disimpan.', 'success')
    return redirect(url_for('dataset_index'))

def detail_data(id):
    nama_data = Dataset.where('id', id).select('nama_data').first().serialize()
    data      = Detail.where('dataset_id', id).get().serialize()
    return render_template('pages/dataset/detail.html', data=data, nama_data=nama_data, segment='dataset')

def delete(id):
    try:
        delete = Dataset.find(id).delete()
        del_detail = Detail.where('dataset_id', id).delete()
        flash('Data berhasil di hapus.', 'success')
        return redirect(url_for("dataset_index"))
    except Exception as e:
        return 'Something went wrong ' + str(e)

def dataset_reset():
    Kategori.truncate()
    Detail.truncate()
    Dataset.truncate()
    flash('Data berhasil direset.', 'success')
    return redirect(url_for('dataset_index'))