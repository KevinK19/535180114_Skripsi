from flask import Flask, request
from app.config.middleware import checkLogin
from app.controllers import home, dataset, algoritma, misc, user, help, evaluasi
import os

app = Flask(__name__)

# login
@app.route("/login")
def login_index():
    return misc.index()

@app.route("/doLogin", methods=['POST'])
def doLogin():
    return misc.doLogin(request.form)

@app.route("/logout")
def logout():
    return misc.logout()

@app.route("/home")
def home_index():
    return home.index()
    
@app.route("/")
@checkLogin
def index():
    return home.index()

@app.route("/help")
def help_index():
    return help.index()
# ---------------------- START USER -----------------------------
@app.route("/users")
@checkLogin
def user_index():
    return user.index() 

@app.route("/users/create")
@checkLogin
def user_create():
    return user.create() 

@app.route("/users/<int:id>/edit")
@checkLogin
def user_edit(id):
    return user.edit(id)

@app.route("/users/store", methods=['POST'])
@checkLogin
def user_store():
    return user.store(request.form)

@app.route("/users/<int:id>/update", methods=['POST'])
@checkLogin
def users_update(id):
    return user.update(request, id)

@app.route("/users/<int:id>/delete")
@checkLogin
def user_delete(id):
    return user.delete(id)
# ---------------------- END USER -----------------------------

# ---------------------- START DATASET -----------------------------
@app.route("/dataset")
@checkLogin
def dataset_index():
    return dataset.index()

@app.route("/kategori/create")
@checkLogin
def kategori_create():
    return dataset.create() 

@app.route("/kategori/store", methods=['POST'])
@checkLogin
def kategori_store():
    return dataset.store_kategori(request.form)

@app.route("/dataset/store", methods=['POST'])
@checkLogin
def dataset_store():
    return dataset.store(request)

@app.route("/dataset/reset")
@checkLogin
def dataset_reset():
    return dataset.dataset_reset()

@app.route("/dataset/detail/<int:id>")
def dataset_detail(id):
    return dataset.detail_data(id)

@app.route("/dataset/<int:id>/delete")
@checkLogin
def dataset_delete(id):
    return dataset.delete(id)
# ---------------------- END DATASET -------------------------------

# ---------------------- END ALGORITMA -------------------------------
@app.route("/prediksi")
def prediksi_index():
    return algoritma.index()
# ---------------------- END ALGORITMA -------------------------------

# ---------------------- END EVALUASI -------------------------------
@app.route("/evaluasi")
def evaluasi_index():
    return evaluasi.index()
# ---------------------- END EVALUASI -------------------------------

app.secret_key = '3RDLwwtFttGSxkaDHyFTmvGytBJ2MxWT8ynWm2y79G8jm9ugYxFFDPdHcBBnHp6E'
app.config['SESSION_TYPE'] = 'filesystem'

@app.context_processor
def inject_stage_and_region():
    return dict(APP_NAME=os.environ.get("APP_NAME"),
        APP_AUTHOR=os.environ.get("APP_AUTHOR"),
        APP_TITLE=os.environ.get("APP_TITLE"),
        APP_LOGO=os.environ.get("APP_LOGO"))

if __name__ == "__main__":
    app.run()
    # app.run(host='0.0.0.0', port=5591)