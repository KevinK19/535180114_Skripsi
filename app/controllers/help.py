from flask import render_template

def index():
    return render_template('pages/help.html', segment='help')