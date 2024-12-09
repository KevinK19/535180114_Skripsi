from flask import render_template, redirect, url_for, flash
from app.models.User import *
import bcrypt

def index():
	data = User.get().serialize()
	return render_template('pages/user/index.html', users=data, segment='user')

def create():
	return render_template('pages/user/create.html', segment='user')

def store(data):
	if data['password'] == data['password1']:
		password = bcrypt.hashpw(data['password'].encode('utf8'), bcrypt.gensalt()) ## encode password menggunakan bcrypt
		checkUsername = User.get_by_username(data['username'])
		if checkUsername == None:
			user = {
				"nama"     : data['nama'],
				"username" : data['username'],
				"password" : password,
			}
			User.insert(user)
			flash('Data Berhasil Di tambahkan.', 'success')
			return redirect(url_for("user_index"))
		else:
			flash('Username sudah terdaftar.', 'danger')
			return redirect(url_for('register'))
	else:
		flash('Password Tidak Sesuai.', 'danger')
		return redirect(url_for("user_index"))

def edit(id):
	data = User.find_or_fail(id).serialize()
	user = User.find(id)
	return render_template('/pages/user/edit.html', data=data, user=user, segment='user')

def update(request, id):
	post = request.form
	user = User.find(id)
	user.username = post['username']
	user.nama     = post['nama']
	if post['password'] != "":
		password = bcrypt.hashpw(post['password'].encode('utf8'), bcrypt.gensalt())
		user.password = password
	user.save()
	flash('Data berhasil diupdate.', 'success')
	return redirect(url_for('user_index'))

def delete(id):
	delete = User.find(id).delete()
	flash('Data berhasil dihapus.', 'success')
	return redirect(url_for("user_index"))