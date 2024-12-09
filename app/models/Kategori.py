from app.config.db import db
from orator import Model, SoftDeletes
import pendulum

Model.set_connection_resolver(db)

class Kategori(SoftDeletes,Model):
    __table__   = 'kategori'
    __guarded__ = ['id']
    __dates__   = ['deleted_at']

    def fresh_timestamp(self):
        return pendulum.now("Asia/Jakarta")

    def get_by_kategori(kategori):
        data = Kategori.where('kategori', kategori).first()
        if data is not None:
            data = data.serialize()
        return data
