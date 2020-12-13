from django.db.models import Model
from django.db.models import CharField
from django.db.models import FloatField


class ImagenPredict(Model):
    nombre = CharField(max_length=100, blank=False)
    prediccion = CharField(max_length=30, blank=False)
    porcentage = FloatField()
    
