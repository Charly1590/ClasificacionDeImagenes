from rest_framework import serializers
from apiCNN import models


class ImagenSerializer(serializers.ModelSerializer):
    class Meta:
        fields = (
            'id',
            'prediccion',
            'porcentage',
            'nombre',
        )
        model = models.ImagenPredict

