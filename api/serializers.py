from rest_framework import serializers
from .models import UploadedImage

class ImageUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedImage
        fields = ['image']
