from rest_framework import serializers
from .models import Project, BlogPost
from django.core.files.storage import DefaultStorage

class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = '__all__'

class BlogPostSerializer(serializers.ModelSerializer):
    class Meta:
        model = BlogPost
        fields = '__all__'