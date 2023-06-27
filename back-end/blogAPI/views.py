from django.shortcuts import render
from rest_framework import viewsets
from .models import Project, BlogPost
from .serializers import ProjectSerializer, BlogPostSerializer

class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer

class BlogPostViewSet(viewsets.ModelViewSet):
    queryset = BlogPost.objects.all()
    serializer_class = BlogPostSerializer