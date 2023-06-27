from django.contrib import admin
from .models import Project, BlogPost
# Register your models here.
admin.site.register([Project, BlogPost])