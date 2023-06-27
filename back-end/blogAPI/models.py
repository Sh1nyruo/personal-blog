from django.db import models

# 项目
class Project(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    imgSrc = models.ImageField(upload_to='projects/', null=True, blank=True)
    href = models.URLField()
    deployed = models.URLField()
    tools = models.JSONField()

    def __str__(self):
        return self.title
    
class BlogPost(models.Model):
    title = models.CharField(max_length=200)
    date = models.DateTimeField()
    tags = models.JSONField(max_length=200)  # could also use a ManyToManyField for more complex tag handling
    draft = models.BooleanField(default=False)
    summary = models.TextField()
    content = models.TextField()
    slug = models.TextField()
    
    def __str__(self):
        return self.title