from django.db import models

# Create your models here.
class KeyNest(models.Model):
    id=models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    role = models.CharField(choices=[('Admin', 'Admin'), ("User", 'User')], default="User", max_length=10)
    email = models.EmailField()
    password = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    def __str__(self):
        return f"{self.name}: {self.role}"