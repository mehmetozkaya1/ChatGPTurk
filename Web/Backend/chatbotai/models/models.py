from django.db import models

class Photo(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='photos/')

    def __str__(self):
        return self.name
