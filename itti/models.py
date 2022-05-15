from django.db import models

# Create your models here.
class Saliency(models.Model):
    idClothe = models.CharField(max_length=24, blank=False, default='')
    vectorImage = models.CharField(max_length=350, blank=False, default='')
    urlImage = models.CharField(max_length=200, blank=False, default='')