from django.db import models
from django.conf import settings
from django.core.files.storage import FileSystemStorage

class Book(models.Model):
    pdf = models.FileField(upload_to='books/pdfs/')

    def __str__(self):
        return self.title

    def delete(self, *args, **kwargs):
        self.pdf.delete()
        super().delete(*args, **kwargs)
