# Generated by Django 2.1.3 on 2019-04-25 21:09

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_book_cover'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='book',
            name='author',
        ),
        migrations.RemoveField(
            model_name='book',
            name='cover',
        ),
        migrations.RemoveField(
            model_name='book',
            name='title',
        ),
    ]