# Generated by Django 2.2.10 on 2022-05-15 02:43

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Saliency',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('idClothe', models.CharField(default='', max_length=24)),
                ('vectorImage', models.CharField(default='', max_length=350)),
                ('urlImage', models.CharField(default='', max_length=200)),
            ],
        ),
    ]
