# Generated by Django 5.0.6 on 2024-09-13 13:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pm', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='password',
            field=models.CharField(default='pbkdf2_sha256$720000$KrDDrY2dnDBJsJMGNMIE8q$jBFe06RYcZ40hd4cL6BAQRSrlRB1af3umIaQW8m/cqk=', max_length=128),
        ),
    ]