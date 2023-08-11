# Generated by Django 3.2.7 on 2023-05-26 16:36

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0007_auto_20230514_0238'),
    ]

    operations = [
        migrations.AlterField(
            model_name='data',
            name='age',
            field=models.PositiveIntegerField(null=True, validators=[django.core.validators.MinValueValidator(18), django.core.validators.MaxValueValidator(40)]),
        ),
        migrations.AlterField(
            model_name='data',
            name='gpa',
            field=models.FloatField(null=True, validators=[django.core.validators.MinValueValidator(0.0), django.core.validators.MaxValueValidator(4.0)]),
        ),
    ]
