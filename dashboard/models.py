from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from sklearn.tree import DecisionTreeClassifier
import joblib

# Create your models here.
ANSWER = (
    (0, 'Yes'),
    (1, 'No'),
)


class Data(models.Model):
    name = models.CharField(max_length=100, null=True)
    age = models.PositiveIntegerField(
        validators=[MinValueValidator(18), MaxValueValidator(40)], null=True)
    gpa = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(4.0)],null=True)
    english = models.PositiveIntegerField(choices=ANSWER, null=True)
    predictions = models.CharField(max_length=100, blank=True)
    date = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        ml_model = joblib.load('ml_model/ml_model.joblib')
        self.predictions = ml_model.predict(
            [[self.age, self.gpa, self.english]])
        return super().save(*args, *kwargs)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return self.name
