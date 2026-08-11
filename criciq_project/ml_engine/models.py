from django.db import models
class Match(models.Model):
    match_id = models.IntegerField(unique=True)
    season = models.IntegerField()
    city = models.CharField(max_length=50)
    team1 = models.CharField(max_length=50)
    team2 = models.CharField(max_length=50)
    winner = models.CharField(max_length=50, null=True, blank=True)
    venue = models.CharField(max_length=100)
    toss_winner = models.CharField(max_length=50)
    toss_decision = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.team1} vs {self.team2} ({self.season})"