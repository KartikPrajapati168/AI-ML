import pandas as pd
from django.core.management.base import BaseCommand
from ml_engine.models import Match

class Command(BaseCommand):
    help = 'Imports matches from CSV into the database'

    def handle(self, *args, **kwargs):
        df = pd.read_csv('data/matches.csv')
        count = 0
        for _, row in df.iterrows():
            winner = row['winner'] if pd.notna(row['winner']) else None
            
            obj, created = Match.objects.update_or_create(
                match_id=row['id'],  # CSV में 'id' column है
                defaults={
                    'season': row['season'],
                    'city': row['city'],
                    'team1': row['team1'],
                    'team2': row['team2'],
                    'winner': winner,
                    'venue': row['venue'],
                    'toss_winner': row['toss_winner'],
                    'toss_decision': row['toss_decision'],
                }
            )
            if created:
                count += 1
                
        self.stdout.write(self.style.SUCCESS(f'✅ {count} new matches imported successfully!'))