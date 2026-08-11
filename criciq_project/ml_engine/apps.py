# ml_engine/apps.py
from django.apps import AppConfig
import joblib
import os

class MlEngineConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_engine'

    def ready(self):
        # Model और Encoders को Server Startup पर Load करें
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'data', 'model.pkl')
        team_enc_path = os.path.join(base_dir, 'data', 'team_encoder.pkl')
        venue_enc_path = os.path.join(base_dir, 'data', 'venue_encoder.pkl')
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.team_encoder = joblib.load(team_enc_path)
            self.venue_encoder = joblib.load(venue_enc_path)
            print("✅ CricIQ Model and Encoders Loaded Successfully!")
        else:
            self.model = None
            print("⚠️ Model file not found! Please run the training notebook first.")