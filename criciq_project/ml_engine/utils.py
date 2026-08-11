# ml_engine/utils.py
import pandas as pd
import joblib
import os
from django.apps import apps
from django.db import models
from ml_engine.models import Match

# ----- Model और Encoders को Load करने वाला Function -----
def load_ml_artifacts():
    """Model और Encoders को data/ फोल्डर से Load करता है"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(base_dir, 'data', 'model.pkl'))
    team_encoder = joblib.load(os.path.join(base_dir, 'data', 'team_encoder.pkl'))
    venue_encoder = joblib.load(os.path.join(base_dir, 'data', 'venue_encoder.pkl'))
    return model, team_encoder, venue_encoder

# ----- Recent Form (पिछले 5 मैचों का प्रदर्शन) -----
def get_recent_form(team_name):
    """किसी टीम के पिछले 5 मैचों में जीत का % निकाले"""
    # उस टीम के सारे मैच डेटाबेस से निकालें (Date के हिसाब से सॉर्ट)
    all_matches = Match.objects.filter(
        models.Q(team1=team_name) | models.Q(team2=team_name)
    ).order_by('-season', '-match_id')[:5]  # पिछले 5 मैच
    
    if len(all_matches) == 0:
        return 0.5  # अगर कोई मैच नहीं मिला तो 50%
    
    wins = 0
    for match in all_matches:
        if match.winner == team_name:
            wins += 1
    
    return wins / len(all_matches)

# ----- Head-to-Head Record -----
def get_head_to_head(team1, team2):
    """दो टीमों के बीच पिछले मैचों में Team1 की जीत % निकाले"""
    matches = Match.objects.filter(
        (models.Q(team1=team1) & models.Q(team2=team2)) |
        (models.Q(team1=team2) & models.Q(team2=team1))
    )
    
    if len(matches) == 0:
        return 0.5  # पहली बार भिड़ रही हैं
    
    wins = 0
    for match in matches:
        if match.winner == team1:
            wins += 1
    
    return wins / len(matches)

# ----- Main Prediction Function -----
def predict_match(team1, team2, venue, toss_winner, toss_decision):
    """सारे Features बनाकर Model से Prediction लगाता है"""
    
    # 1. Recent Form निकालें
    team1_form = get_recent_form(team1)
    team2_form = get_recent_form(team2)
    
    # 2. Head-to-Head निकालें
    head_to_head = get_head_to_head(team1, team2)
    
    # 3. Toss Decision Encode करें (Bat=0, Field=1)
    toss_decision_encoded = 0 if toss_decision.lower() == 'bat' else 1
    
    # 4. Toss Winner Encode करें (Team1 ने जीता = 1, Team2 ने = 0)
    toss_winner_encoded = 1 if toss_winner == team1 else 0
    
    # 5. Model और Encoders Load करें
    model, team_encoder, venue_encoder = load_ml_artifacts()
    
    # 6. Team1, Team2, Venue को Encode करें
    try:
        team1_encoded = team_encoder.transform([team1])[0]
        team2_encoded = team_encoder.transform([team2])[0]
        venue_encoded = venue_encoder.transform([venue])[0]
    except ValueError:
        # अगर कोई नई Team/Venue है (जो Model ने पहले नहीं देखी), तो Default Value डालें
        # Model Training में जितनी Teams थीं, उनका Average लें (सुरक्षित विकल्प)
        team1_encoded = 0
        team2_encoded = 1
        venue_encoded = 0
    
    # 7. Feature Vector बनाएं (Order Model Training जैसा ही होना चाहिए)
    feature_vector = [[
        venue_encoded,
        toss_decision_encoded,
        toss_winner_encoded,
        head_to_head,
        team1_form,
        team2_form,
        team1_encoded,
        team2_encoded
    ]]
    
    # 8. Prediction लगाएं
    prediction = model.predict(feature_vector)[0]  # 1 = Team1 wins, 0 = Team2 wins
    probability = model.predict_proba(feature_vector)[0]  # [Team2_prob, Team1_prob]
    
    # 9. Result तैयार करें
    if prediction == 1:
        winner = team1
        team1_prob = probability[1]
        team2_prob = probability[0]
    else:
        winner = team2
        team1_prob = probability[0]
        team2_prob = probability[1]
    
    return {
        'winner': winner,
        'team1_prob': round(team1_prob * 100, 2),
        'team2_prob': round(team2_prob * 100, 2),
        'team1': team1,
        'team2': team2,
        'venue': venue,
        'toss_winner': toss_winner 
    }