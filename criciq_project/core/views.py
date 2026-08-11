# core/views.py
from django.shortcuts import render
from django.db import models
from ml_engine.models import Match
from ml_engine.utils import predict_match

def index(request):
    """होम पेज - Prediction Form दिखाएगा"""
    # Database से सारी Unique Teams और Venues निकालें
    teams1 = list(Match.objects.values_list('team1', flat=True).distinct())
    teams2 = list(Match.objects.values_list('team2', flat=True).distinct())
    all_teams = sorted(set(teams1 + teams2))
    all_venues = sorted(Match.objects.values_list('venue', flat=True).distinct())
    
    context = {
        'teams': all_teams,
        'venues': all_venues,
    }
    return render(request, 'core/index.html', context)

def predict(request):
    """Prediction करके Result दिखाएगा"""
    if request.method == 'POST':
        team1 = request.POST.get('team1')
        team2 = request.POST.get('team2')
        venue = request.POST.get('venue')
        toss_winner = request.POST.get('toss_winner')
        toss_decision = request.POST.get('toss_decision')
        
        # Validation (चेक करें कि Team1 और Team2 अलग-अलग हों)
        if team1 == team2:
            context = {'error': 'Team1 और Team2 एक जैसी नहीं हो सकतीं!'}
            return render(request, 'core/index.html', context)
        
        # Prediction लगाएं
        result = predict_match(team1, team2, venue, toss_winner, toss_decision)
        
        return render(request, 'core/result.html', result)
    
    # अगर POST नहीं है तो Home Page पर भेजें
    return index(request)