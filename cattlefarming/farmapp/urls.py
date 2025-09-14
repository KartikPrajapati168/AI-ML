from django.urls import path
from . import views

app_name = 'farmapp'
urlpatterns = [
    path('', views.login_page, name='login_page'),path('login/', views.login_process, name='login_process'),
    path('login/', views.login_process, name='login_process'),
    path('register/', views.register_view, name='register_page'),
    path('logout/', views.logout_view, name='logout'),
    path('predict/', views.predict_view, name='predict'),
    path('submit-actuals/', views.submit_actuals, name='submit_actuals'),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("dashboard-data/", views.dashboard_data, name="dashboard_data"),
]
