from django.urls import path
from . import views

app_name = "main"

urlpatterns = [
    path('', views.index, name='index'),
    path('predictImage', views.predictImage, name='predictImage'),
    path('viewDataBase', views.viewDataBase, name='viewDataBase'),
    path('predict/', views.predict_chances, name='submit_prediction'),
    path('results/', views.view_results, name='results'),
]
