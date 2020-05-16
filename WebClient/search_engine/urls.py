from django.urls import path
from . import views

app_name='search_engine'
urlpatterns = [
    path('', views.home, name='home' ),
    path('search',views.Search.as_view(), name='search')
]
