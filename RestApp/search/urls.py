from django.contrib import admin
from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from . import views
from rest_framework_swagger.views import get_swagger_view

schema_view = get_swagger_view(title='Summary Rest API')

urlpatterns = [
    path('tweets/summary', views.TweetSummary.as_view(),name='tweet_summ'),
    path('tweets/process/', views.TweetProcessed.as_view(),name='tweet_process'),
    path('doc/',views.schema_view),
    #path('docs/',schema_view),
]

urlpatterns = format_suffix_patterns(urlpatterns)