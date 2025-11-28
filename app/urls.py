from django.urls import path
from . import views
from django.conf import settings 
from django.conf.urls.static import static
from django.contrib import admin

urlpatterns = [
    path('', views.login_view, name='login_view'),
    path('Register/', views.register_view, name='register_view'),
    path('Home/', views.main_view, name='main_home'),
    path('Capturing/', views.getstart, name='getstart'),
    path('capture/', views.capture_upload, name='capture_upload'),  # UPDATED
    path('Songs/', views.songs, name='songs'),
    path('Videos/', views.search_videos, name='videos'),
    path('Error/', views.error, name='error_page'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
