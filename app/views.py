import os
import base64
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.conf import settings

# -------------------------------
# Load resources once
faceCascade = cv2.CascadeClassifier("app/haarcascade_frontalface_default.xml")
mood_music = pd.read_csv("app/musicData.csv")

# Lazy-load model
MODEL_PATH = "app/face_emotion.h5"
GDRIVE_FILE_ID = "1BHYWzYlxnLEMviQ6jV-6bJYsMNvYWHST"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

emotion_model = None

def get_emotion_model():
    global emotion_model
    if emotion_model is None:
        if not os.path.exists(MODEL_PATH):
            import gdown
            print("Downloading face_emotion.h5 from Google Drive...")
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
            print("Download complete.")
        emotion_model = load_model(MODEL_PATH, compile=False)
    return emotion_model

# -------------------------------
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# Static file paths (no copy, direct load)
emoji_dist = {
    0:"emojis/angry.png",
    1:"emojis/disgusted.png",
    2:"emojis/fearful.png",
    3:"emojis/happy.png",
    4:"emojis/neutral.png",
    5:"emojis/sad.png",
    6:"emojis/surprised.png"
}

video_dist = {
    0:"videos/angry.mp4",
    1:"videos/disgusted.mp4",
    2:"videos/fearful.mp4",
    3:"videos/happy.mp4",
    4:"videos/neutral.mp4",
    5:"videos/sad.mp4",
    6:"videos/surprised.mp4"
}

# Globals
result = None
nameofuser = "Guest"

# -------------------------------
def main_view(request):
    return render(request, "index.html")

def getstart(request):
    global result
    result = None
    return render(request, "capture.html", {'detected': False})

def error(request):
    global nameofuser
    return render(request, "error.html", {"name": nameofuser})

# -------------------------------
# Capture image and detect emotion
def capture_upload(request):
    global result
    if request.method == 'POST' and request.POST.get('image'):
        try:
            data_url = request.POST['image']
            format, imgstr = data_url.split(';base64,')
            img_data = base64.b64decode(imgstr)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            img_resized = cv2.resize(img, (48,48))
            img_array = img_resized.reshape(1,48,48,1)

            model = get_emotion_model()
            predict_x = model.predict(img_array)
            result = np.argmax(predict_x, axis=1)
            label = emotion_dict[result[0]]

            return render(request, 'capture.html', {
                'detected': True,
                'detected_emotion': label,
                'emoji_url': f"/static/{emoji_dist[result[0]]}"
            })
        except Exception as e:
            print("Error in capture_upload:", e)
            return redirect('error_page')

    return render(request, 'capture.html', {'detected': False})

# -------------------------------
# Music list
def music_results(n):
    if n in [0,1,2]:
        f = mood_music[mood_music['mood']=='Chill'].dropna().sample(n=10)
    elif n in [3,4]:
        f = mood_music[mood_music['mood']=='energetic'].dropna().sample(n=10)
    elif n==5:
        f = mood_music[mood_music['mood']=='cheerful'].dropna().sample(n=10)
    elif n==6:
        f = mood_music[mood_music['mood']=='romantic'].dropna().sample(n=10)
    f.reset_index(inplace=True)
    return f

def songs(request):
    global result, nameofuser
    if result is None:
        return redirect('getstart')

    data = music_results(result[0])
    context = {
        "songs": data['id'],
        "name": nameofuser,
        "mood": emotion_dict[result[0]],
    }
    return render(request, "songs.html", context)

# -------------------------------
# YouTube + static video
def youtube_search(label):
    from googleapiclient.discovery import build
    youtube = build('youtube', 'v3', developerKey=settings.YOUTUBE_API_KEY)
    request_yt = youtube.search().list(
        q=f'Telugu video songs {label}',
        part='snippet',
        type='video',
        maxResults=10,
        regionCode='IN',
        relevanceLanguage='te'
    )
    response = request_yt.execute()
    return response.get('items', [])

def search_videos(request):
    global result, nameofuser
    if result is None:
        return redirect('getstart')

    videos = youtube_search(emotion_dict[result[0]])

    context = {
        "videos": videos,
        "mood": emotion_dict[result[0]],
        "name": nameofuser,
        "media": {
            "img": emoji_dist[result[0]],
            "video": video_dist[result[0]]
        }
    }
    return render(request, 'videos.html', context)

# -------------------------------
# User auth
def register_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        retype = request.POST['retype']
        if password != retype:
            return render(request,'register.html',{'error_message':'Password does not match.'})
        if User.objects.filter(username=username).exists():
            return render(request,'register.html',{'error_message':'Username already exists.'})
        user = User.objects.create_user(username=username,password=password)
        login(request,user)
        return redirect('login_view')
    return render(request,'register.html')

def login_view(request):
    global nameofuser
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        nameofuser = username
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('main_home')
        return render(request, 'login.html', {'error_message':'Invalid username or password.'})
    return render(request,'login.html')

# -------------------------------
class HandleErrorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
    def __call__(self, request):
        return self.get_response(request)
    def process_exception(self, request, exception):
        print("Exception caught:", exception)
        return redirect('error_page')
