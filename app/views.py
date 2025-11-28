import os
import shutil
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
        # compile=False saves memory
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

emoji_dist = {
    0:"app/emojis/angry.png",
    1:"app/emojis/disgusted.png",
    2:"app/emojis/fearful.png",
    3:"app/emojis/happy.png",
    4:"app/emojis/neutral.png",
    5:"app/emojis/sad.png",
    6:"app/emojis/surpriced.png"
}

video_dist = {
    0:"angry.mp4",
    1:"disgusted.mp4",
    2:"fearful.mp4",
    3:"happy.mp4",
    4:"neutral.mp4",
    5:"sad.mp4",
    6:"surprised.mp4"
}

# Global variables
result = None
nameofuser = "Guest"

# -------------------------------
# Pages
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
# Capture uploaded image from browser
def capture_upload(request):
    global result
    if request.method == 'POST' and request.POST.get('image'):
        try:
            # Convert base64 image to OpenCV image
            data_url = request.POST['image']
            format, imgstr = data_url.split(';base64,')
            img_data = base64.b64decode(imgstr)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            # Preprocess for model
            img_resized = cv2.resize(img, (48,48))
            img_array = img_resized.reshape(1,48,48,1)

            # Predict emotion
            model = get_emotion_model()
            predict_x = model.predict(img_array)
            result = np.argmax(predict_x, axis=1)
            label = emotion_dict[result[0]]

            emoji_path = emoji_dist[result[0]]
            shutil.copy(emoji_path, 'static/emoji.png')

            return render(request, 'capture.html', {
                'detected': True,
                'detected_emotion': label,
                'emoji_url': '/static/emoji.png'
            })
        except Exception as e:
            print("Error in capture_upload:", e)
            return redirect('error_page')

    return render(request, 'capture.html', {'detected': False})

# -------------------------------
# Music filtering
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
# YouTube search
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
    try:
        # Copy emotion video to static
        src = os.path.join('app/videos', video_dist[result[0]])
        dst = 'static/video.mp4'
        shutil.copyfile(src, dst)

        videos = youtube_search(emotion_dict[result[0]])
        context = {
            "videos": videos,
            "mood": emotion_dict[result[0]],
            "name": nameofuser,
        }
        return render(request, 'videos.html', context)
    except Exception as e:
        print("Error in search_videos:", e)
        return redirect('error_page')

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
        if user is not None:
            login(request, user)
            return redirect('main_home')
        else:
            return render(request, 'login.html', {'error_message':'Invalid username or password.'})
    return render(request,'login.html')

# -------------------------------
# Error handling middleware
class HandleErrorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
    def __call__(self, request):
        return self.get_response(request)
    def process_exception(self, request, exception):
        print("Exception caught:", exception)
        return redirect('error_page')
