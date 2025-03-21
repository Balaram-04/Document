from django.shortcuts import render, redirect
from . models import *
# Create your views here.
from django.contrib import messages
from tensorflow.keras.models import load_model
import imutils, pickle
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from django.db.models import Q
import cv2
import os
import uuid
from PIL import Image
from django.core.paginator import Paginator

# Create your views here.
def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')


def register(request):
    # UserModel.objects.all().delete()
    # UploadFileModel.objects.all().delete()
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']
        if password == confirm_password:
            if UserModel.objects.filter(Q(email=email) | Q(username=username)).exists():
                messages.error(request, 'Email or User Name already exists')
                return redirect('register')
            
            # Capture live images from webcam
            cam = cv2.VideoCapture(0)
            harcascadePath = "Haarcascade/haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(harcascadePath)
            sampleNum = 0
            # folder_path = os.path.join('IndentorImages', email)
            # os.makedirs(folder_path, exist_ok=True)

            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sampleNum += 1
                    # Save captured face images
                    cv2.imwrite("UserImages/ " +username + '.' +str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])

                cv2.imshow('frame', img)

                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                elif sampleNum >= 350:  # Stop after capturing 50 images
                    break

            cam.release()
            cv2.destroyAllWindows()
            user = UserModel.objects.create(username=username, email=email, password=password)
            user.save()
            messages.success(request, 'User created successfully')
            return redirect('training')
        else:
            messages.error(request, 'Passwords do not match')
            return redirect('register')    
    return render(request, 'register.html')



def training(request):
    le = LabelEncoder()
    faces, Id = getImagesAndLabels("UserImages")
    Id=le.fit_transform(Id)
    output = open('model/encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(Id))
    recognizer.save(r"model\Trainner.yml")
    messages.success(request, 'Your model has been trained successfully!!')
    return redirect('login')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = str(os.path.split(imagePath)[-1].split(".")[0])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def detect_and_predict_person(img, faceNet, model):
	(h, w) = img.shape[:2]
	blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224),(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = img[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (32, 32))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = model.predict(faces, batch_size=32)
	return (locs, preds)


def login(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        if UserModel.objects.filter(email=email, password=password).exists():

            cam = cv2.VideoCapture(0)
            harcascadePath = "Haarcascade/haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(harcascadePath)
            recognized = False
            prototxtPath = r"face_detector\deploy.prototxt"
            weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
            model = load_model("model/my_model.h5")
            print("Model loaded successfully")
            font= cv2.FONT_HERSHEY_SIMPLEX 
            faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
            pkl_file = open('model/encoder.pkl', 'rb')
            le = pickle.load(pkl_file)
            pkl_file.close()
            det = 0
            while True:
                ret, img = cam.read()
                img = imutils.resize(img, width=1000)
                (locs, preds) = detect_and_predict_person(img, faceNet, model)
                for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (fake, real) = pred
                    if np.argmax(np.array(pred)) == 0:
                        color = (255, 0, 0)
                    else:
                        color = (0, 0, 255)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = detector.detectMultiScale(gray, 1.2, 5)
                        for (x, y, w, h) in faces:
                            # cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
                            face_model = "model/Trainner.yml"
                            recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
                            recognizer.read(face_model)
                            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                            print(conf)
                            if (conf > 20):
                                tt = le.inverse_transform([Id])
                                tt = tt[0]
                                det += 1
                                if det == 3:
                                    # request.session['login'] = 'indentor'
                                    request.session['email'] = email
                                    cam.release()
                                    cv2.destroyAllWindows()
                                    return redirect('home')
                            else:
                                tt = "Unknown"
                            cv2.putText(img, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
                    label = "Fake" if fake > real else "Real"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    label = "{}: {:.2f}%".format(label, max(fake, real) * 100)
                    cv2.putText(img, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
                cv2.imshow("Frame", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        else:
            messages.error(request, 'Invalid email or password!')
            return redirect('login')


    return render(request, 'login.html')

import random
from django.core.mail import send_mail
def forgotpass(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        if UserModel.objects.filter(email=email).exists():
            user = UserModel.objects.get(email=email)
            
            otp = random.randint(10000,99999)
            user.otp = otp
            user.save()
            email_subject = 'Reset Passward Details'
            email_message = f'Hello {email},\n\nWelcome To Our Website!\n\nHere are your OTP details:\nEmail: {email}\OTP: {otp}\n\nPlease keep this information safe.\n\nBest regards,\nYour Website Team'
            from_email = 'cse.takeoff@gmail.com'
            send_mail(email_subject, email_message, from_email, [email])
            messages.success(request, 'OTP sent successfully')
            return redirect('resetpassword')
        else:
            messages.error(request, 'Invalid email!')
            return redirect('forgotpass')


    return render(request, 'forgotpass.html')


def resetpassword(request):
    if request.method == 'POST':
        otp = request.POST.get('otp')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        email = request.POST.get('email')
        if UserModel.objects.filter(otp=otp, email=email).exists():
            user = UserModel.objects.get(email=email)
            if password == confirm_password:
                user.password = password
                user.save()
                messages.success(request, 'Password reset successfully')
                return redirect('login')
            else:
                messages.error(request, 'Password and confirm password does not match!')
                return redirect('resetpassword')
        else:
            messages.error(request, 'Invalid OTP!')
            return redirect('resetpassword')
    return render(request, 'resetpassword.html')



def home(request):
     email =request.session['email']
    #  print(email)
     return render(request, 'home.html',{'email':email})

import hashlib

def hash_string(input_string):
    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()
    # Update the hash object with the bytes of the input string
    sha256_hash.update(input_string.encode('utf-8'))
    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()

import os
import uuid
from datetime import datetime
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import UploadFileModel, UserModel  # Import your models
import hashlib

def hash_string(content):
    """Generate a hash for the file content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def uploadfiles(request):
    if request.method == 'POST':
        file = request.FILES['file']
        original_filename = file.name
        extension = os.path.splitext(original_filename)[1].lower()

        # Generate a unique filename with timestamp and UUID
        unique_filename = f"{os.path.splitext(original_filename)[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{extension}"

        file_path = os.path.join('static/assets', 'Files', unique_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists

        # Save the file
        with open(file_path, 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)

        # Read file content safely
        if extension in ['.txt', '.csv', '.log']:  # Only read text files
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                messages.error(request, "Error decoding text file. Ensure it is UTF-8 encoded.")
                return redirect('uploadfiles')
        else:
            # For binary files like PDFs, DOCs, etc.
            with open(file_path, 'rb') as f:
                text = f.read()  # Read binary data
            text = str(hashlib.sha256(text).hexdigest())  # Generate hash for binary files

        # Hash and check for duplicates
        hashed_string = hash_string(text)
        if UploadFileModel.objects.filter(datahash=hashed_string).exists():
            messages.success(request, 'Data already exists!')
            return redirect('uploadfiles')

        user = UserModel.objects.get(email=request.session['email'])
        UploadFileModel.objects.create(
            filename=unique_filename,
            file=file_path,
            user=user,
            datahash=hashed_string
        ).save()

        messages.success(request, f"File '{original_filename}' uploaded successfully as '{unique_filename}'")
        return redirect('uploadfiles')

    return render(request, 'uploadfiles.html')


def viewfiles(request):
    # UploadFileModel.objects.all().delete()
    email = request.session['email']
    files = UploadFileModel.objects.all()
    paginator = Paginator(files, 4)  
    page_number = request.GET.get('page')
    page_data = paginator.get_page(page_number)
    return render(request, 'viewfiles.html', {'data':page_data, 'email':email})


from django.shortcuts import render
from gtts import gTTS
import os
from datetime import datetime
import uuid
import chardet  # Auto-detect file encoding
import fitz  # PyMuPDF for PDF text extraction
from docx import Document  # Extract text from DOCX files
import pythoncom  # Required for pywin32 on Windows
import win32com.client  # Extract text from DOC files (Microsoft Word)
from .models import UploadFileModel

# Supported file types
TEXT_FILE_EXTENSIONS = {'.txt', '.csv', '.log'}
PDF_FILE_EXTENSIONS = {'.pdf'}
DOCX_FILE_EXTENSIONS = {'.docx'}
DOC_FILE_EXTENSIONS = {'.doc'}

def detect_encoding(file_path, default='utf-8'):
    """Detect file encoding to avoid decoding errors."""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read first 10,000 bytes for detection
    result = chardet.detect(raw_data)
    return result.get('encoding', default)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)  # Open the PDF file
        text = ""
        for page in doc:
            text += page.get_text("text")  # Extract text from each page
        return text.strip()
    except Exception as e:
        return None  # Return None if extraction fails

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file using python-docx."""
    try:
        doc = Document(docx_path)  # Open the DOCX file
        text = "\n".join([para.text for para in doc.paragraphs])  # Extract paragraphs as text
        return text.strip()
    except Exception as e:
        return None  # Return None if extraction fails

def extract_text_from_doc(doc_path):
    """Extract text from a DOC file using pywin32 (Microsoft Word)."""
    try:
        pythoncom.CoInitialize()  # Initialize COM (Required for Windows)
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False  # Run MS Word in the background

        doc = word.Documents.Open(os.path.abspath(doc_path))  # Open the document
        text = doc.Content.Text.strip()  # Extract text
        doc.Close(False)  # Close the document (without saving)
        word.Quit()  # Quit Word application

        return text
    except Exception as e:
        return None  # Return None if extraction fails

def text_to_speech(request, id, req):
    # Retrieve the file path from the model
    if req == 'owner':
        # print('skfsdnfd',req)
        data = UploadFileModel.objects.get(id=id)
        file_path = data.file.path
    else:
        # print('ppdddkd',req)
        data=RequestFileModel.objects.get(id=id)
        file_path = data.file_id.file.path
          
    extension = os.path.splitext(file_path)[1].lower()  # Get file extension

    text = ""

    if extension in TEXT_FILE_EXTENSIONS:
        # Detect file encoding
        encoding = detect_encoding(file_path)
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
        except (UnicodeDecodeError, IOError):
            return render(request, 'text_to_speech.html', {'error': "Unable to read file content. Ensure it's a valid text file."})

    elif extension in PDF_FILE_EXTENSIONS:
        text = extract_text_from_pdf(file_path)
        if not text:
            return render(request, 'text_to_speech.html', {'error': "Unable to extract text from PDF file."})

    elif extension in DOCX_FILE_EXTENSIONS:
        text = extract_text_from_docx(file_path)
        if not text:
            return render(request, 'text_to_speech.html', {'error': "Unable to extract text from DOCX file."})

    elif extension in DOC_FILE_EXTENSIONS:
        text = extract_text_from_doc(file_path)
        if not text:
            return render(request, 'text_to_speech.html', {'error': "Unable to extract text from DOC file."})

    else:
        return render(request, 'text_to_speech.html', {'error': "Unsupported file format. Please upload a valid text, PDF, or DOC file."})

    # Convert extracted text to speech
    tts = gTTS(text=text, lang='en')
    filename = "speech.mp3"
    extension = os.path.splitext(filename)[1]
    unique_filename = f"{os.path.splitext(filename)[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{extension}"
    output_file_path = os.path.join('static/assets/audio', unique_filename)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # Ensure directory exists
    tts.save(output_file_path)  # Save the audio file

    return render(request, 'text_to_speech.html', {'audio_file': output_file_path})


def logout(request):
    del request.session['email']
    return redirect('index')


def profile(request):
    # UserProfile.objects.all().delete()
    email =  request.session['email']
    user = UserModel.objects.get(email=email)
    data = UserProfile.objects.filter(user_id=user.id).exists()
    if data:
        profile = UserProfile.objects.filter(user_id=user.id)

        user = UserModel.objects.filter(email=email)
        return render(request, 'profile.html', {'profile': profile,'user': user})
    else:
        userdata = UserModel.objects.filter(email=email)

        return render(request, 'updateprofile.html',{'user':userdata})
    
    # return render(request, 'profile.html')

def updateprofile(request):
    email =  request.session['email']
    user = UserModel.objects.get(email=email)
    if request.method == 'POST':
        # name = request.POST.get('name')
        phone = request.POST.get('phone')
        address = request.POST.get('location')
        bio = request.POST['bio']
        image = request.FILES['image']
        data = UserProfile.objects.create(
            user_id=user.id,
           
            phone=phone,
            address=address,
            bio=bio,
            image=image 
        
        )
        data.save()
        return redirect('profile')

def editprofile(request):
    # UserProfile.objects.filter(id=4).delete()
    email =  request.session['email']
    user = UserModel.objects.get(email=email)
    profile = UserModel.objects.filter(email=email)
    if request.method == 'POST':
        phone = request.POST.get('phone')
        address = request.POST.get('location')
        bio = request.POST['bio']
        image = request.FILES['image']
        data = UserProfile.objects.get(user_id=user.id)
        if phone:
            data.phone = phone
        if address :
            data.address = address
        data.bio = bio
        if image:
            data.image = image
        data.save()
        messages.success(request, 'Profile Updated Successfully!')
        return redirect('profile')
    return render(request, 'editprofile.html',{'user':profile})

from django.core.mail import send_mail
from django.conf import settings
def sendrequest(request, id):
    requester = request.session['email']
    data = UploadFileModel.objects.get(id=id)

    if RequestFileModel.objects.filter(requester=requester, file_id=data).exists():
        messages.success(request, 'You already requested this file')
        return redirect('viewfiles')

    req = RequestFileModel.objects.create(
        file_id=data,
        requester=requester
    )
    req.save()

    # Send email notification
    file_owner_email = data.user.email  # Assuming UploadFileModel has a ForeignKey to user
    send_mail(
        'New File Request',
        f'You have a new file request from {requester}. Please check your dashboard.',
        settings.EMAIL_HOST_USER,
        [file_owner_email],
        fail_silently=False,
    )

    messages.success(request, 'Request Sent Successfully!')
    return redirect('viewfiles')
        

def viewrequests(request):
    email =  request.session['email']
    # user = UserModel.objects.get(email=email)
    requests = RequestFileModel.objects.filter(file_id__user__email=email, status='Pending')
    paginator = Paginator(requests, 4)  
    page_number = request.GET.get('page')
    page_data = paginator.get_page(page_number)
    return render(request, 'viewrequests.html', {'data':page_data, 'email':email})

def acceptrequest(request, id):
    email = request.session['email']
    req = RequestFileModel.objects.get(id=id)
    req.status = 'Accepted'
    req.save()

    # Send email notification
    send_mail(
        'File Request Accepted',
        'Your file request has been accepted. You can now access the file.',
        settings.EMAIL_HOST_USER,
        [req.requester],
        fail_silently=False,
    )

    messages.success(request, 'Request Accepted Successfully!')
    return redirect('viewrequests')


def viewresponses(request):
    email =  request.session['email']
    responses = RequestFileModel.objects.filter(requester=email, status='Accepted')
    paginator = Paginator(responses, 4)
    page_number = request.GET.get('page')
    page_data = paginator.get_page(page_number)
    return render(request, 'viewresponses.html',{'data':page_data, 'email':email})


