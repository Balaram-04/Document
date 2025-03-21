from django.db import models
import os 
# Create your models here.


class UserModel(models.Model):
    username = models.CharField(max_length=255)
    email = models.EmailField(max_length=255, unique=True)
    password = models.CharField(max_length=255)
    otp = models.IntegerField(null=True)
   


    def __str__(self):
        return self.username
    
    class Meta:
        db_table = "UserModel"  # Optional, but can be used to specify the table name

class UserProfile(models.Model):
    user_id = models.IntegerField(null=True)
    phone = models.IntegerField()
    address = models.CharField(max_length=255)
    image = models.FileField(upload_to=os.path.join('static/assets/' 'UserProfiles'))
    bio = models.TextField()
    def __str__(self):
        return self.user.username
    class Meta:
        db_table = "UserProfile"  # Optional, but can be used to specify the table name
    

class UploadFileModel(models.Model):
    file = models.FileField(upload_to=os.path.join('static/assets' 'Files'))
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(UserModel, on_delete=models.CASCADE)
    filename = models.CharField(max_length=100)
    datahash = models.TextField(null=True)
    
    
    def __str__(self):
        return self.file.name
    class Meta:
        db_table = "UploadFileModel"  # Optional, but can be used to specify the


class RequestFileModel(models.Model):
    file_id = models.ForeignKey(UploadFileModel, on_delete=models.CASCADE)
    requester =  models.EmailField()
    request_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=255, default='Pending')

    def __str__(self):
        return self.requester
    class Meta:
        db_table = "RequestFileModel"  # Optional, but can be used to specify the