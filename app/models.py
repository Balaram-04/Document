from django.db import models
import os 
from django.contrib.auth.models import AbstractUser, Group, Permission
from django.contrib.auth.models import User
import random
import string
from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.core.validators import RegexValidator
from django.core.mail import send_mail
from django.utils.crypto import get_random_string
from django.db import models
from django.core.validators import RegexValidator

class AdminUserModel(models.Model):
    admin_id = models.CharField(
        max_length=10, 
        unique=True, 
        validators=[RegexValidator(regex=r'^DV\d{8}$', message="Admin ID must start with 'DV' followed by 8 digits.")]
    )
    email = models.EmailField(max_length=255, unique=True)
    password = models.CharField(max_length=255)
    otp = models.IntegerField(null=True)
    is_verified = models.BooleanField(default=False)  # Email OTP verification

    def __str__(self):
        return self.admin_id

    class Meta:
        db_table = "AdminUserModel"


# Custom Manager for Admin Users
'''class AdminManager(BaseUserManager):
    def create_admin(self, admin_id, email, password=None):
        if not admin_id.startswith("DVM"):
            raise ValueError("Admin ID must start with 'DVM'")

        if not email:
            raise ValueError("Admins must have an email address")

        admin = self.model(admin_id=admin_id, email=self.normalize_email(email))
        admin.set_password(password)  # Hashes the password
        admin.otp = get_random_string(length=6, allowed_chars='0123456789')  # Generate OTP
        admin.save(using=self._db)
        return admin'''

'''# Admin Model
class AdminUser(AbstractBaseUser):
    admin_id = models.CharField(
        max_length=10,
        unique=True,
        validators=[RegexValidator(r'^DVM\d{6}$', "Admin ID must start with 'DVM' followed by 6 digits")],
    )
    email = models.EmailField(unique=True)
    otp = models.CharField(max_length=6, blank=True, null=True)
    is_verified = models.BooleanField(default=False)  # Email OTP verification status
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=True)

    objects = AdminManager()

    USERNAME_FIELD = "admin_id"
    REQUIRED_FIELDS = ["email"]

    def __str__(self):
        return f"Admin {self.admin_id} - {self.email}"

    def send_otp(self):
        """Generates and sends OTP via email."""
        self.otp = ''.join(random.choices(string.digits, k=6))  # Generate 6-digit OTP
        self.save()

        send_mail(
            "Your Admin OTP Verification Code",
            f"Your OTP code is {self.otp}. Please enter this to verify your email.",
            "noreply@yourdomain.com",
            [self.email],
            fail_silently=False,
        )'''

class UserModel(models.Model):
    username = models.CharField(max_length=255)
    email = models.EmailField(max_length=255, unique=True)
    password = models.CharField(max_length=255)
    #profile_image = models.ImageField(upload_to='user_images/', null=True, blank=True)
    otp = models.IntegerField(null=True)
    is_admin = models.BooleanField(default=False) 


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
    is_duplicate = models.BooleanField(default=False)  
    similarity = models.FloatField(default=0.0)
    insights_path = models.CharField(max_length=500, blank=True, null=True)
    status = models.CharField(
        max_length=20,
        choices=[("Pending", "Pending"), ("Accepted", "Accepted"), ("Rejected", "Rejected")],
        default="Pending",
    )

    def __str__(self):
        return f"{self.filename} - {self.get_status_display()}"
    
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

