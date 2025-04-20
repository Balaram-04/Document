from django import forms
from .models import AdminUser
from django.contrib.auth.forms import AuthenticationForm

class AdminLoginForm(AuthenticationForm):
    admin_id = forms.CharField(
        max_length=10,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "DVM123456"}),
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={"class": "form-control", "placeholder": "Password"}),
    )
