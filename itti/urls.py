from django.conf.urls import url 
from itti import views 

urlpatterns = [ 
    url(r'^api/images$', views.image_list)
]