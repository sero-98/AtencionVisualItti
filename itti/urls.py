from django.conf.urls import url 
from itti import views 

urlpatterns = [ 
    url(r'^api/images/arriba$', views.image_list_arriba),
    url(r'^api/images/abajo$', views.image_list_abajo),
    url(r'^api/images/full$', views.image_list_full)
]