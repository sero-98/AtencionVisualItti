from rest_framework import serializers
from itti.models import Saliency
        

class SaliencySerializer(serializers.ModelSerializer):
    
    class Meta:
        model = Saliency
        fields = ('idClothe',
                  'vectorImage',
                  'urlImage')
