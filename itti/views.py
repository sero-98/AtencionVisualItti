from django.shortcuts import render
from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser
from rest_framework import status

from itti.models import Saliency
from itti.serializers import SaliencySerializer
from rest_framework.decorators import api_view

import cv2
from itti.pySaliencyMap import pySaliencyMap
import urllib.request
import numpy as np

# Create your views here.


@api_view(['POST'])
def image_list(request):

    # Recupere todos los tutoriales / busque por titulo de la base de datos de MongoDB:
    # if request.method == 'GET':
    #     saliencys = Saliency.objects.all()
    #     title = request.GET.get('title', None)
    #     if title is not None:
    #         saliencys = saliencys.filter(title__icontains=title)
    #     saliencys_serializer = SaliencySerializer(saliencys, many=True)
    #     return JsonResponse(saliencys_serializer.data, safe=False)

    # Crear y guardar un nuevo tutorial:
    if request.method == 'POST':
        print('####################################')
        print('####################################')
        print('####################################')

        vectorimage_data = JSONParser().parse(request)

        urlImage = vectorimage_data['urlImage']
        print('urlImage', urlImage)

        # READ
        url_response = urllib.request.urlopen(urlImage)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        # INITIALIZE
        imgsize = img.shape
        img_width = imgsize[1]  # img_width = 864
        img_height = imgsize[0]  # img_height = 1080

        sm = pySaliencyMap(img_width, img_height)  # def __init__

        # computation
        saliency_map = sm.SMGetSM(img)
        print('saliency_map', saliency_map)

        vectorimage_data['vectorImage'] = saliency_map

        print('####################################')
        print('####################################')
        print('####################################')

        saliency_serializer = SaliencySerializer(data=vectorimage_data)
        print('saliency_serializer', saliency_serializer)
        print('####################################')
        print('####################################')
        print('####################################')

        if saliency_serializer.is_valid():
            saliency_serializer.save()
            return JsonResponse(saliency_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(saliency_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
