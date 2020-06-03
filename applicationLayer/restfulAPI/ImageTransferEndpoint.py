from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from Django.domainLayer.service.NeuralStyle import NeuralStyle
from Django.domainLayer.service.FileDirectory import FileDirectory

from multiprocessing import Process


def test(data):
    # neuralStyle = NeuralStyle("lion.jpg", ["kandinsky.jpg"])
    neuralStyle = NeuralStyle(data)
    neuralStyle.render_single_image()


@api_view(['POST'])
def transferImage(request):
    if request.method == 'POST':
        neuralStyle = NeuralStyle(request.data)
        process = Process(target=neuralStyle.render_single_image)
        process.start()
    return HttpResponse(1)
@api_view(['POST'])
def transferVideo(request):
    if request.method == 'POST':
        neuralStyle = NeuralStyle(request.data)
        process = Process(target=neuralStyle.render_video)
        process.start()
    return HttpResponse(1)

@api_view(['POST'])
def findDirectory(request):
    if request.method == 'POST':
        fileDirectory = FileDirectory(request.data)
    return JsonResponse( fileDirectory.findFileDirectory())
