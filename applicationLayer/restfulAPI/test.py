from django.http import HttpResponse
from rest_framework.decorators import api_view


@api_view(['POST', 'GET', 'PUT'])
def hello(request):
    if request.method == "POST":
        for key, value in request.data.items():
            print(key)
    return HttpResponse(1)
