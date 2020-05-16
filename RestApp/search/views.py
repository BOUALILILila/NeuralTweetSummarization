from django.shortcuts import render
from rest_framework import  status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.reverse import reverse
from . import select_tweets as st
from . import prepare_tweets as pt
from rest_framework.permissions import IsAuthenticated  # <-- Here
import datetime
import json
import coreapi

from . import models

# Create your views here.
class TweetSummary(APIView):

    permission_classes = (IsAuthenticated,)             # <-- And here
    def get(self, request, format='json'):
            """
            get:
            Return a list of tweet ids that constitues the summary fot the given query.

            """
            # Return it in your custom format
            q=request.query_params.get('q')
            since=request.query_params.get('since')
            if q is None:
                return Response({
                    "error": "query not specified"
                    },status=status.HTTP_400_BAD_REQUEST
                )
            if not q.strip():
                return Response({
                    "error": "empty query"
                    },status=status.HTTP_400_BAD_REQUEST
                )
            if since is not None:
                try:
                    datetime.datetime.strptime(since, '%Y-%m-%d')
                except ValueError:
                    return Response({
                    "error": "since date must be YYYY-MM-DD"
                    },status=status.HTTP_400_BAD_REQUEST
                )
            
            tweets,retrieved,candidates=st.get_summary(q,since, models.embed_model)
            return Response({"summary":tweets, "retrieved":retrieved, "candidates":candidates})


class TweetProcessed(APIView):
    permission_classes = (IsAuthenticated,) 
    def post(self,request, format='json'):
        body=request.data
        try:
            tweets=json.loads(body['tweets'])
        except KeyError:
            return Response({
                    "error": "tweets key missing in body"
                    },status=status.HTTP_400_BAD_REQUEST
                )
        processed_tweets=pt.preporcess_submitted_tweets(tweets)
        return Response({"processed_tweets":processed_tweets})


#Doc
from rest_framework.decorators import api_view, renderer_classes
from rest_framework_swagger import renderers as swagger_renderer
from rest_framework import renderers
from .schema import api_schema_generator
@api_view()
@renderer_classes([renderers.CoreJSONRenderer,
                   swagger_renderer.OpenAPIRenderer,
                   swagger_renderer.SwaggerUIRenderer,
                   ])
def schema_view(request):
    api_schema = api_schema_generator()
    return Response(api_schema)
    

