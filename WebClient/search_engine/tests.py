from django.test import TestCase

# Create your tests here.
from django.test import Client
from django.urls import reverse
import json

class Search_Test(TestCase):
        
    def setUp(self):
        with open('context.json') as f:
            self.resp=json.load(f)

    def test_search_form(self):
        response = self.client.post(reverse('search_engine:search'), {'search':'algeria', 'date': '14/04/2019'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['summary'], len(self.resp['summary']))

    def test_search_form_error(self):
        response = self.client.post(reverse('search_engine:search'), {'search':'    ', 'date': '14/04/2019'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['no_tweet'], 1)