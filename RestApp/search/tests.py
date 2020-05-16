
# Create your tests here.
from rest_framework.test import APITestCase
from rest_framework.test import APIRequestFactory
from rest_framework.views import status
from django.urls import reverse
from django.contrib.auth import get_user_model
from rest_framework.authtoken.models import Token
from .views import TweetSummary


class TestSummary(APITestCase):
    def setUp(self):
        # ...
        self.user = self.setup_user()
        self.token = Token.objects.create(user=self.user)
        self.token.save()

    @staticmethod
    def setup_user():
        User = get_user_model()
        return User.objects.create_user(
            'test',
            email='testuser@test.com',
            password='test'
        )

    def test_summ(self):
        # hit the API endpoint
        response = self.client.get(
            reverse("tweet_summ"),
            {'q':'algeria'},
            HTTP_AUTHORIZATION='Token {}'.format(self.token.key)
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        if response.status_code==400:
            print(response.data)

    def test_summ_no_query(self):
        # hit the API endpoint
        response = self.client.get(
            reverse("tweet_summ"),
            HTTP_AUTHORIZATION='Token {}'.format(self.token.key)
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        if response.status_code==400:
            print(response.data)
            
    def test_summ_empty_query(self):
        # hit the API endpoint
        response = self.client.get(
            reverse("tweet_summ"),
            {'q':' '},
            HTTP_AUTHORIZATION='Token {}'.format(self.token.key)
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        if response.status_code==400:
            print(response.data)

    def test_summ_unauthorized(self):
        # hit the API endpoint
        response = self.client.get(
            reverse("tweet_summ"),
            {'q':'Twitter'},
        )
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        if response.status_code==401:
            print(response.data)

        