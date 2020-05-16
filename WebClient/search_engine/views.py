from django.shortcuts import render
from django.http import HttpResponse
from django.core.paginator import Paginator,PageNotAnInteger, EmptyPage
# Create your views here.
from . import datePicker
from . import apiConsumer
import json
from django.utils.safestring import mark_safe
from django.template.loader import render_to_string
from datetime import datetime
import dateutil.parser
from django.views import View
import ast,sys

response_saved=None
paginator=None
class Search(View):

    def post(self,request):
        global response_saved, paginator

        form=datePicker.Form()
        if 'search' in request.POST:
            paginator=None
            response_saved=None
            q=request.POST['search']
            date=request.POST['date']
            date_parts=date.split('/')
            date=f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
            if not q.strip():
                context={
                    'placeholder_query':q,
                    'no_tweet': 1,
                    'error':  "Sorry! No available tweets",
                    'retrieved': 0,
                    'candidates': 0,
                    'form':form
                }
            else :
                response= apiConsumer.get_summarization(q,date)
                if response==400:
                    context={
                        'placeholder_query':q,
                        'no_tweet': 1,
                        'error': "Sorry! No available tweets",
                        'retrieved': 0,
                        'candidates': 0,
                        'form':form
                    }
                elif response==429:
                    context={
                        'placeholder_query':q,
                        'server': 1,
                        'error': "Ouie! Rate limite exceeded for this ressource",
                        'retrieved': 0,
                        'candidates': 0,
                        'form':form
                    }
                elif response==500:
                    context={
                        'placeholder_query':q,
                        'server': 1,
                        'error': "Oops! Couldn't reach server",
                        'retrieved': 0,
                        'candidates': 0,
                        'form':form
                    }
                else:
                    tweets_all=response['summary']
                    if len(tweets_all)==0:
                        context={
                        'placeholder_query':q,
                        'no_tweet': 1,
                        'error': "Sorry! No available tweets",
                        'retrieved': 0,
                        'candidates': 0,
                        'form':form
                        }
                    else:
                        
                        for t in tweets_all:
                            dt=dateutil.parser.parse(t['created_at'])
                            t['created_at']=datetime.strftime(dt,'%I:%M%p - %d %b %Y')
                            #t['user'] = ast.literal_eval(t['user'])
                            #print( type(t['user']),file=sys.stderr)
                        paginator = Paginator(tweets_all, 5) # Show 5 tweets per page
                        page = request.GET.get('page')
                        try:
                            tweets = paginator.get_page(page)
                        except PageNotAnInteger:
                            tweets = paginator.get_page(1)
                        except EmptyPage:
                            tweets = paginator.get_page(paginator.num_pages)
                            
                        candidates_stat= float(response['candidates'])/float(response['retrieved'])*100
                        summary_stat= len(tweets_all)/float(response['retrieved'])*100
                        response_saved={'retrieved':response['retrieved'],'candidates': response['candidates'],
                                        'candidates_stat': candidates_stat,'summary_stat':summary_stat,
                                        'summary':len(tweets_all),'q':q}
                        context = {
                            'placeholder_query':q,
                            #'all_t': k,
                            'tweets': tweets,
                            'retrieved': response['retrieved'],
                            'candidates': response['candidates'],
                            'candidates_stat': candidates_stat,
                            'summary_stat':summary_stat,
                            'summary':len(response['summary']),
                            'form':form,
                            'js_data': json.dumps(q)
                        }           
            return render(request,'search_engine/search.html',context)
        '''
        elif request.is_ajax():
            print('hi')
            page = request.GET.get('page')
            print(request.GET.get('t'))
            paginator = paginator(request.GET.get('t'), 5)
            try:
                tweets = paginator.get_page(page)
            except PageNotAnInteger:
                tweets = paginator.get_page(1)
            except EmptyPage:
                tweets = paginator.get_page(paginator.num_pages)        
            context = {
                        'placeholder_query': request.GET.get['q'],
                        'tweets': tweets,
                        'retrieved': request.GET.get['retrieved'],
                        'candidates': request.GET.get['candidates'],
                        #'candidates_stat': response_saved['candidates_stat'],
                        #'summary_stat':response_saved['summary_stat'],
                        'summary': request.GET.get['summary'],
                        'form':datePicker.Form()
                    }
            html = render_to_string('search_engine/search.html',context)
            return HttpResponse(html)
        '''
        return render(request, 'base.html',{'form':form})
    def get(self,request):
        if ('page' in request.GET and response_saved is not None):
            page = request.GET.get('page')
            try:
                tweets = paginator.get_page(page)
            except PageNotAnInteger:
                tweets = paginator.get_page(1)
            except EmptyPage:
                tweets = paginator.get_page(paginator.num_pages)        
            context = {
                        'placeholder_query': response_saved['q'],
                        'tweets': tweets,
                        'retrieved': response_saved['retrieved'],
                        'candidates': response_saved['candidates'],
                        'candidates_stat': response_saved['candidates_stat'],
                        'summary_stat':response_saved['summary_stat'],
                        'summary': response_saved['summary'],
                        'form':datePicker.Form()
                    }
            return render(request,'search_engine/search.html',context)
        return render(request, 'base.html',{'form':datePicker.Form()})


def home(request): 
    form=datePicker.Form()
    return render(request, 'base.html',{'form':form})