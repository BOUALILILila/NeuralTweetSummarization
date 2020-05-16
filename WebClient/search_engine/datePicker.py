from datetime import datetime, timedelta

from django import forms
from bootstrap_datepicker_plus import DatePickerInput
from .Config import LIMIT_SINCE

class MyDatePickerInput(DatePickerInput):
    template_name = 'search_engine/dateInput.html'

class Form(forms.Form):
    
    yesterday=(datetime.now()-timedelta(days=1)).strftime('%Y-%m-%d')
    since_date=(datetime.now()-timedelta(days=LIMIT_SINCE)).strftime('%Y-%m-%d')
    '''
    date_time_str = '2019-05-28'  
    date_time = datetime.strptime(date_time_str, '%Y-%m-%d')
    yesterday=(date_time-timedelta(days=1)).strftime('%Y-%m-%d')
    since_date=(date_time-timedelta(days=LIMIT_SINCE)).strftime('%Y-%m-%d')
    '''
    date = forms.DateField(
            widget=MyDatePickerInput( options={
                        "format": "MM/DD/YYYY", 
                        "showClose": True,
                        "showClear": True,
                        "showTodayButton": True,
                        'minDate': since_date,
                        'maxDate': yesterday,
                    },attrs={
                        "data-toggle":"tooltip" ,
                        "data-placement":"bottom" ,
                        "title":"TChoose the start day of the summarization periode"
                    })
        )

