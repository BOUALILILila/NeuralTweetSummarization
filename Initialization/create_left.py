import json
from pprint import pprint
import preprocessor as pre
from nltk.tokenize import RegexpTokenizer

with open('/home/lila/CORPUS/topics/topics_2016.json') as f:
    data = json.load(f)
print(len(data))
left_data=[]
tokenizer = RegexpTokenizer(r'\w+')
for topic in data:
    text=topic['title']
    text=pre.preProcessText(text)
    tokens=tokenizer.tokenize(text)
    stems=pre.stemTokens(tokens)
    text=" ".join([stem for stem in stems])
    left_data+= [{"title":text,"topid":topic['topid']}]
print(len(left_data))
print(left_data[:10])
with open('/home/lila/CORPUS/left/left_2016.json', 'w') as outfile:
    json.dump(left_data, outfile)