import xml.etree.ElementTree as ET
import json
from xml.dom import minidom

with open('/projets/iris/PROJETS/lboualil/CORPUS/right/right_2015.json') as f:
    tweets=json.load(f)


for tweet in tweets:
    tree = ET.ElementTree()
    doc=ET.Element('DOC')
    # create the file structure 
    doc_id= ET.SubElement(doc, 'DOCNO')  
    doc_text = ET.SubElement(doc, 'TEXT')   
    doc_id.text = tweet['tweetid'] 
    doc_text.text = tweet['text']
    # create a new XML file with the results
    tree._setroot(doc)
    name="/projets/iris/PROJETS/lboualil/CORPUS/TREC/TREC_2015/"+tweet['tweetid']+".xml"
    xmlstr = minidom.parseString(ET.tostring(doc)).toprettyxml(indent="   ")
    with open(name, "w") as f:
        f.write(xmlstr)

