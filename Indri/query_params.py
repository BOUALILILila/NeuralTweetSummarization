import xml.etree.ElementTree as ET
import json
from xml.dom import minidom

with open('/data/CORPUS/left/left_2017_nist_evaluated_qrels_real.json') as f:
    topics=json.load(f) 

tree = ET.ElementTree()
root=ET.Element('parameters')
for topic in topics:
    #if topic['topid'] in evaluated_topic_ids:
        q=ET.SubElement(root,'query')
        # create the file structure 
        q_id= ET.SubElement(q, 'number')  
        q_text = ET.SubElement(q, 'text')
        q_id.text = topic['topid'] 
        q_text.text =topic['title']
# create a new XML file with the results
tree._setroot(root)
name='/data/okapi/query_parameters_baseline_2017.xml'
xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
with open(name, "w") as f:
    f.write(xmlstr)

