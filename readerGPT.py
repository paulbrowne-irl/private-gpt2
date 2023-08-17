# because prompt generation can be slow, we have a reader to "playback" logs


import time
from termcolor import colored, cprint

# contants
line_delay = 0.01 # in seconds
line_question ="> Question"
line_answer="> Answer"
line_source="> source_documents"

#Code
file1 = open('privateGPT.log', 'r')
Lines = file1.readlines()

#default settings pre loop
effects=["dark"]
pcolour="light_grey"
delay=0
 

# Loop and print lines
for line in Lines:
    
    #set colour - it continues that colour until otherwise set
    if line.startswith(line_question):
        #question
        effects=["bold"]
        pcolour="white"
        delay=3.2

    if line.startswith(line_answer):
        #default
        effects=[]
        pcolour="light_blue"
        delay=0.8

    if line.startswith(line_source):
        #default
        effects=[]
        pcolour="dark_grey"
        delay=0.1

    # print the line
    print(colored(line, pcolour, attrs=effects), end='')

    

    # calculate the sleep time
    time.sleep(delay)