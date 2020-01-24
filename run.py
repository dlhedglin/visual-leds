from multiprocessing import Process, Value, Array, Queue
import requests
import json
from time import sleep
from themes import Quicksort, Solid, Slide, Off, Spectrum, Sparkle, Fade

def getCurrentTheme(queue):
    while True:
        url = 'http://127.0.0.1:5000/get_theme'
        try:
            theme = requests.get(url)
        except requests.ConnectionError:
            print('Api request failed')
            continue
        newTheme = theme.json()['theme']
        queue.put(newTheme)
        sleep(.5)
def startTheme(theme):
    globals()[theme]().start()

curTheme = None
visualizer = None
queue = Queue()

p = Process(target=getCurrentTheme, args=(queue,))
p.start()

while True:
    if not queue.empty():
        newTheme = queue.get()
        if(newTheme != curTheme):
            if(visualizer != None):
                visualizer.terminate()
                visualizer.join()
            curTheme = newTheme
            print(curTheme)
            visualizer = Process(target=startTheme, args=(curTheme,))
            visualizer.start()

