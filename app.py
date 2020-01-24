from flask import Flask, escape, render_template, request
import json
app = Flask(__name__)
themes = ['Spectrum', 'Solid', 'Quicksort', 'Fade', 'Sparkle', 'Slide', 'Off']
curTheme = 'Spectrum'

@app.route('/<themeName>')
def setTheme(themeName):
    global curTheme
    curTheme = themeName
    return render_template('home.html', themes = themes, curTheme = themeName)

@app.route('/')
def home():
    return render_template('home.html', themes = themes, curTheme = curTheme)

@app.route('/get_theme', methods=['GET'])
def get_settings():
	return json.dumps({'theme': curTheme})
