'''
export FLASK_APP=web_server.py
export FLASK_DEBUG=True
'''
from flask import Flask
from flask import render_template, request
import recommender

app = Flask('movie recommender')


@app.route('/') # Decorator (add extra functionality to a function) In this case enable the function to talk to Flask
def hello():
    return render_template(/home/Llana13/movie_recommender/templates/main_page.html)

@app.route('/recommend')
def run_recommender():
    try:
        n = int(request.args['n_movies'])
    except ValueError:
        n = 3
    fav_movie0 = request.args['fav_movie0']
    fav_movie1 = request.args['fav_movie1']
    fav_movie2 = request.args['fav_movie2']

    fav_movies = [fav_movie0,fav_movie1,fav_movie2]

    result = recommender.get_recommendation(fav_movies,n)

    return render_template('recommendation.html',title='I recommend you to watch:',
     movie_name = result)
    # fill in values in the JINJA2 template
