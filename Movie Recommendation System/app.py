from flask import Flask, render_template, request, redirect, url_for, session, flash

from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

app = Flask(__name__)
app.secret_key = "your_secret_key"
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                      (username, email, hashed_password))
            conn.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists!", "danger")
            return redirect(url_for('register'))
        finally:
            conn.close()
    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):  # user[3] is the hashed password
            session['user_id'] = user[0]  # user[0] is the user ID
            session['username'] = user[1]  # user[1] is the username
            flash(f"Welcome back, {user[1]}!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password!", "danger")
            return redirect(url_for('login'))
    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))


# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Preprocess data
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
user_movie_matrix.fillna(0, inplace=True)
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Recommend movies for existing users
def recommend_movies(user_id, num_recommendations=5):
    if user_id not in user_movie_matrix.index:
        return []

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    user_ratings = user_movie_matrix.mul(similar_users, axis=0).sum(axis=0)
    user_ratings /= similar_users.sum()
    recommendations = user_ratings[user_movie_matrix.loc[user_id] == 0]
    recommendations = recommendations.sort_values(ascending=False).head(num_recommendations)

    recommended_movies = movies[movies['movieId'].isin(recommendations.index)][['movieId', 'title']]
    return recommended_movies.to_dict(orient='records')

# Recommend movies for new users
def recommend_for_new_user(preferences, num_recommendations=5):
    new_user_ratings = pd.Series(0.0, index=user_movie_matrix.columns)
    for movie_id, rating in preferences.items():
        new_user_ratings[movie_id] = float(rating)
    user_movie_matrix.loc['new_user'] = new_user_ratings

    updated_similarity = cosine_similarity(user_movie_matrix)
    similarity_df = pd.DataFrame(updated_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    similar_users = similarity_df.loc['new_user'].sort_values(ascending=False).iloc[1:]
    user_ratings = user_movie_matrix.mul(similar_users, axis=0).sum(axis=0)
    user_ratings /= similar_users.sum()
    recommendations = user_ratings[user_movie_matrix.loc['new_user'] == 0]
    recommendations = recommendations.sort_values(ascending=False).head(num_recommendations)

    recommended_movies = movies[movies['movieId'].isin(recommendations.index)][['movieId', 'title']]
    return recommended_movies.to_dict(orient='records')

@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('home.html', username=session['username'])
    flash("Please log in to access the home page.", "info")
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/movies', methods=['GET', 'POST'])
def movies_page():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if user_id:
            user_id = int(user_id)
            recommended_movies = recommend_movies(user_id)
            return render_template("movies.html", movies=recommended_movies)
    return render_template("movies.html", movies=[])

@app.route('/new_user', methods=['GET', 'POST'])
def new_user():
    if request.method == 'POST':
        preferences = {}
        for movie_id in request.form.keys():
            preferences[int(movie_id)] = float(request.form[movie_id])
        recommended_movies = recommend_for_new_user(preferences)
        return render_template("new_user_recommendations.html", movies=recommended_movies)

    initial_movies = movies.sample(10).to_dict(orient='records')
    return render_template("new_user.html", movies=initial_movies)
@app.route('/recommendation')
def recommendation():
    return render_template("recommendation.html")
@app.route('/register', methods=['GET', 'POST'])



@app.route('/contact')
def contact():
    return render_template("contact.html")


if __name__ == '__main__':
    init_db()  # Initialize database
    app.run(debug=True)