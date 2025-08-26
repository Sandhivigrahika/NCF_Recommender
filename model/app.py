import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import random
from tmdb_api_file import get_movie_details
import requests




movies = pd.read_csv(
    "movies.dat",
    sep="::",
    engine="python",  # Needed because "::" is multi-character separator
    header=None,      # No header row in file
    names=["movieId", "title", "genres"],  # Assign column names
    encoding="latin-1"
)

movie_dict = dict(zip(movies["movieId"], movies["title"]))





# Load model + mappings

@st.cache_resource
def load_model_and_mappings():
    model = tf.keras.models.load_model("ncf_model.h5",compile=False)  # update path if needed
    with open("user2id.pkl", "rb") as f:
        user2id = pickle.load(f)
    with open("movie2id.pkl", "rb") as f:
        movie2id = pickle.load(f)
    with open("id2movie.pkl", "rb") as f:
        id2movie = pickle.load(f)

    # load later
    with open("movies_with_ratings.pkl", "rb") as f:
        movies_with_ratings = pickle.load(f)
    return model, user2id, movie2id, id2movie, movies_with_ratings

model, user2id, movie2id, id2movie, movies_with_ratings = load_model_and_mappings()




# Recommendation function

def recommend_movies(user_id, top_n=10):
    if user_id not in user2id:
        return None

    uid = user2id[user_id]
    all_movie_ids = list(movie2id.values())

    user_array = np.full(len(all_movie_ids), uid)
    movie_array = np.array(all_movie_ids)

    preds = model.predict([user_array, movie_array], verbose=0).flatten()
    top_indices = preds.argsort()[-top_n:][::-1]

    recommended_movie_ids = movie_array[top_indices]

    # 🔹 map movieId → title using movie_dict
    recommended_titles = [
        movie_dict[mid] if mid in movie_dict else f"Movie ID {mid}"
        for mid in recommended_movie_ids
    ]

    return recommended_titles


# Streamlit UI


import streamlit as st
import time

with st.spinner("✨ Setting up your personalized movie world..."):
    time.sleep(2)  # simulate loading
st.success("New Users will have to rate a few movies before we can recommend 🎬")

quotes = [
    "I'm king of the world! - Titanic, 1997",
    "Nobody puts Baby in a corner. - Dirty Dancing, 1987",
    "Snap out of it! - Moonstruck, 1987",
    "I feel the need - the need for speed! - Top Gun, 1986",
    "Cinderella story. Outta nowhere. A former greenskeeper, now, about to become the Masters champion. It looks like a mirac...It's in the hole! It's in the hole! It's in the hole! - Caddyshack, 1980",
    "A martini. Shaken, not stirred. - Goldfinger, 1964",
    "Listen to me, mister. You're my knight in shining armor. Don't you forget it. You're going to get back on that horse, and I'm going to be right behind you, holding on tight, and away we're gonna go, go, go! - On Golden Pond, 1981",
    "Attica! Attica! - Dog Day Afternoon, 1975",
    "Oh, no, it wasn't the airplanes. It was Beauty killed the Beast. - King Kong, 1933",
    "Toga! Toga! - National Lampoon's Animal House, 1978",
    "Yo, Adrian! - Rocky II, 1979",
    "Open the pod bay doors, HAL. - 2001: A Space Odyssey, 1968",
    "Hasta la vista, baby. - Terminator 2: Judgment Day, 1991",
    "Forget it, Jake, it's Chinatown. - Chinatown, 1974",
    "No wire hangers, ever! - Mommie Dearest, 1981",
    "Is it safe? - Marathon Man, 1976",
    "Here's Johnny! - The Shining, 1980",
    "Get your stinking paws off me, you damned dirty ape. - Planet of the Apes, 1968",
    "Gentlemen, you can't fight in here! This is the War Room! - Dr. Strangelove, 1964",
    "What a dump. - Beyond the Forest, 1949",
    "Well, here's another nice mess you've gotten me into! - Sons of the Desert, 1933",
    "Keep your friends close, but your enemies closer. - The Godfather Part II, 1974",
    "A boy's best friend is his mother. - Psycho, 1960",
    "There's no crying in baseball! - A League of Their Own, 1992",
    "You had me at hello. - Jerry Maguire, 1996",
    "Houston, we have a problem. - Apollo 13, 1995",
    "Well, nobody's perfect. - Some Like It Hot, 1959",
    "Oh, Jerry, don't let's ask for the moon. We have the stars. - Now, Voyager, 1942",
    "I see dead people. - The Sixth Sense, 1999",
    "Plastics. - The Graduate, 1967",
    "Mama always said life was like a box of chocolates. You never know what you're gonna get. - Forrest Gump, 1994",
    "Today, I consider myself the luckiest man on the face of the earth. - The Pride of the Yankees, 1942",
    "Badges? We ain't got no badges! We don't need no badges! I don't have to show you any stinking badges! - The Treasure of the Sierra Madre, 1948",
    "You know how to whistle, don't you, Steve? You just put your lips together and blow. - To Have and Have Not, 1944",
    "Round up the usual suspects. - Casablanca, 1942",
    "I want to be alone. - Grand Hotel, 1932",
    "Play it, Sam. Play 'As Time Goes By.' - Casablanca, 1942",
    "Why don't you come up sometime and see me? - She Done Him Wrong, 1933",
    "I am big! It's the pictures that got small. - Sunset Blvd., 1950",
    "Bond. James Bond. - Dr. No, 1962",
    "Louis, I think this is the beginning of a beautiful friendship. - Casablanca, 1942",
    "Made it, Ma! Top of the world! - White Heat, 1949",
    "They call me Mister Tibbs! - In the Heat of the Night, 1967",
    "The stuff that dreams are made of. - The Maltese Falcon, 1941",
    "I love the smell of napalm in the morning. - Apocalypse Now, 1979",
    "You talking to me? - Taxi Driver, 1976",
    "May the Force be with you. - Star Wars, 1977",
    "Go ahead, make my day. - Sudden Impact, 1983",
    "Toto, I've got a feeling we're not in Kansas anymore. - The Wizard of Oz, 1939",
    "I'm going to make him an offer he can't refuse. - The Godfather, 1972",
    "Frankly, my dear, I don't give a damn. - Gone With the Wind, 1939",
    "You don't understand! I coulda had class. I coulda been a contender. I could've been somebody, instead of a bum, which is what I am. - On the Waterfront, 1954",
    "Here's looking at you, kid. - Casablanca, 1942",
    "All right, Mr. DeMille, I'm ready for my close-up. - Sunset Boulevard, 1950",
    "Fasten your seatbelts. It's going to be a bumpy night. - All About Eve, 1950",
    "What we've got here is failure to communicate. - Cool Hand Luke, 1967",
    "Love means never having to say you're sorry. - Love Story, 1970",
    "E.T. phone home. - E.T. The Extra-Terrestrial, 1982",
    "Rosebud. - Citizen Kane, 1941",
    "I'm as mad as hell, and I'm not going to take this anymore! - Network, 1976",
    "A census taker once tried to test me. I ate his liver with some fava beans and a nice Chianti. - The Silence of the Lambs, 1991",
    "There's no place like home. - The Wizard of Oz, 1939",
    "Show me the money! - Jerry Maguire, 1996",
    "I'm walking here! I'm walking here! - Midnight Cowboy, 1969",
    "You can't handle the truth! - A Few Good Men, 1992",
    "After all, tomorrow is another day! - Gone With the Wind, 1939",
    "I'll have what she's having. - When Harry Met Sally, 1989",
    "You're gonna need a bigger boat. - Jaws, 1975",
    "I'll be back. - The Terminator, 1984",
    "If you build it, he will come. - Field of Dreams, 1989",
    "We rob banks. - Bonnie and Clyde, 1967",
    "We'll always have Paris. - Casablanca, 1942",
    "Stella! Hey, Stella! - A Streetcar Named Desire, 1951",
    "Shane. Shane. Come back! - Shane, 1953",
    "It's alive! It's alive! - Frankenstein, 1931",
    "You've got to ask yourself one question: 'Do I feel lucky?' Well, do ya, punk? - Dirty Harry, 1971",
    "One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't know. - Animal Crackers, 1930",
    "La-dee-da, la-dee-da. - Annie Hall, 1977",
    "Greed, for lack of a better word, is good. - Wall Street, 1987",
    "As God is my witness, I'll never be hungry again. - Gone With the Wind, 1939",
    "Say 'hello' to my little friend! - Scarface, 1983",
    "Mrs. Robinson, you're trying to seduce me. Aren't you? - The Graduate, 1967",
    "Elementary, my dear Watson. - The Adventures of Sherlock Holmes, 1929",
    "Of all the gin joints in all the towns in all the world, she walks into mine. - Casablanca, 1942",
    "They're here! - Poltergeist, 1982",
    "Wait a minute, wait a minute. You ain't heard nothin' yet! - The Jazz Singer, 1927",
    "Mother of mercy, is this the end of Rico? - Little Caesar, 1930",
    "I have always depended on the kindness of strangers. - A Streetcar Named Desire, 1951",
    "Soylent Green is people! - Soylent Green, 1973",
    "Surely you can't be serious. I am serious—and don't call me Shirley. - Airplane!, 1980",
    "Hello, gorgeous. - Funny Girl, 1968",
    "Listen to them. Children of the night. What music they make. - Dracula, 1931",
    "My precious. - The Lord of the Rings: Two Towers, 2002",
    "Sawyer, you're going out a youngster, but you've got to come back a star! - 42nd Street, 1933",
    "Tell 'em to go out there with all they got and win just one for the Gipper. - Knute Rockne All American, 1940",
    "Who's on first. - The Naughty Nineties, 1945",
    "Life is a banquet, and most poor suckers are starving to death! - Auntie Mame, 1958",
    "Carpe diem. Seize the day, boys. Make your lives extraordinary. - Dead Poets Society, 1989",
    "My mother thanks you. My father thanks you. My sister thanks you. And I thank you. - Yankee Doodle Dandy, 1942",
    "I'll get you, my pretty, and your little dog, too! - The Wizard of Oz, 1939",
    "Just keep swimming. - Finding Nemo, 2003"
]
st.sidebar.markdown("*Movie quote of the moment*")
st.sidebar.success(random.choice(quotes))

st.title("🎬 Movie Recommender System (NCF)")
st.write("Enter your User ID to get personalized recommendations.")

user_id_input = st.text_input("Enter User ID:")

if st.button("Get Recommendations"):
    if user_id_input.strip():
        try:
            user_id_int = int(user_id_input)
            recommendations = recommend_movies(user_id_int)
            if recommendations:
                st.subheader("Top Recommendations for you:")
                for i, movie in enumerate(recommendations, 1):
                    st.write(f"{i}. {movie}")
                    poster, summary = get_movie_details(movie)
                    if poster:
                        st.image(poster, caption=movie)
                    st.write("**Summary**", summary)
            else:
                st.warning("User ID not found. Please try again or sign up as a new user.")
        except ValueError:
            st.error("Please enter a valid numeric User ID.")
    else:
        st.warning("Please enter a User ID first.")




# Cold Start Recommendation - Uses genre to recommend top movies from that particular genre
def cold_start_recommendations(genre_choice,top_n=10):
    #filter movies which are from the chosen genre
    genre_movies = movies_with_ratings[movies_with_ratings["genres"].str.contains(genre_choice,na=False)]\
    #sort by average rating
    genre_movies = genre_movies.sort_values("avg_rating",ascending=False)

    #pick top n
    top_movies = genre_movies.head(top_n)

    return top_movies["title"].tolist()




# Cold Start UI



all_genres = sorted(["Action", "Comedy", "Drama", "Romance", "Thriller", "Sci-Fi", "Horror", "Animation"])

st.write("🎬 Pick 2–3 genres you love to get started:")
genre_choice = st.selectbox("Choose genres:", all_genres)

if st.button("Get Starter Recommendations"):
    recs = cold_start_recommendations(genre_choice, top_n=10)
    st.success("Here are some great movies for you 🎥:")
    for movie in recs:
        st.write(f"🍿 {movie}")
        poster, summary = get_movie_details(movie)
        if poster:
            st.image(poster, caption=movie)
        st.write("**Summary**", summary)