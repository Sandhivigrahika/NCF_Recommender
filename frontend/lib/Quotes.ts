export interface Quote {
  text: string;
  movie: string;
  year: string;
  poster: string; // TMDB w1280 backdrop for best banner quality
}
 
export const QUOTES: Quote[] = [
  {
    text: "Frankly, my dear, I don't give a damn.",
    movie: "Gone With the Wind",
    year: "1939",
    poster: "https://image.tmdb.org/t/p/w1280/azsurvivor4dposter.jpg",
  },
  {
    text: "I'm going to make him an offer he can't refuse.",
    movie: "The Godfather",
    year: "1972",
    poster: "https://image.tmdb.org/t/p/w1280/3bhkrj58Vtu7enYsLasi3IOa09z.jpg",
  },
  {
    text: "You can't handle the truth!",
    movie: "A Few Good Men",
    year: "1992",
    poster: "https://image.tmdb.org/t/p/w1280/vvpubke9kJSMFJZBb5QqmfHrGmi.jpg",
  },
  {
    text: "May the Force be with you.",
    movie: "Star Wars",
    year: "1977",
    poster: "https://image.tmdb.org/t/p/w1280/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg",
  },
  {
    text: "To infinity and beyond!",
    movie: "Toy Story",
    year: "1995",
    poster: "https://image.tmdb.org/t/p/w1280/uXDfjJbdP4ijW5hWSBrPl9KcertP.jpg",
  },
  {
    text: "Why so serious?",
    movie: "The Dark Knight",
    year: "2008",
    poster: "https://image.tmdb.org/t/p/w1280/hkBaDkMWbLaf8B1lsWsKX7Ew3Xq.jpg",
  },
  {
    text: "You talking to me?",
    movie: "Taxi Driver",
    year: "1976",
    poster: "https://image.tmdb.org/t/p/w1280/ekstpH614fwDX8DUln1a2Opz0N8.jpg",
  },
  {
    text: "I'll be back.",
    movie: "The Terminator",
    year: "1984",
    poster: "https://image.tmdb.org/t/p/w1280/qvktm0BHcnmDpul4Hz01GIazWPr.jpg",
  },
  {
    text: "There's no place like home.",
    movie: "The Wizard of Oz",
    year: "1939",
    poster: "https://image.tmdb.org/t/p/w1280/gAmX6OYa7nOJVABiDFoFlVKNcSL.jpg",
  },
  {
    text: "Mama always said life was like a box of chocolates.",
    movie: "Forrest Gump",
    year: "1994",
    poster: "https://image.tmdb.org/t/p/w1280/h3051doNiR7vOAB6HO7RbCzUIJj.jpg",
  },
];
 