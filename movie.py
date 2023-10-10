class Movie:
    def __init__(self, id:int, title:str, genres:list):
        self.movieId = id
        self.title = title
        self.genres = genres

        self.ratings = dict()

    
    def add_rating(self, userID:int, rating:float):
        self.ratings[userID] = rating

    
    def __str__(self):
        text = f'Movie ID: {self.movieId}\nTitle: {self.title}\nGenres: '
        for genre in self.genres:
            text += genre + "\t"
        text += "\n\nRatings:\n"
        for userID in self.ratings:
            text += f"User ID: {userID}\tRating: {self.ratings.get(userID)}"