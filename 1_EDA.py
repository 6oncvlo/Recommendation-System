"""
EDA (Exploratory Data Analysis)

- how many users? how many movies?
- how many rated movies? how many tagged movies?
- # rated movies per user
- # users per rated movies
- ratings: median, mean, max, min
- unique values of each column, relative and absolute frequency
- standard deviation, top 5 & bottom-5 regarding the frequency

"""


print('DATASET users\n')
print(users.info())
display(users)

print('\n','*'*50,'\n')
print('DATASET movies\n')
print(movies.info())
display(movies)

print('\n','*'*50,'\n')
print('DATASET ratings\n')
print(ratings.info())
display(ratings)


print('EDA on users dataset\n')
print('num of users            : ',users.userId.nunique())
print('userId indexes          : ',users.userId.min() ,'-',users.userId.max())
print('users age               : ',users.age.min() ,'-',users.age.max())
print('users gender            : ',users.gender.unique())
print('num of user jobs        : ',users.job.nunique())
print('num of user location    : ',users.zipcode.nunique(),'\n'*2,'*'*70)

print('\nEDA on movies dataset\n')
print('num of movies           : ',movies.movieId.nunique())
print('movieID indexes         : ',movies.movieId.min() ,'-',movies.movieId.max())
print('movie release year      : ',
      movies.title.str.slice(-5,-1).astype(int).min() ,'-',movies.title.str.slice(-5,-1).astype(int).max(),
     '\n'*2,'*'*70)

print('\nEDA on ratings dataset\n')
print('num of users that rated : ',ratings.userId.nunique())
print('num of rated movies     : ',ratings.movieId.nunique())
print('rating values           : ',np.sort(ratings.rating.unique()) )

aux=ratings.groupby(by=['userId','movieId']).rating.count().reset_index().sort_values('rating',ascending=False)
print('Number of users that rated the same movie multiple times: ',aux[aux.rating>1].userId.nunique(),'users')
