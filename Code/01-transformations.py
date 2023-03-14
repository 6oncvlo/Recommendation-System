""" movies dataset transformations """
aux=ratings.groupby('movieId').userId.count().reset_index()
aux.rename(columns={'userId':'nRatings'},inplace=True)
movies=movies.merge(aux, how='left',on='movieId')
movies.nRatings.fillna(0,inplace=True)
movies.nRatings=movies.nRatings.astype(int)
movies.sort_values(by=['nRatings'],ascending=[False],inplace=True)
movies.reset_index(inplace=True)
movies['movId']=movies.index
movies=movies[['movieId','movId','title','nRatings','genres']]



""" movDetails dataset transformations """
movDetails.crew=movDetails.crew.apply(ast.literal_eval)
movDetails.cast=movDetails.cast.apply(ast.literal_eval)
movDetails.tags=movDetails.tags.apply(lambda x: x[1:-1].replace("'","").split(','))
movDetails=movDetails.merge(movies[['movId','title']], how='left', left_on='titleM',right_on='title')
movDetails.drop(columns=['title_y'], inplace=True)
movDetails.rename(columns={'title_x':'title'}, inplace=True)

aux=movies['genres'].str.split('|',expand=True)
aux['title']=movies.title
aux=pd.melt(aux,id_vars='title',value_name='genre')
aux=aux[['title','genre']]
aux.dropna(inplace=True)
aux['val']=1
aux=aux.set_index(['title','genre']).val.unstack().fillna(0)
display
movDetails=movDetails.merge(aux, how='left', left_on='titleM',right_on=aux.index)



""" ratings dataset transformations """
ratings=ratings.merge(movies[['movieId','movId']],how='left', on='movieId')
ratings=ratings[['userId','movId','rating','timestamp']]
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'],unit='s')
ratings.userId=ratings.userId-1
ratings.sort_values(by=['userId','movId'], inplace=True)
ratings.reset_index(drop=True, inplace=True)



""" users dataset transformations """
users.userId=users.userId-1
aux={
    0:  "other or not specified",
    1:  "academic/educator",
    2:  "artist",
    3:  "clerical/admin",
    4:  "college/grad student",
    5:  "customer service",
    6:  "doctor/health care",
    7:  "executive/managerial",
    8:  "farmer",
    9:  "homemaker",
    10:  "K-12 student",
    11:  "lawyer",
    12:  "programmer",
    13:  "retired",
    14:  "sales/marketing",
    15:  "scientist",
    16:  "self-employed",
    17:  "technician/engineer",
    18:  "tradesman/craftsman",
    19:  "unemployed",
    20:  "writer",
}
users=users.merge(pd.DataFrame(aux.items(),columns=['job','jobT']), how='left', on='job')
aux={
    1:  "Under 18",
    18:  "18-24",
    25:  "25-34",
    35:  "35-44",
    45:  "45-49",
    50:  "50-55",
    56:  "56+"
}
users=users.merge(pd.DataFrame(aux.items(),columns=['age','ageT']), how='left', on='age')
users=users[['userId', 'gender','age','ageT','job','jobT','zipcode']]
