def dataSplit(ratings,ptest):
    
    """
    Split ratings dataset in 3 smaller datasets:
        - Training
        - Validation
        - Test
    Note that
        - validation dataset will be the same size as test dataset
        - Hold Out or Cross validation 
        - validation dataset is within training dataset   """
    
    dtrain,dtest=train_test_split(ratings,
                       test_size=ptest, random_state=123, shuffle=True)
    dtest.sort_values(by=['userId','movId'],inplace=True)
    dtrain.sort_values(by=['userId','movId'],inplace=True)
    
    k=10
    cv=np.repeat( range(1,k+1), np.floor(dtrain.shape[0]/k) )
    cv=np.concatenate((cv,np.random.choice(range(1,k+1), size=dtrain.shape[0]-cv.shape[0], replace=False)))
    np.random.shuffle(cv)
    dtrain['cvSplit']=cv
    
    aux=ratings.groupby('movId').userId.count().reset_index().merge(
        dtrain.groupby('movId').userId.count().reset_index(), how='left', on='movId')
    print('Highest % of ratings used from a certain movie in the test dataset is :',
          np.round(100-(aux.userId_y/aux.userId_x*100).min(),1),'%')
    aux=ratings.groupby('userId').movId.count().reset_index().merge(
        dtrain.groupby('userId').movId.count().reset_index(), how='left', on='userId')
    print('Highest % of ratings used from a certain user in the test dataset is  :',
          np.round(100-(aux.movId_y/aux.movId_x*100).min(),1),'%\n')
    
    return dtrain,dtest


def simMatrix(ratings, mncmr):
    
    start=time.time()
    #compute similarity matrix using surprise package
    aux=ratings.copy()
    aux['userRat'] = list(zip(aux.userId, aux.rating))
    simUsers={}

    for m in np.sort(aux.movId.unique()):
        simUsers[m]=list(aux[aux.movId==m].userRat)
        
    simUsers=surprise.similarities.msd(aux.userId.nunique(), simUsers, min_support=mncmr)
    np.fill_diagonal(simUsers, 0)
        
    print('Similarity matrix computation time (secs): ', np.round(time.time()-start,1))
    return simUsers
    
    
def neighborhood(sim, nnei, simMin):
    
    """
    2.2 Compute neighborhoods & exclude non correlated neighbors.
    Compute users where CF will be applied. """
    nu=sim.shape[0]
    start=time.time()
    viz=[[np.sort(np.argpartition(sim[u], -nnei)[-nnei:]) 
         ,sim[u][np.sort(np.argpartition(sim[u], -nnei)[-nnei:])]]
                  
         for u in range(nu)]
    
    #exclude non correlated neighbors
    viz=[viz[u][0][np.where(viz[u][1]>=simMin)]
         for u in range(nu)]
    
    cfUsers=np.where(np.array([len(v) for v in viz])>=nnei)[0]
    ncfUsers=list(set(range(nu))-set(cfUsers))
    
    print('Neighborhood computation time (secs)     : ',np.round(time.time()-start,1))
    
    return viz, cfUsers, ncfUsers
    
def itemsToPredict(rmatrix, viz, cfUsers):
    
    start=time.time()
    movPred=[np.sort(list(set(np.where((~np.isnan(rmatrix[viz[u],:])).sum(axis=0)!=0)[-1]) -
                           set(np.where(~np.isnan(rmatrix[u,:]))[-1]) 
                          ))
              for u in cfUsers]

    print('Movies to predict computation time (secs): ',np.round(time.time()-start,1))
    
    return movPred


def itemsPredictions(rmatrixMC, cfUsers, movPred, viz, sim, rmatrix):
    
    start=time.time()
    pred=[]

    rmatrixMC0=rmatrixMC.copy()
    rmatrixMC0[np.isnan(rmatrixMC0)] = 0
    for u,v in zip(cfUsers, range(len(movPred))):

        rmatrixMC0_NeiuPredm=rmatrixMC0[viz[u]][:,movPred[v]]
        simUNeiu=sim[u,viz[u]]

        predU=np.matmul(simUNeiu,
                        rmatrixMC0_NeiuPredm)/np.matmul(np.abs(simUNeiu),
                                                        ~np.isnan(rmatrix[viz[u]][:,movPred[v]]))
        
        pred.append(predU)
        
    pred=[p+np.nanmean(rmatrix[u]) for p,u in zip(pred,cfUsers)]
    print('Prediction computation time (secs)       : ', np.round(time.time()-start,1),'\n')
    
    return pred

def collabFiltering(sim, nnei, simMin, rmatrix, rmatrixMC):          

    # 2.2 Compute neighborhoods and cfUsers
    viz, cfUsers, ncfUsers = neighborhood(sim, nnei, simMin)

    # 2.3 Compute movies to predict
    movPred=itemsToPredict(rmatrix, viz, cfUsers)

    # 2.4 Predict ratings and get top nri recommendations
    pred =itemsPredictions(rmatrixMC, cfUsers, movPred, viz, sim, rmatrix)
    
    return pd.DataFrame({'userId':np.repeat(cfUsers,[len(mo) for mo in movPred]),
                         'movId':np.hstack(movPred), 'ratingPred':np.hstack(pred)}), cfUsers, ncfUsers
    

