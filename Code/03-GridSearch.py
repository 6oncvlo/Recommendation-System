def GSearch(ratings, ptest, crossValid):
    
    """
    Parameters:
    
    crossvalid=True or False          # whether or not to perform Cross Validation
    mncmr=37                          # number of common items (movies)
    simMin=0.3                        # minimum similarity between users to be in a neighborhood
    nnei=45                           # number of neighborhoods 
    
    """
    
    mncmrSpace=np.arange(25,45,5)
    nneiSpace=np.arange(30,50,5)
    simMinSpace=np.arange(0.5,0.7,0.1)
    
    #Split original dataset in training and test
    dtrain, dtest = dataSplit(ratings,ptest)
    
    if crossValid==True:
        
        erro={}
        for mncmr in mncmrSpace:
            for nnei in nneiSpace:
                for simMin in simMinSpace:
                    
                    NRMSE=[]
                    NMAE=[]
                    for i in np.sort(dtrain.cvSplit.unique()):

                        dvaldCV=dtrain[dtrain.cvSplit==i]
                        dtrainCV=dtrain[dtrain.cvSplit!=i]

                        # 2.1 Compute rating matrix and similarity matrix
                        rmatrix   = np.array(dtrainCV.set_index(['userId','movId']).rating.unstack())
                        rmatrixMC = rmatrix-np.nanmean(rmatrix, axis=1, keepdims=True)
                        sim=simMatrix(dtrainCV, mncmr)            

                        # 2.234 Collaborative Filtering
                        pred, cfUsers, ncfUsers =collabFiltering(sim, nnei, simMin, rmatrix, rmatrixMC)

                        # Access the prediction quality on the validation set
                        predVald=pred.merge(dvaldCV[['userId','movId','rating']], how='inner', on=['userId','movId'])
                        predVald.ratingPred=np.where(predVald.ratingPred>5,5,
                                                     np.where(predVald.ratingPred<1,1,predVald.ratingPred))

                        erroi={'% of users w/ ratings': np.round(100*len(cfUsers)/(len(cfUsers)+len(ncfUsers)),1),
                               'Normalized RMSE (NRMSE %)': np.round(
                                   100*np.sqrt(np.sum((predVald.ratingPred-predVald.rating)**2)/predVald.shape[0])/5,
                                   1),
                               'Normalized MAE (NMAE %)': np.round(
                                   100*np.sqrt(np.sum(np.abs(predVald.ratingPred-predVald.rating))/predVald.shape[0])/5,
                                   1)
                              }
                        NRMSE.append(erroi['Normalized RMSE (NRMSE %)'])
                        NMAE.append(erroi['Normalized MAE (NMAE %)'])
                        
                   
                    erroP={'Normalized RMSE (NRMSE %)': np.mean(NRMSE),
                           'Normalized MAE (NMAE %)': np.mean(NMAE)}
                    erro[(mncmr, nnei, simMin)]=erroP
        
        return dtrain, dtest, erro
    
    
    
    else:
        
        #Split training dataset in training and validation
        dvald=dtrain[dtrain.cvSplit==np.random.choice(dtrain.cvSplit.unique(), size=1, replace=False)[0] ]
        dtrain=dtrain[dtrain.cvSplit!=dvald.cvSplit.unique()[0]]
        
        # 2.1 Compute rating matrix
        rmatrix   = np.array(dtrain.set_index(['userId','movId']).rating.unstack())
        rmatrixMC = rmatrix-np.nanmean(rmatrix, axis=1, keepdims=True)
        
        erro={}
        for mncmr in mncmrSpace:
            for nnei in nneiSpace:
                for simMin in simMinSpace:
                    
                    # 2.1 Compute similarity matrix
                    sim=simMatrix(dtrain, mncmr)            

                    # 2.234 Collaborative Filtering
                    pred, cfUsers, ncfUsers =collabFiltering(sim, nnei, simMin, rmatrix, rmatrixMC)

                    # Access the prediction quality on the validation set
                    hitRateVald=dvald[dvald.rating>=4].merge(
                        pred[pred.ratingPred>=4.5],
                        how='inner', on=['userId','movId']).sort_values(by=['userId','movId'])
                    
                    predVald=pred.merge(dvald[['userId','movId','rating']], how='inner', on=['userId','movId'])
                    
                    predVald.ratingPred=np.where(predVald.ratingPred>5,5,
                                                 np.where(predVald.ratingPred<1,1,predVald.ratingPred))
                    
                    
                    # Access the prediction quality on the test set
                    hitRateTest=dtest[dtest.rating>=4].merge(
                        pred[pred.ratingPred>=4.5],
                        how='inner', on=['userId','movId']).sort_values(by=['userId','movId'])
                    
                    predTest=pred.merge(dtest[['userId','movId','rating']], how='inner', on=['userId','movId'])
                    
                    predTest.ratingPred=np.where(predTest.ratingPred>5,5,
                                                 np.where(predTest.ratingPred<1,1,predTest.ratingPred))
                    
                    
                    erroP={'% of users w/ ratings': np.round(100*len(cfUsers)/(len(cfUsers)+len(ncfUsers)),1),
                           'dvald (NRMSE %)': np.round(
                               100*np.sqrt(np.sum((predVald.ratingPred-predVald.rating)**2)/predVald.shape[0])/5,
                               1),
                           'dtest (NRMSE %)': np.round(
                               100*np.sqrt(np.sum((predTest.ratingPred-predTest.rating)**2)/predTest.shape[0])/5,
                               1),
                           'dvald (NMAE %)': np.round(
                               100*(np.sum(np.abs(predVald.ratingPred-predVald.rating))/predVald.shape[0])/5,
                               1),
                           'dtest (NMAE %)': np.round(
                               100*(np.sum(np.abs(predTest.ratingPred-predTest.rating))/predTest.shape[0])/5,
                               1),
                           'dvald Hit Rate (%)': np.round(100*hitRateVald.userId.nunique()/len(cfUsers)),
                           'dtest Hit Rate (%)': np.round(100*hitRateTest.userId.nunique()/len(cfUsers))
                          }
                    erro[(mncmr, nnei, simMin)]=erroP
                    print('param done')
                    
        return dtrain, dtest, erro

