from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def treedepth (inicial, final,X_train,y_train,X_test,y_test):
    """
    recives two integers numbers to acomodate the range
    and the variables of the train,test,split of our data
    it fits the variables to the RandomForestRegressor model
    returns a dataframe that shows the mean_squared_error of the X_train and X_test variables
    """
    results = []

    for depth in range(inicial,final):
        model = RandomForestRegressor(max_depth=depth)
        model.fit(X_train,y_train)

        result = {
            "model": model,
            "depth": depth,
            "train_error": mean_squared_error(y_train, model.predict(X_train)).round(2),
            "test_error": mean_squared_error(y_test, model.predict(X_test)).round(2)
        }
        results.append(result)
    
    return pd.DataFrame(results)