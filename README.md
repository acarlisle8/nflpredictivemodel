# nflpredictivemodel

**Predicting NFL Touchdowns using Ridge Regression**

This repository contains code for a model that predicts NFL touchdowns based on various statistics such as yards, passes, yards gained, explosive plays, and third down conversions. The model uses Ridge Regression to make predictions.

**Data**

The data used came from https://github.com/nflverse/nflverse-data/releases/tag/pbp

**Model** 

The Ridge Model was trained on 10 years of data from 2010 to 2020. The alpha was set to 4.0. 

**Results** 

The model achieved an RMSE of 8.223 and an R-squared score of 0.655. The RMSE indicates that, on average, the model's predictions are off by about 8 touchdowns. The R-squared score indicates that the model explains about 65% of the variation in the number of touchdowns. The results could definitely be improved on. 

**Next Steps**

The next thing I would want to do with this project is to improve on the accuracy of this model. I would want to use a tool like autosklearn or Watson AutoAI to test what the best model would be. 
