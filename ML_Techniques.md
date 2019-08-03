# KNN Improvement Techniques

- Adjust parameters of analysis
  - maybe K is not right, or maybe out prediction point was just a little bit off
- add more features to explain the analysis
- change the prediction point
- Accept that maybe there isn't a strong correlation between our prediction and the data


## Measuring the accuracy of the algorithm

- gather a whole bunch of data, and split that data into two sets; one for training and one for testing our predictions
- for each record in the test data run KNN algo using the training data

## Linear Regression

- y = mx + b
  - for the purpose of the housing prices m * lot_size + b

- Goal of linear regression is to determine an equation that relates an independent variable to some dependent variable we are trying to solve for

- Mean Squared Error - 1/n * n sum i = 1 (guess_i - actual_i) ** 2
  - Must run MSE twice with two differnt guesses to produce any meaningful data from it
  - the lower the value produced by MSE the better the guess is

- When MSE is as low as it possibly can be ( unlikely it will ever be zero ) m and b should be as correct as we can get them

## Derivatives

- taking the derivative of ```javascript y = x ** 2 + 5``` give an equation ```javascript dy / dx = 2x```
