# Stock Prediction Using SimpleRNN
RNN stands for Recurrent Neural Networks. RNN is used when we have sequential data.
`SimpleRNN` is a Function that is provided by `Tensorflow`, `Keras`. It is a fully-connected RNN where the output is fed back as input.
RNN is used for **Time Series Forecasting**.

## Project objective
The objective of the Project is to execute an RNN model inorder to **predict the performance of Tesla Stocks**.

<img src="https://github.com/navi1910/StockPrediction-SimpleRNN-TimeSeriesForecasting/blob/master/tesla_logo.png" width=30% height=30%>

## Methods
* Deep Learning Neural Networks
* Simple RNN by `Tensorflow`
* Visualizations
* Feature Engineering
* Data Normalization
* Saving the Model using `os`

## Technologies used
* Python
* Pandas
* Tensorflow
* Keras
* Scikit-learn
* Matplotlib

## Project Description
The data was obtained from **Kaggle**. The project aims to predict the movement of Tesla Stock. 
Recurrent neural networks have the feature of taking the **output of the same artificial neuron as the input again**. The Neural Network takes the first 50 observations and then uses regression to predict the 51st observation. 
The process continues for predicting the 52nd observation and so on. This method is also called **Autoregression**.

The output-input loop feature of RNN model and Autoregression process used help in predicting the future values.

## Procedure
* The required python modules are imported
* The data is imported as a **Data Frame using Pandas**.
* The data is separated in to **Traing set and Validation set**.
* The data from both Training as well as Validation set is Reshaped.
* The Data is Normalised using `MinMaxScaler` from Scikitlearn.
* **Feature Engineering** is done according to the model's needs. The data is segregated in order use 50 observations to predict 51st observation.
* The data is Reshaped again.

### Model Building
* `Tensorflow`, `Keras` is used for building the Neural Network.
* `SimpleRNN`, `Dropout`, `Dense` layers are added to `Sequential`.
* The model is `compiled` using
    + `'adam'` optimizer
    - `'mean_squared_error'` loss
    + `['accuracy']` metrics

* Model is then summarized

<img src="https://github.com/navi1910/StockPrediction-SimpleRNN-TimeSeriesForecasting/blob/master/model_summary.png" width=50% height=50%>

* The model is fit to Training data.
* Loss and Accuracy is plotted.
* X_train is predicted using `predict` then denormalized using `inverse_transform`.
* The predicted values are plotted along side actual values
![Train Plot](https://github.com/navi1910/StockPrediction-SimpleRNN-TimeSeriesForecasting/blob/master/train_prediction.png "Train Plot")
* The process is repeated for validation set and the results are plotted.
![Validation Plot](https://github.com/navi1910/StockPrediction-SimpleRNN-TimeSeriesForecasting/blob/master/validation_prediction.png "Validation Plot")

## Saving the Model
The model is saved to a h5 file using `model.save` and python `os` into a directory called `model`. The model can be loaded using `load_model`.

## Contact
[Naveen's LinkedIn](https://www.linkedin.com/in/naveen-a-902a671b3/)

## Acknowledgements
[Kaggle Notebook](https://www.kaggle.com/code/ozkanozturk/stock-price-prediction-by-simple-rnn-and-lstm)
