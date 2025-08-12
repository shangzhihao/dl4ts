# a simple tool for time series forecasting using deep learning model
This is a simple tool for time series forecasting using deep learning model. It is written in Python and uses the pytorch library. The tool can be used for univariate time sereis data now.

The data format should be as follows: two columns in a csv file, the first column is the training data, the second column is the validation data.

- Docker is required sice it trains the model in a docker container.
- It does not support GPU training yet.
- This is a FastAPI based web app. You can run it with the following command:
``` bash
uvicorn main:app --reload
```
![alt text](ui.png)