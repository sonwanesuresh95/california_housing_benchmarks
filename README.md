# california_housing_benchmarks
observing behaviour of learning algorithms on regression task with feature scaling
## Info
This is a research project on regression. <br>
Goal of this project is to observe how traditional machine learning algorithms adapt real world data.<br><br>
The example which was followed for studies is california housing dataset <br>https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html<br><br>
Total 10 machine learning models were trained on california housing dataset and benchmarked.<br>
Performance metrics used for benchmark are mean squared error, training time and inference time
<br><br>
## Usage
For installing requirements, do<br>
<code>
$cd california_housing_benchmarks
</code><br>
<code>
$pip install requirements.txt
</code><br><br>
For training your own machine learning models on California Housing, do<br>
<code>
$python train.py<br>
</code><br><br>

## Review
The train.py script trains following models and generates benchmarks<br>

| models      | mses | training times     | 
| :---        |    :----:   |          ---: | 
| LinearRegression      | 0.3543       | 0.0136   | 
| Ridge   | 0.3543        |    0.006   | 
|Lasso |0.3629 | 0.0229| 
| SVR | 0.2338| 11.6984| 
| KernelRidge |0.3585 | 41.7658| 
| GaussianProcessRegressor |828.3268 |84.7025 | 
| DecisionTreeRegressor |0.3621 |0.2184 | 
| RandomForestRegressor |0.1729 |25.9781 | 
| <b>AdaBoostRegressor |<b>0.1684 |<b>34.1281 | 
| GradientBoostingRegressor | 0.1963|7.9679 | 


## Bonus! - Inference API slash Web App
You can run the API directly into your browser to predict housing housing price.<br>
Go find sample example features to predict on homepage.<br>
To run the web app, do<br>
<code>
$pip install requirements.txt
</code><br>
<code>
$python app.py
</code><br><br>
Open [http://localhost:5000/](http://localhost:5000/) on your local machine.<br><br>
API in action<br>
![image](https://github.com/sonwanesuresh95/california_housing_benchmarks/blob/master/root.png "homepage")

![image](https://github.com/sonwanesuresh95/california_housing_benchmarks/blob/master/predict.png "predict")
