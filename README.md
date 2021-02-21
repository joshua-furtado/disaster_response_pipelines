# Disaster Response Pipelines

## Description

**disaster_response_pipelines** is a machine learning pipeline to categorize messages sent during disaster events. This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Dependencies

disaster_response_pipelines requires:

- NumPy
- pandas
- SQLAlchemy
- scikit-learn
- Natural Language Toolkit

## Folder Descriptions

1. [ETL_pipeline](https://github.com/joshua-furtado/disaster_response_pipelines/tree/main/ETL_pipeline)

	- Contains Jupyter notebook for the Extract, Transform, and Load process. 
	- Reads the datasets, merges them, cleans the data, and then stores it in a SQLite database.
	- Cleaned code is stored in the final ETL script, [process_data.py](https://github.com/joshua-furtado/disaster_response_pipelines/blob/main/web_app/data/process_data.py).

2. [ML_pipeline](https://github.com/joshua-furtado/disaster_response_pipelines/tree/main/ML_pipeline)

	- Contains Jupyter notebook for machine learning pipeline.
	- Uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification).
	- Exports model to a pickle file.
	- Final machine learning code is stored in [train_classifier.py](https://github.com/joshua-furtado/disaster_response_pipelines/blob/main/web_app/models/train_classifier.py).

3. [web_app](https://github.com/joshua-furtado/disaster_response_pipelines/tree/main/web_app)

	- Flask web app to display results.

4. [images](https://github.com/joshua-furtado/disaster_response_pipelines/tree/main/images)

	- Folder containing screenshots of the web app.

## How to run the web app

1. Run the following commands in the web_app's directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory of the web_app to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

	- Try http://localhost:3001 if that does not work.

## How to use the web app

1. Enter message.
![](https://github.com/joshua-furtado/disaster_response_pipelines/blob/main/images/input.png?raw=true)

2. Hit the `Classify Message` button to use model to predict message categories.
![](https://github.com/joshua-furtado/disaster_response_pipelines/blob/main/images/classify.png?raw=true)

3. Scroll down to see the resulting categories.
![](https://github.com/joshua-furtado/disaster_response_pipelines/blob/main/images/result.png?raw=true)


## Authors

**Joshua Furtado**