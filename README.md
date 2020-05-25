# Disaster Response Pipeline Project

## Getting Started

To get a copy of this project in the local project, please execute:
```
git clone https://github.com/ooisiang/disaser-response.git 
```

## Prerequisites

Project is still under work. _requirements.txt_ file will be provided soon.

## Instructions:
_**Web app is currently not working because it is still under work**_
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Authors

* **Ooi Siang Tan** - *Initial work* - [ooisiang](https://github.com/ooisiang)