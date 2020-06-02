# Disaster Response Pipeline Project
This project aims to build an ETL and ML pipeline that categorizes disaster messages into 36 different categories. So that each disaster message can be distributed to the relevant department automatically.


## Getting Started

To get a copy of this project in the local project, please execute:
```
git clone https://github.com/ooisiang/disaser-response.git 
```

## Prerequisites

A requirements.txt file is provided in this repo to prepare the environment needed to run the code.
Please install the packages by executing:
```
pip install -r requirements.txt
```

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to [http://localhost:3001/](http://localhost:3001/)

## Authors

* **Ooi Siang Tan** - *Initial work* - [ooisiang](https://github.com/ooisiang)