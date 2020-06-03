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

    - To run ETL pipeline that cleans data and stores in database, go to data's directory and run:
        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves, go to models' directory and run
        `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to [http://localhost:3001/](http://localhost:3001/)

## Authors

* **Ooi Siang Tan** - *Initial work* - [ooisiang](https://github.com/ooisiang)