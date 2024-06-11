# Weather-Classifier-ML
Naïve Bayes Classifier built for UCSB's CMPSC 165A S24 offering.

# Run Instructions
1. Clone the repository.
2. Run `python3 WeatherClassifier.py training.xlsx tests ground_truth.json`

# Explanation
- `training.xlsx` is an excel sheet containing ~7000 rows of weather data where each row corresponds to a single day.
    - columns correspond to weather features, e.g. cloudy, precipitation, wind speed, etc.
- `tests` is a folder containing 1000 excel files, each containing 28 rows of data to represent a month of weather data. The classifier aims to predict the 29th day's weather for each excel file.
- `ground_truth.json` contains the actual weather for `tests`. This is used to calculate the accuracy of the model.