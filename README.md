# README

The following report details the Naive Bayes weather prediction algorithm I developed for a project in a machine learning model competition. When evaluated against the test cases, my algorithm achieved the highest accuracy in a submission pool of 100 competitors, earning the top ranking.

The code can be found here: [Naive Bayes Weather Classifier](https://github.com/gracefeng05/Weather-Classifier-ML)

## Run Instructions
1. Clone the repository.
2. Run `python3 WeatherClassifier.py training.xlsx tests ground_truth.json`

## Tips
- `training.xlsx` is an excel sheet containing ~7000 rows of weather data where each row corresponds to a single day.
    - columns correspond to weather features, e.g. cloudy, precipitation, wind speed, etc.
- `tests` is a folder containing 1000 excel files, each containing 28 rows of data to represent a month of weather data. The classifier aims to predict the 29th day's weather for each excel file.
- `ground_truth.json` contains the actual weather for `tests`. This is used to calculate the accuracy of the model.

## Architecture

| Attribute           | time       | weather_description | precip    | visibility | temperature |
|---------------------|------------|---------------------|-----------|------------|-------------|
| **Data Format**      | Categorical| Categorical         | Categorical| Categorical| Continuous  |
| **Naive Bayes Variant** | Categorical| Categorical         | Categorical| Categorical| Gaussian    |

| Attribute           | wind_speed | wind_degree | heatindex | wd1        | wd2        | wd3        |
|---------------------|------------|-------------|-----------|------------|------------|------------|
| **Data Format**      | Continuous | Continuous  | Continuous| Categorical| Categorical| Categorical|
| **Naive Bayes Variant** | Gaussian  | Gaussian    | Gaussian  | Categorical| Categorical| Categorical|

*Figure 1: Hybrid Naive Bayes Approach*

## Class Structure

I built a single class, called `NaiveBayes`, with five class variables:

```python
self.labels = {}
# A dictionary of the possible ‘weather_descriptions’ labels;
# key = label, value = frequency in training dataset.

self.totalEntries = totalEntries
# The total number of rows in the training dataset
# (used to calculate frequency ratios).

self.attributes = {}
# A dictionary of the training attributes and their different values;
# key = attribute, value = possible values of this attribute
# (e.g. 0, 600, 1200, 1800 for ‘time’).

self.condFrequencies = {}
# A dictionary of the conditional frequencies for each attribute given each label;
# key = (attribute, value, label), value = conditional frequency.
# These conditional frequencies are used to calculate the conditional probabilities of categorical attributes.

self.condProbabilities = {}
# A dictionary of the conditional probabilities for each attribute given each label; key = (attribute, value, label), value = conditional probability.

self.smoothing_factor = 20
# A smoothing factor I applied to my formulas for calculating conditional frequency and conditional probability.
# condCount = len(y_subset[y_subset.index.isin(x_subset.index)]) + self.smoothing_factor
```
Within the NaiveBayes class, I defined six functions to populate these dictionaries:
1. ```getLabelFrequencies()```
2. ```getAttributes()```
3. ```getCondFrequencies()```
4. ```getCondProbabilities()``` - Fill self.condProbabilities for categorical attributes.
5. ```getCondProbabilitiesContinuous()``` - Fill self.condProbabilities for continuous attributes with Gaussian PDF formula
6. ```predict()``` - Iterates through the 28th day’s attributes and generates keys (attribute, value, label) to search for its corresponding conditional probability in self.condProbabilities.

## Pre-processing
The following screenshot illustrates the attributes that I trained my model on (‘wd1’, ‘wd2’, ‘wd3’ are the 3 weather descriptions that came before the weather description in question). A single instance (day) in the dataset is represented as a row in a pandas dataframe object.

<img width="823" alt="Screenshot 2024-09-26 at 2 07 45 PM" src="https://github.com/user-attachments/assets/5cfe8cbe-1f61-4826-955c-b6e5e546f8f2">

x_categorical is the dataframe where each column is one of my selected categorical attributes. Same concept for x_continuous. I calculate the conditional probabilities differently using my functions getCondProbabilities() and getCondProbabilitiesContinuous().

## Model Building
<img width="1132" alt="Screenshot 2024-09-26 at 2 12 20 PM" src="https://github.com/user-attachments/assets/92623aef-a2c4-4331-9f65-a4b5c0079804">
<img width="1133" alt="Screenshot 2024-09-26 at 2 12 37 PM" src="https://github.com/user-attachments/assets/7e13b3c1-6d5b-493c-ae39-89ae20cb623f">

To train my classifier, I wrote separate functions (see above) to calculate and pre-save the conditional probabilities depending on the kind of attribute in question (categorical or continuous). I used the formulas provided in the lecture slides. In my predict() function, I calculated the conditional probabilities of each possible weather label given the attributes in the 28th day’s row by retrieving the necessary conditional probabilities from my self.condProbability dictionary and multiplying them together.

When looking through the training.xlsx file, I noticed that the presence of any precipitation ruled out any possibility of the weather description being “Sunny”, “Clear”, “Cloudy”, “Partly cloudy”, or “Overcast”. After printing my incorrect predictions on the given test dataset, I realized that most of my missed predictions were false “Sunny” and “Clear” predictions for days which had precipitation. As such, I coded a check in my prediction() function to set the probability of the prediction being “Sunny”, “Clear”, “Cloudy”, “Partly cloudy”, or “Overcast” to 0 if the 28th day’s ‘precip’ attribute was either “Light precipitation”, “Moderate precipitation”, or “Heavy precipitation”. This check bumped my accuracy from 0.669 to 0.786

## Results
Accuracy: 0.786
Runtime: 18.37 seconds
I attribute my improvements in accuracy to including the previous three days of weather_descriptions in my training dataset and filtering out “Sunny”, “Clear”, “Cloudy”, “Partly cloudy”, or “Overcast” predictions for days which were preceded with any sort of precipitation.

## Challenges
I struggled with selecting the optimal combination of attributes to train on. After arbitrarily testing random combinations, I realized that my approach was uninformed and entirely random. To gain more insight into the training dataset, I manually looked into the training.xlsx file to manually pinpoint patterns.



