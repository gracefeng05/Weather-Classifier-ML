from collections import namedtuple
import itertools
import pandas as pd
import numpy as np
import json
import sys
import os
import math

class NaiveBayes:
    def __init__(self, totalEntries):
        # key = class of weather_descriptions, value = frequency
        self.labels = {}
        # total number of 
        self.totalEntries = totalEntries
        # key = attribute, value = classes of attribute
        self.attributes = {}
        # key = tuple; tuple structure: (attribute, value, label), value = conditional frequency
        self.condFrequencies = {}
        # key = tuple; tuple structure: (attribute, value, label), value = probability
        self.condProbabilities = {}
        self.smoothing_factor = 20

    
    def printLabels(self):
        print("Label Frequencies:")
        for label, frequency in self.labels.items():
            print(f"{label}: {frequency}")

    def printAttributes(self):
        print("Attribute Values:")
        for attribute, values in self.attributes.items():
            print(f"{attribute}: {values}")
    
    def printCondFrequencies(self):
        print("Conditional Frequencies:")
        for key, frequency in self.condFrequencies.items():
            print(f"Frequency of {key[0]} being {key[1]} given weather is {key[2]}: {frequency}")

    def printCondProbabilities(self):
        print("Conditional Probabilities:")
        for key, probability in self.condProbabilities.items():
            for attribute in key:
                if attribute == "precip":
                    print(attribute)
            print(f"COND PROBABILITY: {probability}")

    def getLabelFrequencies(self, y):
        self.labels = y['weather_descriptions'].value_counts()

    def getAttributes(self, x_categorical, x_continuous):
        for column in x_categorical.columns:
            self.attributes[column] = x_categorical[column].unique().tolist()
        for column in x_continuous.columns:
            self.attributes[column] = x_continuous[column].unique().tolist()

    def getCondFrequencies(self, x_categorical, y):
        for label in self.labels.keys():
            for attribute, values in self.attributes.items():
                for value in values:
                    # filter rows where the 'weather_descriptions' column in y matches the current label
                    y_subset = y[y['weather_descriptions'] == label]
                    # filter rows where the current value in x_categorical 'attribute' column matches the current attribute value
                    if attribute != "temperature" and attribute != "wind_speed" and attribute != "wind_degree" and attribute != "heatindex":
                        x_subset = x_categorical[x_categorical[attribute] == value]

                        # conditional count = intersection of y_subset and x_subset plus Laplace smoothing factor
                        condCount = len(y_subset[y_subset.index.isin(x_subset.index)]) + self.smoothing_factor
                        condFrequency = (attribute, value, label)
                        self.condFrequencies[condFrequency] = condCount

    def getCondProbabilities(self, x_categorical, y):
        for label in self.labels.keys():
            for attribute, values in self.attributes.items():
                for value in values:
                    if attribute != "temperature" and attribute != "wind_speed" and attribute != "wind_degree" and attribute != "heatindex":
                        key = (attribute, value, label)
                        condFreq = self.condFrequencies[key]
                        labelFreq = self.labels[label]
                        # apply Laplace smoothing
                        condProb = (condFreq + 1) / (labelFreq + len(self.labels) * self.smoothing_factor)
                        self.condProbabilities[key] = condProb
    
    def getCondProbabilitiesContinuous(self, x_continuous, y):
        epsilon = 1e-6  # small epsilon value to avoid division by zero

        for label in self.labels.keys():
            for attribute in x_continuous.columns:
                # get list of possible temperatures
                values = x_continuous[attribute].unique().tolist() 
                # filter x_continuous based on weather_description
                x_filtered = x_continuous[y['weather_descriptions'] == label]
                mean = x_filtered[attribute].mean()
                std = x_filtered[attribute].std()
                if np.isnan(std) or std == 0: # add epsilon to avoid division by zero
                    std = epsilon
                # iterate through different values (e.g. different temperatures)
                for value in values:
                    condProb = (1 / (np.sqrt(2 * np.pi * std ** 2))) * math.e * np.exp(-((value - mean) ** 2) / (2 * std ** 2))
                    condProb += epsilon  # apply Laplace smoothing
                    if condProb > 1:
                        condProb = 0.5
                    # print(f"Label: {label}, {attribute}: {value}, condProb: {condProb}")
                    key = (attribute, value, label)
                    # print(key)
                    self.condProbabilities[key] = condProb

    def predict(self, test_data):
        # filter test data to include only the last day (day 28)
        day_28_data = test_data.iloc[27]

        max_probability = -1
        predicted_weather = None

        # iterate over all possible weather descriptions
        for label in self.labels.keys():
            # initialize probability for the current weather description
            probability = self.labels[label] / self.totalEntries

            # calculate conditional probability for each attribute value given the current weather description
            temp = 100
            light_precipitation = False
            for attribute, value in day_28_data.items():
                key = (attribute, value, label)
                # print(key)
                # if the key exists in condProbabilities, multiply the probability by the conditional probability
                if key in self.condProbabilities:
                    # rule out non-precipitation weather_descriptions when there is even the slightest bit of precipitation present
                    if value == "Light precipitation" or value == "Moderate precipitation" or value == "Heavy precipitation":
                        if label == "Sunny" or label == "Clear" or label == "Cloudy" or label == "Overcast" or label == "Partly cloudy":
                            probability *= 0
                    else:
                        probability *= self.condProbabilities[key]
            # update maximum probability and predicted weather description if necessary
            if probability > max_probability:
                max_probability = probability
                predicted_weather = label

        return predicted_weather

def parse_data(file_path):
    df = pd.read_excel(file_path)

    x_categorical = ['time', 'weather_descriptions', 'precip', 'visibility', 'wd1', 'wd2', 'wd3']
    df['wd1'] = df['weather_descriptions'].shift(1)
    df['wd2'] = df['weather_descriptions'].shift(2)
    df['wd3'] = df['weather_descriptions'].shift(3)
    y = ['weather_descriptions']
    x_continuous = ['temperature', 'wind_speed', 'wind_degree', 'heatindex']
    
    return df[x_categorical], df[y], df[x_continuous]

def main():
    training_file = sys.argv[1]
    test_folder = sys.argv[2]
    ground_truth_file = sys.argv[3]

    x_categorical, y, x_continuous = parse_data(training_file)
    length = len(y) - 1

    classifier = NaiveBayes(length)

    classifier.getLabelFrequencies(y)

    classifier.getAttributes(x_categorical, x_continuous)

    y = y.shift(-1)
    classifier.getCondFrequencies(x_categorical, y)

    classifier.getCondProbabilities(x_categorical, y)

    classifier.getCondProbabilitiesContinuous(x_continuous, y)

    predictions = []
    for i in range(1, 1001):
        test_data = pd.read_excel(f"./{test_folder}/test{i}.xlsx")
        test_data['wd1'] = test_data['weather_descriptions'].shift(1)
        test_data['wd2'] = test_data['weather_descriptions'].shift(2)
        test_data['wd3'] = test_data['weather_descriptions'].shift(3)
        columns = ['time', 'temperature', 'weather_descriptions', 'precip', 'visibility', 'heatindex', 'wind_speed', 'wind_degree', 'wd1', 'wd2', 'wd3']
        prediction = classifier.predict(test_data[columns])
        predictions.append(prediction)
    
    # Load ground truth data
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)

    correct_predictions = 0
    for prediction, truth in zip(predictions, ground_truth):
        if prediction == truth:
            correct_predictions += 1

    accuracy = correct_predictions / len(predictions)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
