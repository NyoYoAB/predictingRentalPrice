import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from datetime import datetime

# Define a function for scaling Train Data with minMaxScaling:
def min_max_scaling(df, cols):
    for column in cols:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    return df

# Define a function for calculating minMaxScaling for input users:
def min_max_scaling_value(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Import Cleaned Dataset (Clean UP already done in previous section)
cleaned_data = pd.read_csv("raw_house_data_cleaned.csv")

# Renaming "price" column
newColumns = {'price': "rental_price"}
cleaned_data.rename(columns=newColumns, inplace=True)

# Setting Monthly Renting Price according to 0.4% radio as discussed in GenG class:
cleaned_data["rental_price"] = cleaned_data["rental_price"] * 0.004

# New feature engineering:
cleaned_data["year_sqrt"] = cleaned_data["year_built"] * cleaned_data["sqrt_ft"]

# Get the current year
current_year = datetime.now().year

# Create a new column 'house_age' that calculates the age of the house
cleaned_data['house_age'] = current_year - cleaned_data['year_built']
cleaned_data['inverse_house_age'] = 1 / cleaned_data['house_age'].replace(0, 1)

# Feature selections:
selected_columns = ['rental_price', "sqrt_ft", "year_sqrt" , "lot_acres", "latitude" , "longitude" , 'taxes', "inverse_house_age" , "bedrooms", "bathrooms" ,"fireplaces", "garage"]
rentalHouse_df = cleaned_data[selected_columns]

# Scaling using the MinMax
scaled_columns = ["sqrt_ft", "taxes", "lot_acres", "year_sqrt" , "latitude", "longitude", 'bedrooms', 'bathrooms', "fireplaces", "garage"]
rentalHouse_df = min_max_scaling(rentalHouse_df, scaled_columns)
min_vals = rentalHouse_df[scaled_columns].min()
max_vals = rentalHouse_df[scaled_columns].max()

# Define the KNN Regressor class
class KNNRegressor:

    def __init__(self, k=5):
        self.k = k  # Number of neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = [] 
        
        for x_test in X_test:
            # Calculate Euclidean distances between x_test and all points in X_train
            distances = np.sqrt(np.sum((self.X_train - x_test) ** 2, axis=1))
            
            # Find the indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get the target values of the k nearest neighbors
            k_nearest_targets = self.y_train[k_indices]
            
            # Predict the target as the mean of the nearest neighbors' targets
            y_pred.append(np.mean(k_nearest_targets))
        
        return np.array(y_pred)

# Split Data into train and test datasets (allocating 80% for Trainning):
def train_test_random_spliter(df, train_size, seed_value=42):
    np.random.seed(seed_value)
    shuffled_indices = np.random.permutation(len(df))
    split_index = int(train_size * len(shuffled_indices))
    df_train = df.iloc[shuffled_indices[:split_index]]
    df_test = df.iloc[shuffled_indices[split_index:]]
    return df_train, df_test

df_train, df_test = train_test_random_spliter(rentalHouse_df, train_size=0.8)

# Separate the train and test datasets into features and predictor
X_train = df_train[selected_columns[1:]]
y_train = df_train['rental_price']

X_test = df_test[selected_columns[1:]]
y_test = df_test['rental_price']

# Initialize and train the KNN model with k=5
knn_regressor = KNNRegressor(k=5)
knn_regressor.fit(X_train.values, y_train.values)

# Creating the Tkinter window
root = tk.Tk()
root.title("House Rental Price Predictor")
root.geometry("500x600")
root.configure(bg='#f4f4f4')

# Function to calculate background features
def calculate_background_features(year_built, sqrt_ft, house_age):
    year_sqrt = year_built * sqrt_ft
    inverse_house_age = 1 / house_age if house_age != 0 else 0
    return year_sqrt, inverse_house_age

# Function to predict the rental price
def predict_rental_price():
    try:
        sqrt_ft = float(entry_sqrt_ft.get())
        year_built = int(entry_year_built.get())
        lot_acres = float(entry_lot_acres.get())
        latitude = float(entry_latitude.get())
        longitude = float(entry_longitude.get())
        taxes = float(entry_taxes.get())
        bedrooms = int(entry_bedrooms.get())
        bathrooms = int(entry_bathrooms.get())
        fireplaces = int(entry_fireplaces.get())
        garage = int(entry_garage.get())

        current_year = datetime.now().year
        house_age = current_year - year_built
        year_sqrt, inverse_house_age = calculate_background_features(year_built, sqrt_ft, house_age)

        # Scale the input features:
        sqrt_ft = min_max_scaling_value(sqrt_ft, min_vals["sqrt_ft"], max_vals["sqrt_ft"])
        year_sqrt = min_max_scaling_value(year_sqrt, min_vals["year_sqrt"], max_vals["year_sqrt"])
        lot_acres = min_max_scaling_value(lot_acres, min_vals["lot_acres"], max_vals["lot_acres"])
        latitude = min_max_scaling_value(latitude, min_vals["latitude"], max_vals["latitude"])
        longitude = min_max_scaling_value(longitude, min_vals["longitude"], max_vals["longitude"])
        taxes = min_max_scaling_value(taxes, min_vals["taxes"], max_vals["taxes"])
        bedrooms = min_max_scaling_value(bedrooms, min_vals["bedrooms"], max_vals["bedrooms"])
        bathrooms = min_max_scaling_value(bathrooms, min_vals["bathrooms"], max_vals["bathrooms"])
        fireplaces = min_max_scaling_value(fireplaces, min_vals["fireplaces"], max_vals["fireplaces"])
        garage =      min_max_scaling_value(garage, min_vals["garage"], max_vals["garage"])

        # Combine all features into a single array
        features = np.array([
            sqrt_ft, year_sqrt, lot_acres, latitude, 
            longitude, taxes, inverse_house_age, bedrooms, bathrooms, fireplaces, garage
        ])

        # Reshape input for prediction
        features = features.reshape(1, -1)

        # Make the prediction using the KNN model
        predicted_price = knn_regressor.predict(features)[0]

        # Display the prediction result
        messagebox.showinfo("Prediction", f"The predicted rental price is: ${predicted_price:.2f}")
    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# Labels and Entry fields for required features
labels = ["Square Footage (sqrt_ft)", "Year Built", "Lot Acres", 
          "Latitude", "Longitude", "Taxes", "Bedrooms", "Bathrooms", "Fireplaces", "Garage"]

entries = []

form_frame = tk.Frame(root, bg='#f4f4f4')
form_frame.place(relx=0.5, rely=0.5, anchor="center")

for i, label_text in enumerate(labels):
    label = tk.Label(form_frame, text=label_text, font=("Arial", 12), bg='#f4f4f4')
    label.grid(row=i, column=0, padx=10, pady=5, sticky="e")
    
    entry = tk.Entry(form_frame, font=("Arial", 12))
    entry.grid(row=i, column=1, padx=10, pady=5, ipadx=5, ipady=5)
    
    entries.append(entry)

entry_sqrt_ft, entry_year_built, entry_lot_acres, entry_latitude, \
entry_longitude, entry_taxes, entry_bedrooms, entry_bathrooms, entry_fireplaces, \
entry_garage = entries

# Create the "PREDICT THE RENTAL PRICE" button
predict_button = tk.Button(form_frame, text="PREDICT THE RENTAL PRICE", font=("Arial", 14), bg='#3498db', fg='white', 
                           command=predict_rental_price)
predict_button.grid(row=len(labels), column=0, columnspan=2, pady=20, ipadx=10, ipady=10)

# Run the Tkinter event loop
<<<<<<< HEAD:tKinter_app.py
root.mainloop()
=======
root.mainloop()

>>>>>>> e3187ff0b9aaea0204b4aa8d297525cf6846d520:bugCorrection.py
