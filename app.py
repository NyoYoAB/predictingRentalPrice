import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from datetime import datetime

# Define a function for calculating minMaxScaling:
def min_max_scaling(df, cols):
    for column in cols:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    return df

# Define a function for randomly splitting data into train and test:
def train_test_random_spliter(df, train_size, seed_value=42):

    # random value for split
    np.random.seed(seed_value)
    shuffled_indices = np.random.permutation(len(df))

    # Define the split index
    split_index = int(train_size * len(shuffled_indices))

    # Create train and test sets by splitting the shuffled DataFrame
    df_train = df.iloc[shuffled_indices[:split_index]]
    df_test = df.iloc[shuffled_indices[split_index:]]

    return df_train, df_test

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
scaled_columns = ["sqrt_ft","taxes","lot_acres", "year_sqrt" , "latitude", "longitude", 'bedrooms', 'bathrooms', "fireplaces", "garage"]
rentalHouse_df = min_max_scaling(rentalHouse_df, scaled_columns)
rentalHouse_df.head()

# Split Data into train and test datasets (allocating 80% for Trainning):
df_train, df_test = train_test_random_spliter(rentalHouse_df, train_size=0.8)

# Separate the train and test datasets into features and predictor
X_train = df_train[selected_columns[1:]]
y_train = df_train['rental_price']

X_test = df_test[selected_columns[1:]]
y_test = df_test['rental_price']


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
    
# Initialize and train the KNN model with k=5
knn_regressor = KNNRegressor(k=5)
knn_regressor.fit(X_train.values, y_train.values)

# Predict on the test set
y_pred_knn = knn_regressor.predict(X_test.values)



# Creating the Tkinter window
root = tk.Tk()
root.title("House Rental Price Predictor")
root.geometry("500x600")
root.configure(bg='#f4f4f4')


# Function to calculate background features
def calculate_background_features(year_built, house_age):
    # Calculate year_sqrt, and inverse_house_age
    year_sqrt = np.sqrt(year_built)
    
    # Calculate inverse_house_age, ensuring no division by zero
    inverse_house_age = 1 / house_age if house_age != 0 else 0
    
    return year_sqrt, inverse_house_age

# Function to predict the rental price
def predict_rental_price():
    try:
        # Extract the feature values entered by the user
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

        # Calculate the age of the house
        current_year = datetime.now().year 
        house_age = current_year - year_built

        # Calculate the background features
        year_sqrt, inverse_house_age = calculate_background_features(year_built, house_age)

        # Combine all features into a single array
        features = [
            sqrt_ft, year_sqrt, lot_acres, latitude, 
            longitude, taxes, inverse_house_age, bedrooms, bathrooms, fireplaces, garage
        ]

        # Reshape input for prediction
        features = np.array(features).reshape(1, -1)

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

# Creating a frame to center the form elements
form_frame = tk.Frame(root, bg='#f4f4f4')
form_frame.place(relx=0.5, rely=0.5, anchor="center")

for i, label_text in enumerate(labels):
    label = tk.Label(form_frame, text=label_text, font=("Arial", 12), bg='#f4f4f4')
    label.grid(row=i, column=0, padx=10, pady=5, sticky="e")
    
    entry = tk.Entry(form_frame, font=("Arial", 12))
    entry.grid(row=i, column=1, padx=10, pady=5, ipadx=5, ipady=5)
    
    entries.append(entry)

# Mapping the entries for easy access
entry_sqrt_ft, entry_year_built, entry_lot_acres, entry_latitude, \
entry_longitude, entry_taxes, entry_bedrooms, entry_bathrooms, entry_fireplaces, \
entry_garage = entries

# Create the "PREDICT THE RENTAL PRICE" button
predict_button = tk.Button(form_frame, text="PREDICT THE RENTAL PRICE", font=("Arial", 14), bg='#3498db', fg='white', 
                           command=predict_rental_price)
predict_button.grid(row=len(labels), column=0, columnspan=2, pady=20, ipadx=10, ipady=10)

# Run the Tkinter event loop
root.mainloop()
