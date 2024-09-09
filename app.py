import tkinter as tk
from tkinter import messagebox
import numpy as np

class KNNRegressor:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for x_test in X_test:
            distances = np.sqrt(np.sum((self.X_train - x_test) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_targets = self.y_train[k_indices]
            y_pred.append(np.mean(k_nearest_targets))
        return np.array(y_pred)

# Creating the Tkinter window
root = tk.Tk()
root.title("House Rental Price Predictor")
root.geometry("500x600")
root.configure(bg='#f4f4f4')

# Creating the KNN model (dummy for demonstration)
X_train = np.random.rand(100, 12)  # Replace with your actual training data
y_train = np.random.rand(100) * 5000  # Replace with your actual target values
knn_regressor = KNNRegressor(k=5)
knn_regressor.fit(X_train, y_train)

# Function to calculate background features
def calculate_background_features(year_built, zipcode, house_age):
    # Calculate year_sqrt, zipcode_mean_price, and inverse_house_age
    year_sqrt = np.sqrt(year_built)
    
    # Dummy mapping of zipcode_mean_price (this should be calculated from actual data)
    zipcode_mean_price = 2000  # Replace with actual calculation based on dataset
    
    # Calculate inverse_house_age, ensuring no division by zero
    inverse_house_age = 1 / house_age if house_age != 0 else 0
    
    return year_sqrt, zipcode_mean_price, inverse_house_age

# Function to predict the rental price
def predict_rental_price():
    try:
        # Extract the feature values entered by the user
        sqrt_ft = float(entry_sqrt_ft.get())
        year_built = int(entry_year_built.get())
        zipcode = int(entry_zipcode.get())
        lot_acres = float(entry_lot_acres.get())
        latitude = float(entry_latitude.get())
        longitude = float(entry_longitude.get())
        taxes = float(entry_taxes.get())
        bedrooms = int(entry_bedrooms.get())
        bathrooms = int(entry_bathrooms.get())
        fireplaces = int(entry_fireplaces.get())
        garage = int(entry_garage.get())

        # Calculate the age of the house
        current_year = 2024  # Replace with dynamic year if needed
        house_age = current_year - year_built

        # Calculate the background features
        year_sqrt, zipcode_mean_price, inverse_house_age = calculate_background_features(year_built, zipcode, house_age)

        # Combine all features into a single array
        features = [
            sqrt_ft, year_sqrt, zipcode_mean_price, lot_acres, latitude, 
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
labels = ["Square Footage (sqrt_ft)", "Year Built", "Zipcode", "Lot Acres", 
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
entry_sqrt_ft, entry_year_built, entry_zipcode, entry_lot_acres, entry_latitude, \
entry_longitude, entry_taxes, entry_bedrooms, entry_bathrooms, entry_fireplaces, \
entry_garage = entries

# Create the "PREDICT THE RENTAL PRICE" button
predict_button = tk.Button(form_frame, text="PREDICT THE RENTAL PRICE", font=("Arial", 14), bg='#3498db', fg='white', 
                           command=predict_rental_price)
predict_button.grid(row=len(labels), column=0, columnspan=2, pady=20, ipadx=10, ipady=10)

# Run the Tkinter event loop
root.mainloop()
