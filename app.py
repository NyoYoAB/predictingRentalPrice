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

# Function to predict the rental price
def predict_rental_price():
    try:
        # Extract the feature values entered by the user
        features = [
            float(entry_sqrt_ft.get()),
            float(entry_year_sqrt.get()),
            float(entry_zipcode_mean_price.get()),
            float(entry_lot_acres.get()),
            float(entry_latitude.get()),
            float(entry_longitude.get()),
            float(entry_taxes.get()),
            float(entry_inverse_house_age.get()),
            float(entry_bedrooms.get()),
            float(entry_bathrooms.get()),
            float(entry_fireplaces.get()),
            float(entry_garage.get())
        ]

        # Reshape input for prediction
        features = np.array(features).reshape(1, -1)

        # Make the prediction using the KNN model
        predicted_price = knn_regressor.predict(features)[0]

        # Display the prediction result
        messagebox.showinfo("Prediction", f"The predicted rental price is: ${predicted_price:.2f}")
    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# Labels and Entry fields configuration
labels = ["Square Footage (sqrt_ft)", "Year Sqrt", "Zipcode Mean Price", "Lot Acres", 
          "Latitude", "Longitude", "Taxes", "Inverse House Age", "Bedrooms", 
          "Bathrooms", "Fireplaces", "Garage"]

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
entry_sqrt_ft, entry_year_sqrt, entry_zipcode_mean_price, entry_lot_acres, entry_latitude, \
entry_longitude, entry_taxes, entry_inverse_house_age, entry_bedrooms, entry_bathrooms, \
entry_fireplaces, entry_garage = entries

# Create the "PREDICT THE RENTAL PRICE" button
predict_button = tk.Button(form_frame, text="PREDICT THE RENTAL PRICE", font=("Arial", 14), bg='#3498db', fg='white', 
                           command=predict_rental_price)
predict_button.grid(row=len(labels), column=0, columnspan=2, pady=20, ipadx=10, ipady=10)

# Run the Tkinter event loop
root.mainloop()
