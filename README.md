## Time Series Forecasting with LSTM

---

### Project Overview

This project implements **Time Series Forecasting** using a **Long Short-Term Memory (LSTM)** neural network built with **Keras**. The primary objective is to predict future values of the monthly international airline passengers by modeling historical trends.

The model is trained on the classic **"Airline Passengers"** dataset, demonstrating how deep learning can be applied to sequence prediction tasks.

### Core Technologies

* **Python**
* **TensorFlow/Keras** (LSTM Model)
* **NumPy** (Numerical Operations)
* **Pandas** (Data Handling)
* **Scikit-learn** (Preprocessing and Metrics)

### Implementation Walkthrough

The project follows a standard machine learning workflow for sequence data:

#### 1. Data Preparation and Normalization

* **Loading:** The passenger count column from the `airline-passengers.csv` dataset is loaded.
* **Normalization:** Data is scaled using `MinMaxScaler` to restrict values between 0 and 1. This step is crucial for stable and efficient neural network training.
* **Splitting:** The dataset is divided into 67% for training and 33% for testing.

#### 2. Sequence Transformation

To use the data with an LSTM, the sequential time series is transformed into a supervised learning problem using a `look_back` window (set to 1).

* **Input (X):** The passenger count at time $t-1$.
* **Output (Y):** The passenger count at time $t$.

The data is then reshaped to the **3D format** required by LSTM: `[samples, time steps, features]`.

#### 3. LSTM Model Construction

A simple **Sequential** model architecture is defined:

| Layer       | Type  | Units | Activation       | Purpose                               |
| :---------- | :---- | :---- | :--------------- | :------------------------------------ |
| **1** | LSTM  | 4     | Tanh (Default)   | Capturing temporal dependencies       |
| **2** | Dense | 1     | Linear (Default) | Outputting the single predicted value |

The model is compiled with the **Adam optimizer** and **Mean Squared Error (MSE)** as the loss function, then trained for **100 epochs**.

#### 4. Evaluation and Results

* **Inverse Transformation:** Predictions are scaled back to the original passenger count values.
* **Metric:** **Root Mean Squared Error (RMSE)** is calculated for both the training and testing sets to assess model accuracy.
* **Visualization:** A plot is generated to visually compare the actual time series data against the model's predictions on both the training (fit) and testing (forecast) periods.

---

### Getting Started

To run this notebook:

1. **Clone the repository:**
   ```bash
   git clone [Your Repository URL]
   ```
2. **Install dependencies:**
   ```bash
   pip install numpy pandas tensorflow matplotlib scikit-learn
   ```
3. Ensure the `airline-passengers.csv` file is in the working directory.
4. Execute the cells in the `notebook.ipynb` file.
