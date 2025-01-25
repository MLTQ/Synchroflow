import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 1: Load the dataset
df = pd.read_csv('testdata1.csv')



# Step 3: Define the features and target (Persona labels are already added in the dataset)
features = ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 
            'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7',
            'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2',
            'Digital Channel 0 (D11)', 'Digital Channel 1 (D12)', 
            'Digital Channel 2 (D13)', 'Digital Channel 3 (D17)', 'Digital Channel 4 (D18)']

X = df[features].values
y = df['Persona'].values

# Step 4: Ensure the labels are in the range [0, 5) for training, i.e., from 0 to 4
y = y - 1  # If labels are in the range 1 to 5, subtract 1 to bring it to 0 to 4.

# Step 5: Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 6: Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Define the neural network model
model = models.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),  # 15 input features
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(5, activation='softmax')  # Output layer with 5 units (for the 5 personas)
])

# Step 8: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse if y is integer labels
              metrics=['accuracy'])

# Step 9: Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 10: Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Step 11: Save the model as a .tflite file (as "testmodel.tflite")
with open('testmodel.tflite', 'wb') as f:
    f.write(tflite_model)

# Step 12: Inference with TFLite Model

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="testmodel.tflite")
interpreter.allocate_tensors()

# Prepare new data (unlabeled group)
new_data = df[features].iloc[0].values  # Example for the first row
new_data = scaler.transform([new_data])  # Ensure it’s standardized the same way as training data

# Set the input tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], new_data.astype(np.float32))

# Run inference
interpreter.invoke()

# Get the output tensor (predicted class)
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)

# Print the predicted persona
print(f"Predicted Persona: {predicted_class + 1}")  # Add 1 to get the original label range (1 to 5)
