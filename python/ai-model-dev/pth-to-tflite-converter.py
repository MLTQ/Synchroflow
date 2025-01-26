import torch
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

# Step 1: Convert PyTorch model to ONNX
def convert_pytorch_to_onnx(pytorch_model_path, onnx_model_path, dummy_input_shape):
    model = torch.load(pytorch_model_path)
    model.eval()  # Set the model to evaluation mode
    
    # Dummy input for tracing the model
    dummy_input = torch.randn(*dummy_input_shape)  # Adjust the shape as needed
    
    # Export the model to ONNX
    torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=12)
    print(f"Model exported to {onnx_model_path} successfully.")

# Step 2: Convert ONNX model to TensorFlow model
def convert_onnx_to_tensorflow(onnx_model_path, tensorflow_model_path):
    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)
    
    # Convert the ONNX model to TensorFlow
    tf_rep = prepare(onnx_model)
    
    # Export the TensorFlow model
    tf_rep.export_graph(tensorflow_model_path)
    print(f"Model exported to {tensorflow_model_path} successfully.")

# Step 3: Convert TensorFlow model to TensorFlow Lite (.tflite)
def convert_tensorflow_to_tflite(tensorflow_model_path, tflite_model_path):
    # Load the saved TensorFlow model
    model = tf.saved_model.load(tensorflow_model_path)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_saved_model(tensorflow_model_path)
    tflite_model = converter.convert()
    
    # Save the converted model
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model converted to {tflite_model_path} successfully.")

# Main conversion pipeline
def convert_pytorch_to_tflite(pytorch_model_path, tflite_model_path, dummy_input_shape):
    # Convert PyTorch model to ONNX
    onnx_model_path = "model.onnx"
    convert_pytorch_to_onnx(pytorch_model_path, onnx_model_path, dummy_input_shape)
    
    # Convert ONNX model to TensorFlow model
    tensorflow_model_path = "model_tf"
    convert_onnx_to_tensorflow(onnx_model_path, tensorflow_model_path)
    
    # Convert TensorFlow model to TensorFlow Lite
    convert_tensorflow_to_tflite(tensorflow_model_path, tflite_model_path)

# Usage
if __name__ == "__main__":
    pytorch_model_path = "transformer_model.pth"  # Replace with your .pth model path
    tflite_model_path = "transformer_model.tflite"  # Output .tflite model path
    # dummy_input_shape = (1, 3, 224, 224)  # Adjust input shape to match your model's input
    dummy_input_shape = (1, 10, 8)

    convert_pytorch_to_tflite(pytorch_model_path, tflite_model_path, dummy_input_shape)
