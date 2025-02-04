import torch
from torchvision import transforms
from PIL import Image
import json
import torch.nn as nn

def predict_and_save(model_path, img_path, label_to_class_path, output_prediction_path):
    """
    Load the model, perform prediction on an input image, and save the result.
    
    Args:
        model_path (str): Path to the trained model.
        img_path (str): Path to the input image for prediction.
        label_to_class_path (str): Path to the JSON file containing the label to class mapping.
        output_prediction_path (str): Path to save the prediction result as JSON.
    """
    # Load the model
    model = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")

    # Load the label to class mapping
    with open(label_to_class_path, "r") as f:
        label_to_class = json.load(f)
    print(f"Label to class mapping loaded from {label_to_class_path}")

    # Define preprocessing steps
    preprocess = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((1536, 662)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Load and preprocess the image
    image = Image.open(img_path)
    input_data = preprocess(image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')  # Add batch dimension
    print(f"Image loaded from {img_path} and preprocessed.")

    # Make prediction
    with torch.no_grad():
        prediction = model(input_data)
    
    # Process the prediction
    def process_prediction(pred, label_to_class):
        """Convert model output to human-readable format."""
        x=10
        probabilities = nn.Softmax(dim=1)(pred)
        top_probs, top_indices = torch.topk(probabilities, x, dim=1)  # Get top-k probabilities and indices
        
        results = []
        for i in range(x):
            index = top_indices[0, i].item()
            results.append({
                "class_index": index,
                "class_name": label_to_class.get(str(index), "Unknown"),  # Handle missing keys safely
                "probability": top_probs[0, i].item()
            })
        return results
    

    # Process the prediction
    processed_prediction = process_prediction(prediction, label_to_class)
        
    print(f"Prediction: {processed_prediction}")

    # Save the prediction to a file
    with open(output_prediction_path, "w") as f:
        for processed in processed_prediction:
            json.dump(processed, f)
            f.write(",\n")
    print(f"Prediction saved to {output_prediction_path}")