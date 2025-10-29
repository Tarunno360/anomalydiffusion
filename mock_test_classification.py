import torch
from torchvision.models import resnet34
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Simple mock test for anomaly classification
def mock_test():
    # Paths
    '''mvtec_path = "C:/Users/tarun/AnnomalyDiffusion/testingdata"
    checkpoint_path = "checkpoints/classification/leather.pckl"
    sample_name = "leather"
'''
    mvtec_path = "C:/Users/tarun/AnnomalyDiffusion/testingdata"
    checkpoint_path = "checkpoints/classification/wood.pckl"
    sample_name = "wood"
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return

    # Anomaly types (classes)
    anomaly_names = ['color', 'combined', 'hole', 'liquid','scratch']

    # Load model
    model = resnet34(pretrained=True, progress=True)
    model.fc = nn.Linear(model.fc.in_features, len(anomaly_names))
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    model.cuda()

    # Image preprocessing
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([256, 256])
    ])

    # Test on all images from each anomaly type
    print("Testing anomaly classification on all examples...")

    total_correct = 0
    total_images = 0
    per_anomaly_correct = {anomaly: 0 for anomaly in anomaly_names}
    per_anomaly_total = {anomaly: 0 for anomaly in anomaly_names}

    for anomaly in anomaly_names:  # Test on all anomalies
        test_dir = os.path.join(mvtec_path, sample_name, 'test', anomaly)
        if not os.path.exists(test_dir):
            print(f"Test directory not found: {test_dir}")
            continue

        image_files = os.listdir(test_dir)  # All images per anomaly

        for img_file in image_files:
            img_path = os.path.join(test_dir, img_file)
            image = loader(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()

            with torch.no_grad():
                output = model(image)
                prediction = torch.argmax(output, 1).item()

            predicted_anomaly = anomaly_names[prediction]
            is_correct = (predicted_anomaly == anomaly)
            total_correct += int(is_correct)
            total_images += 1
            per_anomaly_correct[anomaly] += int(is_correct)
            per_anomaly_total[anomaly] += 1

            print(f"Image: {img_file} | True: {anomaly} | Predicted: {predicted_anomaly} | Correct: {is_correct}")

    # Calculate and print accuracies
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_images})")

    print("\nPer-Anomaly Accuracy:")
    for anomaly in anomaly_names:
        if per_anomaly_total[anomaly] > 0:
            acc = per_anomaly_correct[anomaly] / per_anomaly_total[anomaly]
            print(f"{anomaly}: {acc:.4f} ({per_anomaly_correct[anomaly]}/{per_anomaly_total[anomaly]})")
        else:
            print(f"{anomaly}: No images found")

    print("Mock test completed.")

if __name__ == "__main__":
    mock_test()
