from torch import optim
from torch.utils.data import DataLoader
from dataset import ClipEmotionDataset
# from model import ClipLinearProbe  # Original model
from model import ClipDeepProbe  # New deeper model
from train import train_epoch, validate
from utils import parse_args, get_device
import os
import torch
from PIL import Image
from torchvision import transforms
import sys

def run_inference(checkpoint_path, image_path, device=None):
    """
    Run inference on a single image using a trained model checkpoint
    Args:
        checkpoint_path: Path to the model checkpoint
        image_path: Path to the image to run inference on
        device: torch device to run inference on
    """
    if device is None:
        device = get_device('cuda')

    # Initialize model and load checkpoint
    # model = ClipLinearProbe(num_classes=7).to(device)  # Original model
    model = ClipDeepProbe(
        num_classes=7,
        hidden_dim=512,  # Size of hidden layers
        p_drop=0.2       # Dropout probability
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Use the same transforms as in dataset.py
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    # Map class indices to emotion labels
    emotion_labels = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral"
    }

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = logits.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # Print results
    emotion = emotion_labels[predicted_class]
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Predicted emotion: {emotion}")
    print(f"Confidence: {confidence:.2%}")

    # Print top-3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    print("\nTop 3 predictions:")
    for i in range(3):
        emotion = emotion_labels[top3_idx[0][i].item()]
        prob = top3_prob[0][i].item()
        print(f"{emotion}: {prob:.2%}")

    return predicted_class, confidence

def main():
    args = parse_args()
    device = get_device(args.device)

    # datasets & loaders
    train_ds = ClipEmotionDataset(args.data_dir, "train", args.max_samples)
    val_ds   = ClipEmotionDataset(args.data_dir, "test",   args.max_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    # model & optimizer
    # Original model
    # model = ClipLinearProbe(num_classes=7).to(device)
    
    # New deeper model
    model = ClipDeepProbe(
        num_classes=7,
        hidden_dim=512,  # Size of hidden layers
        p_drop=0.2       # Dropout probability
    ).to(device)
    
    # Using a lower learning rate for the deeper model / do some tests
    optimizer = optim.Adam(model.head.parameters(), lr=args.lr * 0.1)  # Reduced learning rate

    # Create checkpoint directory
    checkpoint_dir = os.path.join("expts", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_acc = 0.0

    # training loop
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
              f"Val Acc: {val_acc:.3f}")

        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_deep.pth")  # New name for deep model
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved to {checkpoint_path}")

if __name__ == "__main__":
    # Example of how to use both functions
    if len(sys.argv) > 1 and sys.argv[1] == "inference":
        # For inference mode
        checkpoint_path = "expts/checkpoints/best_model_deep.pth"  # Updated checkpoint name
        image_path = "data/inference/sad.png"
        run_inference(checkpoint_path, image_path)
    else:
        # For training mode
        main()
