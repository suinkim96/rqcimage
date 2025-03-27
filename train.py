import os
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import argparse

from losses import *
from models import *
from utils import *
from datasets import *

def setup_logging(log_file="training.log"):
    # Setup logging: log INFO level messages to both console and file.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

def parse_args():
    parser = argparse.ArgumentParser(description="Train QRestormer model")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of epochs for training (default: 30)")
    parser.add_argument("--loss_type", type=str, default="Q",
                        help="Type of loss to use (default: 'Q')")
    parser.add_argument("--save_checkpoint", action="store_true",
                        help="Flag to save the trained model checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging()  # Initialize logging
    
    # Change working directory (modify if needed)
    logging.info("Working directory changed successfully.")

    # Define transformation functions
    transform_target = transforms.ToTensor()
    transform_input = transforms.Compose([
        GaussianNoise(sigma_type='constant', sigma_range=50),
        transforms.ToTensor()
    ])
    logging.info("Data transformation functions defined.")

    # Create datasets and dataloaders
    train_dir = "./data/BSD400"  # Training data directory
    test_dir = "./data/BSD68/original"
    train_dataset = TrainDataset(image_dir=train_dir,
                                 transform_target=transform_target, 
                                 transform_input=transform_input,
                                 grayscale=True)
    test_dataset = TestDataset(image_dir=test_dir,
                               transform_target=transform_target,
                               transform_input=transform_input,
                               grayscale=True)
    logging.info(f"Number of training images: {len(train_dataset)}")
    logging.info(f"Number of evaluation images: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, 
                              batch_size=32, 
                              shuffle=True, 
                              num_workers=0, 
                              collate_fn=patch_pair_collate_fn)
    test_loader = DataLoader(test_dataset, 
                             batch_size=1, 
                             shuffle=False, 
                             num_workers=0)

    # Initialize model, loss function, and optimizer
    model = Restormer(inp_channels=1, out_channels=1)
    # loss_function = RandomizedCheckLoss()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Model, loss function, and optimizer initialized. Device: {device}")

    total_batches = len(train_loader)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        logging.info(f"Epoch [{epoch+1}/{args.num_epochs}] started.")
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad()
            outputs = model(input_batch)
            if args.loss_type != 'Q':
                loss = loss_function(outputs, target_batch)
                with torch.no_grad():
                    batch_psnr = calculate_batch_psnr(target_batch, outputs, max_val=1.0)
            else:
                loss = loss_function(outputs[0], target_batch, outputs[1])
                with torch.no_grad():
                    batch_psnr = calculate_batch_psnr(target_batch, outputs[0], max_val=1.0)
            loss.backward()
            optimizer.step()

            logging.info(f"[{batch_idx+1}/{total_batches}] Train Loss: {loss.item():.4f}  Batch PSNR: {batch_psnr:.2f} dB")
            epoch_loss += loss.item()
        avg_loss = epoch_loss / total_batches
        logging.info(f"Epoch [{epoch+1}/{args.num_epochs}] - Average Loss: {avg_loss:.4f}")

    # Save model checkpoint if save_checkpoint option is set to True
    if args.save_checkpoint:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, "net.pth")
        torch.save(model.state_dict(), save_path)
        logging.info(f"Model saved successfully at: {save_path}")
    else:
        logging.info("Checkpoint saving is disabled; model not saved.")

if __name__ == "__main__":
    main()
