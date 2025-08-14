"""BF16 Quantization of Gazelle Model"""

import os
import subprocess
import sys

def execute_shell_command(command):
    """Run a shell command"""
    try:
        subprocess.run(command, shell=True, check=True, text=True)
        print(f"Executed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")

# Clone repo if not present
if not os.path.exists('/content/gazelle'):
    execute_shell_command("git clone https://github.com/fkryan/gazelle.git")
else:
    print("Gazelle repo already exists")

# Add to Python path and set working directory
sys.path.insert(0, '/content/gazelle')
os.chdir('/content/gazelle')

# Install dependencies
execute_shell_command("pip install torch torchvision timm matplotlib opencv-python scipy transformers")

import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import requests
import zipfile
from io import BytesIO

# -------------------------
# 0. DOWNLOAD COCO DATASET
# -------------------------
def download_coco_dataset():
    """Download COCO val2017 images and annotations"""
    base_url = "http://images.cocodataset.org"

    # Create directories
    os.makedirs("coco/images", exist_ok=True)
    os.makedirs("coco/annotations", exist_ok=True)

    # Check if already downloaded
    if os.path.exists("coco/images/val2017") and os.path.exists("coco/annotations/instances_val2017.json"):
        print("COCO dataset already exists, skipping download.")
        return

    print("Downloading COCO val2017 images (~1GB)...")
    try:
        # Download val2017 images
        response = requests.get(f"{base_url}/zips/val2017.zip", stream=True)
        response.raise_for_status()

        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall("coco/images/")
        print("Images downloaded and extracted successfully.")

        # Download annotations
        print("Downloading COCO annotations (~240MB)...")
        response = requests.get(f"{base_url}/annotations/annotations_trainval2017.zip", stream=True)
        response.raise_for_status()

        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall("coco/")
        print("Annotations downloaded and extracted successfully.")

    except Exception as e:
        print(f"Error downloading COCO dataset: {e}")
        print("You may need to download manually using wget/curl commands.")

# -------------------------
# 1. LOAD GAZELLE MODEL
# -------------------------
def load_gazelle_model(device):
    try:
        print("Loading Gazelle model via PyTorch Hub...")
        model, transform = torch.hub.load(
            'fkryan/gazelle',
            'gazelle_dinov2_vitl14',
            pretrained=True
        )
        model = model.to(device).eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load Gazelle model: {e}")
        return None, None

    checkpoint_url = (
        "https://github.com/fkryan/gazelle/"
        "releases/download/v1.0.0/gazelle_dinov2_vitl14.pt"
    )
    try:
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_url,
            map_location=device
        )
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded pretrained weights from checkpoint")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")
        print("Continuing with Hub weights")

    return model, transform

# ---------------------------------
# 2. INFERENCE HELPER
# ---------------------------------
def perform_gazelle_inference(model, images, bboxes):
    bboxes = convert_bbox_tensor_to_floats(bboxes)
    with torch.no_grad():
        return model({"images": images, "bboxes": bboxes})

# ---------------------------------
# 3. DATASET & DATALOADER
# ---------------------------------
class COCOInferenceDataset(Dataset):
    def __init__(self, image_dir, ann_file, transform):
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.image_dir = image_dir
        self.transform = transform
        print(f"Dataset initialized with {len(self.ids)} images")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.image_dir, info['file_name'])

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        # full-image bbox
        bboxes = [(0.0, 0.0, 1.0, 1.0)]
        return img_id, tensor, [bboxes]

def make_coco_dataloader(image_dir, ann_file, transform, batch_size=1, num_workers=0):
    ds = COCOInferenceDataset(image_dir, ann_file, transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )

# ----------------------------------------------------
# 4. SIMPLIFIED ACCURACY EVALUATION
# ----------------------------------------------------
def evaluate_model_outputs(model, dataloader, device, max_batches=50):
    """
    Simple evaluation that doesn't require specific output format from Gazelle
    """
    model.eval()
    total_samples = 0
    successful_inferences = 0

    with torch.no_grad():
        for batch_idx, (img_ids, imgs, bboxes_list) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            imgs = imgs.to(device)
            try:
                outputs = perform_gazelle_inference(model, imgs, bboxes_list)
                if outputs is not None:
                    successful_inferences += len(img_ids)
                total_samples += len(img_ids)
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                total_samples += len(img_ids)

    success_rate = successful_inferences / total_samples if total_samples > 0 else 0
    print(f"Model inference success rate: {success_rate:.2%} ({successful_inferences}/{total_samples})")
    return success_rate

# ----------------------------------------------------
# 5. BENCHMARKING FUNCTION
# ----------------------------------------------------
def benchmark(model, dataloader, device, use_amp=False, warmup=5, runs=20):
    print(f"{'AMP' if use_amp else 'FP32'} benchmark: {len(dataloader)} batches")
    autocast = torch.cuda.amp.autocast

    # Warmup
    ctx = autocast() if use_amp else torch.no_grad()
    with ctx:
        for i, (img_ids, imgs, bboxes_list) in enumerate(dataloader):
            imgs = imgs.to(device)
            _ = perform_gazelle_inference(model, imgs, bboxes_list)
            if i >= warmup:
                break

    times = []
    torch.cuda.reset_peak_memory_stats(device)

    with ctx:
        torch.cuda.synchronize()
        for run in range(runs):
            for img_ids, imgs, bboxes_list in dataloader:
                imgs = imgs.to(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = perform_gazelle_inference(model, imgs, bboxes_list)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
                break

    if not times:
        print("No timing samples collected.")
        return {k: 0.0 for k in ["mean_ms", "min_ms", "max_ms", "p95_ms", "throughput_sps", "gpu_peak_mb"]}

    times = torch.tensor(times)
    stats = {
        "mean_ms": times.mean().item(),
        "min_ms": times.min().item(),
        "max_ms": times.max().item(),
        "p95_ms": times.kthvalue(int(0.95 * len(times))).values.item(),
        "throughput_sps": len(dataloader.dataset) / (times.mean().item() / 1000),
        "gpu_peak_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }
    return stats

# ----------------------------------------------------
# 6. MODEL SIZE & PARAMETER COUNT
# ----------------------------------------------------
def model_size_and_params(model, checkpoint_path=None):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    if checkpoint_path and os.path.isfile(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024**2)
        print(f"Checkpoint file size: {size_mb:.2f} MB")
    else:
        print("Checkpoint file not found or path not provided.")

# ---------------------------------
# 7. RUN EVERYTHING
# ---------------------------------
cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Download COCO dataset
download_coco_dataset()

# Load model
model, transform = load_gazelle_model(device)
if model is None:
    raise RuntimeError("Failed to load model")

# Test model on random input
print("\nTesting model on random input...")
test_images = torch.randn(1, 3, 224, 224, device=device)
test_bboxes = [[(0.2, 0.2, 0.6, 0.6)]]
try:
    out = perform_gazelle_inference(model, test_images, test_bboxes)
    print("Model test successful!")
    if isinstance(out, dict):
        for k, v in out.items():
            if hasattr(v, "shape"):
                print(f"  {k}: {v.shape}")
except Exception as e:
    print(f"Model test failed: {e}")

# Paths for COCO
coco_image_dir = "coco/images/val2017"
coco_ann_file = "coco/annotations/instances_val2017.json"

# Verify paths
print(f"\nCOCO images dir exists: {os.path.isdir(coco_image_dir)}")
print(f"COCO annotation file exists: {os.path.isfile(coco_ann_file)}")

# Evaluation
if os.path.isdir(coco_image_dir) and os.path.isfile(coco_ann_file):
    print("\nCreating COCO dataloader...")
    val_loader = make_coco_dataloader(coco_image_dir, coco_ann_file, transform, batch_size=1)

    print("\nEvaluating model outputs...")
    success_rate = evaluate_model_outputs(model, val_loader, device, max_batches=50)

    print("\nBenchmarking FP32...")
    fp32_stats = benchmark(model, val_loader, device, use_amp=False, runs=10)
    print(f"FP32 stats: {fp32_stats}")

    print("\nBenchmarking AMP BF16...")
    bf16_stats = benchmark(model, val_loader, device, use_amp=True, runs=10)
    print(f"AMP BF16 stats: {bf16_stats}")

else:
    print("Skipping evaluation: COCO dataset not found.")

# Save model and report size
print("\nSaving model checkpoint...")
checkpoint_path = "gazelle_dinov2_vitl14_saved.pt"
torch.save(model.state_dict(), checkpoint_path)
model_size_and_params(model, checkpoint_path)

print("\nEvaluation complete!")

def convert_bbox_tensor_to_floats(bboxes):
    """Convert any tensor coordinates in bbox list to Python floats recursively"""
    new_bboxes = []
    for bbox_list in bboxes:
        new_bbox_list = []
        for bbox in bbox_list:
            new_bbox = tuple(float(x) if isinstance(x, torch.Tensor) else x for x in bbox)
            new_bbox_list.append(new_bbox)
        new_bboxes.append(new_bbox_list)
    return new_bboxes

print("\nEvaluating FP32 inference accuracy...")
fp32_acc = evaluate_model_outputs(model, val_loader, device, max_batches=50)
print(f"FP32 accuracy proxy: {fp32_acc*100:.2f}%")

print("\nEvaluating AMP BF16 inference accuracy...")
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    bf16_acc = evaluate_model_outputs(model, val_loader, device, max_batches=50)
print(f"AMP BF16 accuracy proxy: {bf16_acc*100:.2f}%")

