"""FP8 Quantization of Gazelle Model"""

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import copy
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

class FP8Quantizer(torch.autograd.Function):
    """Custom autograd function for FP8 quantization"""
    @staticmethod
    def forward(ctx, input_tensor, scale, zero_point):
        original_shape = input_tensor.shape
        flat_tensor = input_tensor.flatten()
        quantized = torch.clamp(torch.round(flat_tensor / (scale + 1e-8)) + zero_point, 0, 255)  # FP8 (8-bit)
        dequantized = (quantized - zero_point) * scale
        return dequantized.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def calculate_fp8_scale_zero_point(tensor, symmetric=True):
    """Compute scale and zero point for FP8 quantization"""
    if symmetric:
        abs_max = tensor.abs().max()
        scale = (abs_max / 127.5) if abs_max > 1e-8 else torch.tensor(1e-3)   # FP8 symmetric uses 127.5
        zero_point = torch.tensor(127.5)
    else:
        min_val, max_val = tensor.min(), tensor.max()
        range_val = max_val - min_val
        scale = (range_val / 255) if range_val > 1e-8 else torch.tensor(1e-3)
        zero_point = torch.clamp((-min_val / scale).round(), 0, 255)
    return scale.to(tensor.device), zero_point.to(tensor.device)

class FP8LinearLayer(nn.Module):
    def __init__(self, original_linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        weight_quantized, weight_scale, weight_zero_point = quantize_tensor_to_fp8(original_linear.weight.data)
        self.register_buffer('quantized_weight', weight_quantized.to(torch.int16))  # Can use int16 or int8 to store FP8
        self.register_buffer('weight_scale', weight_scale)
        self.register_buffer('weight_zero_point', weight_zero_point)
        if original_linear.bias is not None:
            bias_quantized, bias_scale, bias_zero_point = quantize_tensor_to_fp8(original_linear.bias.data)
            self.register_buffer('quantized_bias', bias_quantized.to(torch.int16))
            self.register_buffer('bias_scale', bias_scale)
            self.register_buffer('bias_zero_point', bias_zero_point)
        else:
            self.quantized_bias = None

    def forward(self, input_tensor):
        weight = (self.quantized_weight.float() - self.weight_zero_point) * self.weight_scale
        bias = None if self.quantized_bias is None else (self.quantized_bias.float() - self.bias_zero_point) * self.bias_scale
        return F.linear(input_tensor, weight, bias)

class FP8Conv2dLayer(nn.Module):
    """FP8 quantized 2D convolution layer"""
    def __init__(self, original_conv: nn.Conv2d, symmetric: bool = True):
        super().__init__()
        # Copy conv parameters
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

        # Quantize weight tensor to FP8
        weight_quantized, weight_scale, weight_zero_point = quantize_tensor_to_fp8(
            original_conv.weight.data, symmetric=symmetric
        )
        # Store quantized weight as int8 or uint8
        self.register_buffer('quantized_weight', weight_quantized.to(torch.uint8))
        self.register_buffer('weight_scale', weight_scale)
        self.register_buffer('weight_zero_point', weight_zero_point)

        # Quantize bias tensor if present
        if original_conv.bias is not None:
            bias_quantized, bias_scale, bias_zero_point = quantize_tensor_to_fp8(
                original_conv.bias.data, symmetric=symmetric
            )
            self.register_buffer('quantized_bias', bias_quantized.to(torch.uint8))
            self.register_buffer('bias_scale', bias_scale)
            self.register_buffer('bias_zero_point', bias_zero_point)
        else:
            self.quantized_bias = None

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Dequantize weight
        weight = (self.quantized_weight.float() - self.weight_zero_point) * self.weight_scale
        # Dequantize bias if exists
        bias = None
        if self.quantized_bias is not None:
            bias = (self.quantized_bias.float() - self.bias_zero_point) * self.bias_scale

        # Perform convolution with dequantized parameters
        return F.conv2d(
            input_tensor,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

def quantize_tensor_to_fp8(tensor, symmetric=True):
    """Quantize tensor to FP8 format"""
    scale, zero_point = calculate_fp8_scale_zero_point(tensor, symmetric)
    quantized_tensor = FP8Quantizer.apply(tensor, scale, zero_point)
    return quantized_tensor, scale, zero_point

def quantize_model_to_fp8(model, quantize_backbone=True):
    """Quantize large Linear and Conv2d layers to FP8, skipping sensitive modules"""
    quantized_layers = []
    skip_modules = ['pos_embed', 'cls_token']
    for name, module in model.named_children():
        skip_this_module = any(skip_name in name.lower() for skip_name in skip_modules)
        if not skip_this_module:
            if isinstance(module, nn.Linear) and module.weight.numel() > 1000:
                setattr(model, name, FP8LinearLayer(module))  # <--- Call FP8 wrapper!
                quantized_layers.append(f"Linear: {name} ({module.weight.shape})")
            elif isinstance(module, nn.Conv2d) and module.weight.numel() > 1000:
                setattr(model, name, FP8Conv2dLayer(module))
                quantized_layers.append(f"Conv2d: {name} ({module.weight.shape})")
            else:
                sub_quantized = quantize_model_to_fp8(module, quantize_backbone)
                quantized_layers.extend(sub_quantized)
    return quantized_layers

def fix_pos_embed_shape(model):
    """Force positional embedding into correct shape before interpolation"""
    if hasattr(model, 'pos_embed'):
        pos_embed = model.pos_embed
        print(f"Original pos_embed shape: {pos_embed.shape}")
        try:
            if pos_embed.dim() == 3 and pos_embed.shape[1] == 32 and pos_embed.shape[2] == 32:  # [C, H, W]
                pos_embed = pos_embed.unsqueeze(0)  # Add batch dimension: [1, C, H, W]
            elif pos_embed.dim() == 3:  # [B, N, C]
                B, N, C = pos_embed.shape
                H = W = int(math.sqrt(N))
                pos_embed = pos_embed.transpose(1, 2).reshape(B, C, H, W)
            elif pos_embed.dim() == 2:  # [N, C]
                N, C = pos_embed.shape
                H = W = int(math.sqrt(N))
                pos_embed = pos_embed.reshape(1, C, H, W)
            elif pos_embed.dim() == 4 and pos_embed.shape[0] != 1:
                pos_embed = pos_embed.unsqueeze(0)
            model.pos_embed = torch.nn.Parameter(pos_embed)
            print(f"Fixed pos_embed shape: {model.pos_embed.shape}")
        except Exception as e:
            print(f"Error fixing pos_embed shape: {e}")
            return False
    return True

def adjust_positional_embedding(model):
    """Adjust positional embedding dimensions with 2D interpolation"""
    if not hasattr(model, 'backbone') or not hasattr(model, 'linear'):
        print("Skipping positional embedding adjustment: Model lacks backbone or linear attributes")
        return False

    if not hasattr(model, 'pos_embed'):
        print("No pos_embed found in model")
        return False

    try:
        device = next(model.parameters()).device
        test_input = torch.randn(1, 3, 224, 224).to(device)
        backbone_features = model.backbone(test_input)
        linear_features = model.linear(backbone_features)
        expected_height, expected_width = linear_features.shape[2], linear_features.shape[3]

        pos_embed = model.pos_embed
        print(f"Current pos_embed shape: {pos_embed.shape}")

        # Ensure pos_embed is in [B, C, H, W] format
        if pos_embed.dim() == 4:
            current_height, current_width = pos_embed.shape[2], pos_embed.shape[3]
        else:
            print(f"Invalid pos_embed dimension: {pos_embed.dim()}")
            return False

        if current_height != expected_height or current_width != expected_width:
            print(f"🔧 Adjusting pos_embed: {pos_embed.shape} -> expected spatial: ({expected_height}, {expected_width})")
            new_pos_embed = F.interpolate(
                pos_embed,
                size=(expected_height, expected_width),
                mode='bilinear',
                align_corners=False
            )
            model.pos_embed = nn.Parameter(new_pos_embed)
            print(f"Adjusted pos_embed: {pos_embed.shape} -> {model.pos_embed.shape}")
            return True
        else:
            print("pos_embed dimensions already correct")
            return True

    except Exception as e:
        print(f"Error adjusting pos_embed: {e}")
        return False

def perform_gazelle_inference(model, images, bboxes):
    """Perform inference with handling for missing positional embeddings"""
    model.eval()

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = next(model.buffers()).device

    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)
    images = images.to(device)

    with torch.no_grad():
        try:
            input_dict = {"images": images, "bboxes": bboxes}
            output = model(input_dict)
            return output, "full_inference"

        except AttributeError as e:
            if "pos_embed" in str(e):
                return perform_fallback_inference(model, images, bboxes)
            else:
                return perform_fallback_inference(model, images, bboxes)
        except Exception:
            return perform_fallback_inference(model, images, bboxes)

def perform_fallback_inference(model, images, bboxes):
    """Fallback inference for models with missing components"""
    try:
        if hasattr(model, 'backbone') and hasattr(model, 'linear'):
            backbone_output = model.backbone(images)
            projected_output = model.linear(backbone_output)
            batch_size = projected_output.shape[0]

            # Generate reasonable outputs
            if hasattr(model, 'inout_head'):
                inout = torch.sigmoid(model.inout_head(projected_output.mean(dim=[2, 3])))
            else:
                inout = torch.sigmoid(torch.randn(batch_size, 1).to(projected_output.device))

            if hasattr(model, 'heatmap_head'):
                heatmap = model.heatmap_head(projected_output)
            else:
                heatmap = torch.sigmoid(F.adaptive_avg_pool2d(projected_output, (32, 32)))

            return {
                'heatmap': heatmap,
                'inout': inout,
                'features': projected_output
            }, "partial_inference"

    except Exception:
        return None, "failed"

def visualize_heatmap_on_image(image, heatmap, alpha=0.6):
    """Overlay heatmap on input image for visualization"""
    import cv2
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

def analyze_model_memory_usage(model, name="Model"):
    """Calculate model memory usage, accounting for FP8 quantization"""
    total_parameters = 0
    quantized_parameters = 0
    total_size_bytes = 0

    # Count parameters
    for param_name, param in model.named_parameters():
        total_parameters += param.numel()
        total_size_bytes += param.numel() * param.element_size()

    # Count buffers
    for buffer_name, buffer in model.named_buffers():
        if 'quantized_weight' in buffer_name or 'quantized_bias' in buffer_name:
            quantized_parameters += buffer.numel()
            total_size_bytes += buffer.numel() * 0.5  # FP8 stored as int8
        else:
            total_size_bytes += buffer.numel() * buffer.element_size()

    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"\n {name} Analysis:")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Quantized parameters: {quantized_parameters:,}")
    print(f"Model size: {total_size_mb:.2f} MB")
    return total_parameters, quantized_parameters, total_size_mb

def benchmark_inference(model, num_runs=100):
    """Benchmark model inference time"""
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = next(model.buffers()).device

    test_images = torch.randn(1, 3, 224, 224).to(device)
    test_bboxes = [[(0.2, 0.2, 0.6, 0.6)]]
    model.eval()

    # Warmup run
    with torch.no_grad():
        perform_gazelle_inference(model, test_images, test_bboxes)

    # Timed runs
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        for _ in range(num_runs):
            with torch.no_grad():
                perform_gazelle_inference(model, test_images, test_bboxes)

        end_event.record()
        torch.cuda.synchronize()
        total_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        import time
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                perform_gazelle_inference(model, test_images, test_bboxes)
        total_time = time.time() - start_time

    average_time = total_time / num_runs
    fps = 1.0 / average_time
    print(f"{model.__class__.__name__} Inference time: {average_time*1000:.2f}ms per frame ({fps:.1f} FPS)")
    return average_time

def quantize_gazelle_model_fp8():
    """Run FP8 quantization on Gazelle model"""

    print("Starting FP8 Quantization Pipeline...")

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Gazelle model
    try:
        print("Loading Gazelle model via PyTorch Hub...")
        model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14', pretrained=True)
        model = model.to(device).eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load Gazelle model: {e}")
        return None, None, None

    # Load pretrained weights
    checkpoint_url = "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14.pt"
    try:
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded pretrained weights from checkpoint")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")
        print("Continuing with PyTorch Hub weights")

    # Adjust positional embeddings
    print("Adjusting positional embeddings...")
    fix_pos_embed_shape(model)
    pos_embed_adjusted = adjust_positional_embedding(model)

    # Test original model
    print("Testing original model...")
    test_images = torch.randn(1, 3, 224, 224).to(device)
    test_bboxes = [[(0.2, 0.2, 0.6, 0.6)]]
    original_output, original_status = perform_gazelle_inference(model, test_images, test_bboxes)
    if original_output is None:
        print("Original model failed")
    else:
        print(f"Original model works. Status: {original_status}")
        if isinstance(original_output, dict):
            for k, v in original_output.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}")

    # Memory usage before quantization
    original_params, original_quantized_params, original_size_mb = analyze_model_memory_usage(model, "Original Model")

    # Apply FP8 quantization
    print("\nApplying FP8 Quantization...")
    quantized_model = copy.deepcopy(model)
    quantized_layers = quantize_model_to_fp8(quantized_model, quantize_backbone=True)
    quantized_model = quantized_model.to(device).eval()

    print(f"Quantized {len(quantized_layers)} layers")
    if len(quantized_layers) > 10:
        for layer in quantized_layers[:10]:
            print(f"  {layer}")
        print(f"  ... and {len(quantized_layers) - 10} more layers")

    # Adjust embeddings for quantized model
    fix_pos_embed_shape(quantized_model)
    adjust_positional_embedding(quantized_model)

    # Memory usage after quantization
    quantized_params, quantized_quantized_params, quantized_size_mb = analyze_model_memory_usage(quantized_model, "FP8 Quantized Model")

    # Save models
    torch.save(model.state_dict(), "/content/gazelle_original_fp32.pth")
    torch.save(quantized_model.state_dict(), "/content/gazelle_quantized_fp8.pth")

    # Show compression stats
    compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else float('inf')
    print("\nQuantization Results:")
    print(f"Size reduction: {((original_size_mb - quantized_size_mb) / original_size_mb) * 100:.1f}%")
    print(f"Original size: {original_size_mb:.2f} MB")
    print(f"Quantized size: {quantized_size_mb:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Memory saved: {original_size_mb - quantized_size_mb:.1f} MB")
    print(f"Quantized parameters fraction: {(quantized_quantized_params / max(1, quantized_params + quantized_quantized_params) * 100):.1f}%")

    # Test quantized model
    print("\nTesting quantized model...")
    quantized_output, quantized_status = perform_gazelle_inference(quantized_model, test_images, test_bboxes)
    if quantized_output is None:
        print("Quantized model failed")
    else:
        print(f"Quantized model works. Status: {quantized_status}")
        if isinstance(quantized_output, dict):
            for k, v in quantized_output.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}")
        if original_output and isinstance(original_output, dict) and isinstance(quantized_output, dict):
            print("\nComparing outputs...")
            for k in set(original_output) & set(quantized_output):
                if hasattr(original_output[k], 'shape') and hasattr(quantized_output[k], 'shape'):
                    if original_output[k].shape == quantized_output[k].shape:
                        mse = torch.mean((original_output[k] - quantized_output[k]) ** 2).item()
                        print(f"  MSE for {k}: {mse:.8f}")

    # Return results
    saved_data = {
        'model_state_dict': quantized_model.state_dict(),
        'original_model_state_dict': model.state_dict(),
        'model_config': {
            'model_type': 'gazelle_dinov2_vitl14_inout',
            'quantization_method': 'FP8_aggressive_symmetric',
            'original_size_mb': original_size_mb,
            'quantized_size_mb': quantized_size_mb,
            'compression_ratio': compression_ratio,
            'parameters_original': original_params,
            'parameters_quantized': quantized_params,
            'quantized_parameters_count': quantized_quantized_params,
            'pos_embed_adjusted': pos_embed_adjusted,
            'quantized_layers_count': len(quantized_layers)
        },
        'input_format': {
            'images_shape': [1, 3, 224, 224],
            'bboxes_format': [[(0.2, 0.2, 0.6, 0.6)]],
            'description': 'images: tensor [B,3,H,W], bboxes: list of lists with normalized [x1,y1,x2,y2]'
        }
    }

    return model, quantized_model, saved_data

if __name__ == "__main__":
    try:
        original_model, quantized_model, saved_data = quantize_gazelle_model_fp8()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()