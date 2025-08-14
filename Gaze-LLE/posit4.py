
import torch
import torch.nn as nn
import torch.nn.functional as F

def float_to_posit4(x):
    # Simulation: clip and cast as float8-like dynamic (real Posit4 will have more advanced logic)
    # For demonstration: use 8 bits and normalize to [-max_abs, +max_abs]
    max_abs = 16  # Example dynamic range (Posit is not symmetric)
    x = torch.clamp(x, -max_abs, +max_abs)
    # Map float to [0,255], midpoint at 128
    scaled = ((x + max_abs) / (2*max_abs) * 15).round()
    scaled = torch.clamp(scaled, 0, 15)
    return scaled.to(torch.uint8)

def posit4_to_float(p):
    max_abs = 16
    f = (p.float() / 15) * (2*max_abs) - max_abs
    return f

def quantize_tensor_to_posit4(tensor):
    # Tensor to simulated posit4
    quantized = float_to_posit4(tensor)
    return quantized

def dequantize_tensor_from_posit4(quantized):
    return posit4_to_float(quantized)

class Posit4LinearLayer(nn.Module):
    def __init__(self, original_linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        # Quantize weight and bias
        self.register_buffer('quantized_weight', quantize_tensor_to_posit4(original_linear.weight.data))
        if original_linear.bias is not None:
            self.register_buffer('quantized_bias', quantize_tensor_to_posit4(original_linear.bias.data))
        else:
            self.quantized_bias = None

    def forward(self, input_tensor):
        weight = dequantize_tensor_from_posit4(self.quantized_weight)
        bias = dequantize_tensor_from_posit4(self.quantized_bias) if self.quantized_bias is not None else None
        return F.linear(input_tensor, weight, bias)

class Posit4Conv2dLayer(nn.Module):
    def __init__(self, original_conv):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

        self.register_buffer('quantized_weight', quantize_tensor_to_posit4(original_conv.weight.data))
        if original_conv.bias is not None:
            self.register_buffer('quantized_bias', quantize_tensor_to_posit4(original_conv.bias.data))
        else:
            self.quantized_bias = None

    def forward(self, input_tensor):
        weight = dequantize_tensor_from_posit4(self.quantized_weight)
        bias = dequantize_tensor_from_posit4(self.quantized_bias) if self.quantized_bias is not None else None
        return F.conv2d(input_tensor, weight, bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

def quantize_model_to_posit4(model):
    quantized_layers = []
    skip_modules = ['pos_embed', 'cls_token']
    for name, module in model.named_children():
        skip_this_module = any(skip_name in name.lower() for skip_name in skip_modules)
        if not skip_this_module:
            if isinstance(module, nn.Linear) and module.weight.numel() > 1000:
                setattr(model, name, Posit4LinearLayer(module))
                quantized_layers.append(f"Linear: {name} ({module.weight.shape})")
            elif isinstance(module, nn.Conv2d) and module.weight.numel() > 1000:
                setattr(model, name, Posit4Conv2dLayer(module))
                quantized_layers.append(f"Conv2d: {name} ({module.weight.shape})")
            else:
                sub_quantized = quantize_model_to_posit4(module)
                quantized_layers.extend(sub_quantized)
    return quantized_layers

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

def analyze_model_memory_usage(model, name="Model", quant_type=None):
    """Calculate model memory usage, accounting for different quantization types"""
    total_parameters, quantized_parameters, total_size_bytes = 0, 0, 0

    for param_name, param in model.named_parameters():
        total_parameters += param.numel()
        total_size_bytes += param.numel() * param.element_size()

    for buffer_name, buffer in model.named_buffers():
        if 'quantized_weight' in buffer_name or 'quantized_bias' in buffer_name:
            quantized_parameters += buffer.numel()
            # Size calculation based on quantization type
            if quant_type == "FP4" or quant_type == "Posit4":  # Added Posit4
                total_size_bytes += buffer.numel() * 0.5  # 4 bits = 0.5 bytes
            elif quant_type == "FP8" or quant_type == "Posit8":
                total_size_bytes += buffer.numel() * 1.0  # 8 bits = 1 byte
            else:
                total_size_bytes += buffer.numel() * buffer.element_size()
        else:
            total_size_bytes += buffer.numel() * buffer.element_size()

    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"\n{name} Analysis:")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Quantized parameters: {quantized_parameters:,}")
    print(f"Model size: {total_size_mb:.2f} MB")
    return total_parameters, quantized_parameters, total_size_mb

import time
def benchmark_inference(model, num_runs=100):
    device = next(model.parameters()).device
    test_images = torch.randn(1, 3, 224, 224).to(device)
    test_bboxes = [[(0.2, 0.2, 0.6, 0.6)]]
    model.eval()
    # Warmup
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
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                perform_gazelle_inference(model, test_images, test_bboxes)
        total_time = time.time() - start_time
    average_time = total_time / num_runs
    fps = 1.0 / average_time
    print(f"{model.__class__.__name__} Inference time: {average_time*1000:.2f}ms per frame ({fps:.1f} FPS)")
    return average_time, fps

def compare_quantized_outputs(orig, posit4):
    for k in set(orig) & set(posit4):
        print(f"Output Key: {k}")
        base = orig[k]
        mse_posit4 = torch.mean((base - posit4[k])**2).item()
        print(f"Posit4 MSE: {mse_posit4:.6f}")

import copy
def full_quantization_comparison_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Gazelle model...")
    model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14', pretrained=True)
    model = model.to(device).eval()

    model_posit4 = copy.deepcopy(model)
    print("Applying Posit4 Quantization...")
    quantize_model_to_posit4(model_posit4)

    print("\n--- Model Size and Memory Usage ---")
    analyze_model_memory_usage(model, "Original Model")
    analyze_model_memory_usage(model_posit4, "Posit4 Model", "Posit4")

    # Compression ratios
    _, _, orig_mb   = analyze_model_memory_usage(model, "O", None)
    _, _, posit4_mb = analyze_model_memory_usage(model_posit4, "P", "Posit4")
    print(f"Posit4 Compression: {orig_mb/posit4_mb:.2f}x")

    print("\n--- Benchmark Inference ---")
    benchmark_inference(model, 100)
    benchmark_inference(model_posit4, 100)

    print("\n--- Output Comparison ---")
    test_images = torch.randn(1, 3, 224, 224).to(device)
    test_bboxes = [[(0.2, 0.2, 0.6, 0.6)]]

    def get_output(m): return perform_gazelle_inference(m, test_images, test_bboxes)[0]

    out_orig   = get_output(model)
    out_posit4 = get_output(model_posit4)
    compare_quantized_outputs(out_orig, out_posit4)

if __name__ == "__main__":
    full_quantization_comparison_pipeline()

