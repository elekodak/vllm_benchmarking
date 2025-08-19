# vLLM Benchmark Script

A comprehensive benchmarking tool for testing the performance of vision-language models (VLMs) that can process both text and image inputs. This script measures key performance metrics including latency, throughput, and Time to First Token (TTFT) across various concurrency levels and batch sizes.

## Features

- **Multi-modal Testing**: Supports benchmarking with both text and image inputs
- **Concurrent Load Testing**: Tests multiple concurrency levels simultaneously
- **Comprehensive Metrics**: Measures latency percentiles, throughput, QPS, and TTFT
- **Flexible Configuration**: Customizable batch sizes, image counts, and system prompts
- **CSV Export**: Automatically exports detailed results to CSV for analysis
- **Streaming Support**: Handles streaming responses and measures TTFT accurately

## Requirements

### Python Dependencies

```bash
pip install requests tqdm
```

### Required Files and Directories

1. **System Prompt File**: A text file containing the system prompt
2. **Image Directory**: A folder containing test images in supported formats
3. **API Endpoint**: A running vLLM server with vision model support

### Supported Image Formats

- JPG/JPEG
- PNG
- GIF
- BMP

## Usage

### Basic Command

```bash
python vllm_benchmark.py \
    --endpoint "http://localhost:8000/v1/chat/completions" \
    --api_key "your-api-key" \
    --model_name "llava-v1.5-7b" \
    --system_prompt_file "prompt.txt" \
    --image_dir "/path/to/images"
```

### Advanced Usage

```bash
python vllm_benchmark.py \
    --endpoint "http://localhost:8000/v1/chat/completions" \
    --api_key "your-api-key" \
    --model_name "llava-v1.5-7b" \
    --system_prompt_file "prompt.txt" \
    --image_dir "/path/to/images" \
    --concurrency 1 4 8 16 32 \
    --num_images 2
```

## Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--endpoint` | string | Yes | Complete API endpoint URL |
| `--api_key` | string | Yes | API key for authentication |
| `--model_name` | string | Yes | Model name to benchmark |
| `--system_prompt_file` | string | Yes | Path to system prompt text file |
| `--image_dir` | string | Yes | Directory containing test images |
| `--concurrency` | int list | No | Concurrency levels to test (default: [16]) |
| `--num_images` | int | No | Images per request: 1-4 (default: 1) |

## Output Metrics

The script measures and reports the following metrics:

### Performance Metrics
- **Throughput**: Tokens generated per second
- **QPS**: Queries (requests) processed per second
- **Total Duration**: Complete benchmark execution time

### Latency Metrics (in milliseconds)
- **Average Latency**: Mean response time
- **P50 Latency**: 50th percentile (median)
- **P90 Latency**: 90th percentile
- **P99 Latency**: 99th percentile

### Time to First Token (TTFT) Metrics (in milliseconds)
- **Average TTFT**: Mean time to first token
- **P50 TTFT**: 50th percentile TTFT
- **P90 TTFT**: 90th percentile TTFT
- **P99 TTFT**: 99th percentile TTFT

## Benchmark Configuration

### Default Batch Sizes
The script automatically tests these batch sizes:
```
4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
```

### Request Configuration
- **Max Tokens**: 20 tokens per response
- **Temperature**: 1e-6 (deterministic responses)
- **Timeout**: 600 seconds per request

## File Structure Example

```
project/
├── vllm_benchmark.py
├── system_prompt.txt
├── images/
│   ├── image1.jpg
│   ├── image2.png
│   └── image3.jpeg
└── benchmark_summary.csv (generated)
```

## Sample System Prompt File

Create a `system_prompt.txt` file with your desired prompt:

```
You are a helpful AI assistant that can analyze images. Please describe what you see in the provided image(s) in detail.
```

## Output Files

The script generates `benchmark_summary.csv` with columns:

- `batch_size`: Number of requests in the batch
- `concurrency`: Number of concurrent workers
- `num_images_per_request`: Images included per request
- `requests_count`: Total completed requests
- `total_duration_s`: Total execution time
- `throughput_tokens_per_s`: Token generation rate
- `qps`: Requests per second
- `avg_latency_ms`: Average latency
- `p50_latency_ms`: 50th percentile latency
- `p90_latency_ms`: 90th percentile latency
- `p99_latency_ms`: 99th percentile latency
- `avg_ttft_ms`: Average time to first token
- `p50_ttft_ms`: 50th percentile TTFT
- `p90_ttft_ms`: 90th percentile TTFT
- `p99_ttft_ms`: 99th percentile TTFT

## Example Output

```
--- Running Benchmark: Batch Size 16, Concurrency 8, Images/Request 1 ---
Processing requests: 100%|██████████| 16/16 [00:12<00:00,  1.31it/s]

Requests Completed: 16
Total Duration: 12.34 s
Throughput: 25.93 tokens/s
QPS: 1.30 requests/s
Average Latency: 6142.50 ms
P50 Latency: 6050.25 ms
P90 Latency: 7200.15 ms
P99 Latency: 7800.45 ms
Average TTFT: 1250.30 ms
P50 TTFT: 1200.15 ms
P90 TTFT: 1450.25 ms
P99 TTFT: 1650.80 ms
```

## Troubleshooting

### Common Issues

1. **No images found**: Ensure your image directory contains supported image formats
2. **Connection errors**: Verify the endpoint URL and that the vLLM server is running
3. **Authentication errors**: Check that your API key is correct
4. **Timeout errors**: Large batch sizes or high concurrency may exceed the 600s timeout

### Performance Tips

- Start with lower concurrency levels and smaller batch sizes
- Ensure sufficient system resources for the target concurrency
- Monitor server resources during benchmarking
- Use a dedicated benchmarking environment for consistent results

## Integration with vLLM

This script is designed to work with vLLM servers running vision-language models. Ensure your vLLM server is configured with:

- Vision model support enabled
- Sufficient GPU memory for concurrent requests
- Appropriate model configuration for your use case

## Analysis

Use the generated CSV file with data analysis tools like:

- **Python**: pandas, matplotlib for visualization
- **Excel**: For basic analysis and charting
- **Jupyter Notebooks**: For detailed performance analysis

The metrics help identify optimal configurations for your specific use case and infrastructure constraints.
