import argparse
import time
import os
import random
import json
import csv
import base64
from glob import glob
from tqdm import tqdm
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

def get_image_paths(image_dir):
    """Retrieves a list of image paths from a given directory."""
    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
    paths = []
    for ext in image_extensions:
        paths.extend(glob(os.path.join(image_dir, f'*.{ext}')))
    if not paths:
        raise FileNotFoundError(f"No images found in the directory: {image_dir}")
    return paths

def generate_random_token(length=10):
    """Generates a random alphanumeric token."""
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def create_payload(system_prompt_path, image_dir, model_name, num_images, num_tokens=10):
    """
    Creates the JSON payload for a single API request with multiple images.
    """
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read().strip()
    
    random_tokens = [generate_random_token() for _ in range(num_tokens)]
    token_str = ' '.join(random_tokens)
    
    combined_prompt = f"{system_prompt} {token_str}"
    
    # Get all available image paths
    all_image_paths = get_image_paths(image_dir)
    
    # Select a random sample of images to include in this payload
    selected_image_paths = random.sample(all_image_paths, min(num_images, len(all_image_paths)))
    
    # Create the content list with text and image parts
    content_list = [{"type": "text", "text": combined_prompt}]
    for img_path in selected_image_paths:
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}
        })
    
    messages = [
        {"role": "user", "content": content_list}
    ]
    
    return {
        "model": model_name,
        "messages": messages,
        "max_tokens": 20,
        "temperature": 1e-6,
    }

def send_request(full_endpoint_url, api_key, payload):
    """
    Sends a single API request to the configurable endpoint and returns latency and TTFT.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    req_start_time = time.time()
    try:
        response = requests.post(full_endpoint_url, headers=headers, json=payload, stream=True, timeout=600)
        
        full_response = ""
        ttft = 0
        ttft_measured = False
        
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                try:
                    data_str = chunk.decode('utf-8').strip()
                    if data_str.startswith("data:"):
                        data_str = data_str[5:]
                    
                    if data_str and data_str != "[DONE]":
                        data = json.loads(data_str)
                        if not ttft_measured and data['choices'][0]['delta'].get('content'):
                            ttft = time.time() - req_start_time
                            ttft_measured = True
                        
                        content = data['choices'][0]['delta'].get('content', '')
                        full_response += content

                except (json.JSONDecodeError, KeyError):
                    continue
        
        latency = time.time() - req_start_time
        return latency, ttft
        
    except RequestException as e:
        print(f"Request to {full_endpoint_url} failed: {e}")
        return None, None

def run_concurrent_benchmark(api_endpoint, endpoint_path, api_key, model_name, batch_size, concurrency, num_images_per_request, system_prompt_path, image_dir):
    """
    Executes a benchmark run with a fixed concurrency level.
    """
    full_endpoint_url = f"{api_endpoint}/{endpoint_path}"
    print(f"--- Running Benchmark: Batch Size {batch_size}, Concurrency {concurrency}, Images/Request {num_images_per_request} ---")
    
    all_payloads = []
    for _ in range(batch_size):
        payload = create_payload(system_prompt_path, image_dir, model_name, num_images_per_request)
        all_payloads.append(payload)

    latencies = []
    ttfts = []
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, full_endpoint_url, api_key, payload) for payload in all_payloads]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing requests"):
            latency, ttft = future.result()
            if latency is not None:
                latencies.append(latency)
            if ttft is not None:
                ttfts.append(ttft)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    if not latencies:
        return {}

    total_requests = len(latencies)
    total_output_tokens = total_requests * 20
    
    latencies.sort()
    ttfts.sort()
    
    metrics = {
        'batch_size': batch_size,
        'concurrency': concurrency,
        'num_images_per_request': num_images_per_request,
        'requests_count': total_requests,
        'total_duration_s': total_duration,
        'throughput_tokens_per_s': total_output_tokens / total_duration if total_duration > 0 else 0,
        'qps': total_requests / total_duration if total_duration > 0 else 0,
        'avg_latency_s': sum(latencies) / len(latencies),
        'p50_latency_s': latencies[int(total_requests * 0.50)],
        'p90_latency_s': latencies[int(total_requests * 0.90)],
        'p99_latency_s': latencies[int(total_requests * 0.99)],
        'avg_ttft_s': sum(ttfts) / len(ttfts) if ttfts else 0,
        'p50_ttft_s': ttfts[int(total_requests * 0.50)] if ttfts else 0,
        'p90_ttft_s': ttfts[int(total_requests * 0.90)] if ttfts else 0,
        'p99_ttft_s': ttfts[int(total_requests * 0.99)] if ttfts else 0,
    }
    
    print_stats(metrics)
    return metrics

def print_stats(metrics):
    """Prints benchmark statistics to the console."""
    print(f"\nRequests Completed: {metrics['requests_count']}")
    print(f"Total Duration: {metrics['total_duration_s']:.2f} s")
    print(f"Throughput: {metrics['throughput_tokens_per_s']:.2f} tokens/s")
    print(f"QPS: {metrics['qps']:.2f} requests/s")
    print(f"Average Latency: {metrics['avg_latency_s']:.2f} s")
    print(f"P50 Latency: {metrics['p50_latency_s']:.2f} s")
    print(f"P90 Latency: {metrics['p90_latency_s']:.2f} s")
    print(f"P99 Latency: {metrics['p99_latency_s']:.2f} s")
    if metrics['avg_ttft_s'] > 0:
        print(f"Average TTFT: {metrics['avg_ttft_s']:.2f} s")
        print(f"P50 TTFT: {metrics['p50_ttft_s']:.2f} s")
        print(f"P90 TTFT: {metrics['p90_ttft_s']:.2f} s")
        print(f"P99 TTFT: {metrics['p99_ttft_s']:.2f} s")

def main():
    parser = argparse.ArgumentParser(description="Benchmark a model API with a specific workload.")
    parser.add_argument("--api_endpoint", type=str, required=True, help="The base API endpoint URL (e.g., https://api.fireworks.ai).")
    parser.add_argument("--endpoint_path", type=str, default="v1/chat/completions", choices=["v1/chat/completions", "v1/completions"], help="The API endpoint path.")
    parser.add_argument("--api_key", type=str, required=True, help="The API key for authentication.")
    parser.add_argument("--model_name", type=str, required=True, help="The model name to use for the benchmark.")
    parser.add_argument("--system_prompt_file", type=str, required=True, help="Path to the system prompt text file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--concurrency", nargs='+', type=int, default=[16], help="A list of concurrency levels to test.")
    parser.add_argument("--num_images", type=int, default=1, choices=[1, 2, 3, 4], help="The number of images to include in each request.")
    
    args = parser.parse_args()

    # The original log did not specify multi-image requests, but this is how you would add them.
    # The image path is picked at random. Ensure your image_dir has at least 4 images.
    batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    output_csv_path = "benchmark_summary.csv"
    
    with open(output_csv_path, 'w', newline='') as f:
        fieldnames = ['batch_size', 'concurrency', 'num_images_per_request', 'requests_count', 'total_duration_s', 'throughput_tokens_per_s', 'qps', 'avg_latency_s', 'p50_latency_s', 'p90_latency_s', 'p99_latency_s', 'avg_ttft_s', 'p50_ttft_s', 'p90_ttft_s', 'p99_ttft_s']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for concurrency_level in sorted(args.concurrency):
            for batch_size in batch_sizes:
                metrics = run_concurrent_benchmark(
                    api_endpoint=args.api_endpoint,
                    endpoint_path=args.endpoint_path,
                    api_key=args.api_key,
                    model_name=args.model_name,
                    batch_size=batch_size,
                    concurrency=concurrency_level,
                    num_images_per_request=args.num_images,
                    system_prompt_path=args.system_prompt_file,
                    image_dir=args.image_dir
                )
                if metrics:
                    writer.writerow(metrics)
                print("\n" + "-"*50 + "\n")

    print(f"Summary statistics written to {output_csv_path}")

if __name__ == "__main__":
    main()