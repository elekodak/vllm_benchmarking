#!/usr/bin/env python3

"""
Multimodal Load Testing Script for OpenAI-compatible API

This script combines load testing functionality with a random dataset generator
for benchmarking multimodal models. It creates combinations of images and sends
requests at specified QPS for load testing.

Assumptions:
- Fireworks VLLM deployment is available.
- Image directory contains image files for testing

What it does:
- Creates random combinations of images from the specified directory
- Sends non-streaming requests at fixed QPS for a duration
- Logs per-request latency to JSONL
- Produces a compact summary with averages and P50/P90 for latency

Usage:
python multimodal_load_testing.py \
  --fireworks-api-key <your_api_key> \
  --model <model_name> \
  --system-prompt-file "../.data/QWEN_PROACTIVE_prompt.txt" \
  --image-dir ../.data/resized_images/ \
  --qps 10 \
  --total-requests 10000 \
  --outputs_path ../.data/results.jsonl
"""

from __future__ import annotations
import argparse
import asyncio
import json
import time
import base64
import io
import os
import random
import string
import logging
import statistics
from itertools import cycle, combinations, permutations
from typing import Any, Dict, Iterator, Optional, Tuple, List
from functools import partial
from openai import OpenAI
from PIL import Image
from torch.utils.data import IterableDataset
from tqdm.contrib.concurrent import thread_map
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("benchmark_debug.log")],
)

# Filter out HTTP client debug logs to prevent base64 image printing
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Constants
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
IMG_CACHE = {}


def fetch_image(params):
    """
    Dummy fetch_image function to replace the qwen_vl_utils dependency.
    In a real scenario, this would handle image resizing and loading.
    For this script, we'll just load the image from the file path.
    """
    image_path = params["image"].replace("file://", "")
    return Image.open(image_path)


def create_random_images_and_prompt_request(
    image_paths: List[str],
    model_name: str,
    temperature: float = 0,
    max_tokens: int = 128,
    logprobs: bool = False,
    top_logprobs: int = 0,
    system_prompt: str = "",
):
    """Create a request with random images and prompt."""
    encoded_image_urls = []
    for image_path in image_paths:
        if image_path in IMG_CACHE:
            encoded_image_urls.append(IMG_CACHE[image_path])
        else:
            # Load the image
            image = fetch_image({"image": "file://" + image_path})
            # Encode the image
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")
            image_bytes = image_bytes.getvalue()
            encoded_image_bytes = base64.b64encode(image_bytes).decode("utf-8")
            encoded_image_url = f"data:image/jpeg;base64,{encoded_image_bytes}"
            encoded_image_urls.append(encoded_image_url)
            IMG_CACHE[image_path] = encoded_image_url

    # Create a random string with random characters
    random_prompt = "".join(random.choices(string.ascii_letters + string.digits, k=10))

    # Create the request
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": random_prompt},
        ]
        + [
            {"type": "image_url", "image_url": {"url": encoded_image_url}}
            for encoded_image_url in encoded_image_urls
        ],
    })

    request = {
        "messages": messages,
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if logprobs:
        request["logprobs"] = True
        request["top_logprobs"] = top_logprobs

    return request


class LoadTestingRandomDataset(IterableDataset):
    """Dataset that creates random combinations of images for load testing."""

    def __init__(
        self,
        image_dir: str,
        model_name: str,
        total_requests: int,
        temperature: float = 0,
        max_tokens: int = 128,
        logprobs: bool = False,
        top_logprobs: int = 0,
        system_prompt: str = "",
        images_per_request: int = 4,
    ):
        self.image_dir = image_dir
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.images_per_request = images_per_request
        self.total_requests = total_requests

        # Get all image files from directory
        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))
        ]
        self.image_paths = [
            os.path.join(image_dir, image_path) for image_path in image_files
        ]
        # Create combinations of images
        image_combinations = list(
            combinations(self.image_paths, self.images_per_request)
        )

        self.image_paths = []
        for combo in image_combinations:
            for perm in permutations(combo):
                self.image_paths.append(list(perm))

        # Calculate repeat factor based on total_requests
        num_combinations = len(self.image_paths)
        if num_combinations > 0:
            repeat_factor = max(
                1, (self.total_requests + num_combinations - 1) // num_combinations
            )
            self.image_paths = repeat_factor * self.image_paths
            self.image_paths = self.image_paths[: self.total_requests]
        else:
            self.image_paths = []

        print(f"Total combinations: {len(self.image_paths)}")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

        # Prefetch all requests
        partial_create_request = partial(
            create_random_images_and_prompt_request,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            system_prompt=self.system_prompt,
        )

        print("Prefetching requests...")
        self.prefetched_requests = thread_map(
            partial_create_request,
            self.image_paths,
            max_workers=min(64, len(self.image_paths)),
            desc="Creating requests",
        )
        print(f"Prefetched {len(self.prefetched_requests)} requests")

    def __iter__(self):
        for request in cycle(self.prefetched_requests):
            yield request

    def __len__(self):
        return len(self.prefetched_requests)


def create_openai_client(api_key: str) -> OpenAI:
    """Create OpenAI client for Fireworks server."""
    return OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")


async def send_one_nonstream(
    client: OpenAI, payload: Dict[str, Any]
) -> Tuple[float, int, int, Optional[Dict[str, Any]]]:
    """Send one non-streaming request and return (latency_s, prompt_tokens, completion_tokens, full_response_json)."""
    logger.debug("Starting send_one_nonstream")
    logger.debug(f"Payload keys: {list(payload.keys())}")
    t_start = time.time()
    try:
        # Offload blocking client call to a thread to avoid blocking the event loop
        logger.debug("Calling client.chat.completions.create")
        resp = await asyncio.to_thread(client.chat.completions.create, **payload)
        latency_s = time.time() - t_start
        logger.debug(f"Request completed in {latency_s:.3f} seconds")
        resp_json = resp.model_dump()
        usage = resp_json.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        logger.debug(f"Response: prompt_tokens={pt}, completion_tokens={ct}")
        return latency_s, pt, ct, resp_json
    except Exception as e:
        import traceback

        logger.error(f"Exception in send_one_nonstream: {e}")
        traceback.print_exc()
        return 0.0, 0, 0, None


async def run_load(
    args: argparse.Namespace, dataset_iterator: Iterator[Dict[str, Any]]
) -> None:
    """Run the load test."""
    logger.info("Starting run_load function")
    logger.info("Creating OpenAI client for Fireworks AI endpoint")
    client = create_openai_client(args.fireworks_api_key)
    logger.info("OpenAI client created successfully")

    # Setup output file
    from pathlib import Path

    outputs_path = Path(args.outputs_path)
    logger.info(f"Setting up output file: {outputs_path}")
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    if outputs_path.exists():
        outputs_path.unlink()
        logger.info("Removed existing output file")

    # QPS scheduling
    interval = 1.0 / args.qps if args.qps > 0 else 0.0
    in_flight: set[asyncio.Task] = set()
    max_concurrency = (
        args.max_concurrency
        if args.max_concurrency is not None
        else max(1, int(args.qps * 10))
    )
    logger.info(f"Max concurrency set to: {max_concurrency}")
    logger.info(f"QPS interval: {interval}")
    logger.info(f"Total requests to send: {args.total_requests}")
    send_counter: int = 0
    start_time = time.perf_counter()
    requests_data = []

    async def rps_printer() -> None:
        nonlocal send_counter
        last_count = 0
        last_time = time.time()
        try:
            while True:
                await asyncio.sleep(1.0)
                current_time = time.time()
                elapsed = current_time - last_time
                sent_this_sec = send_counter - last_count
                rps = sent_this_sec / elapsed
                print(
                    f"[client] send RPS ~= {rps:.2f} (target {args.qps})",
                    flush=True,
                )
                last_count = send_counter
                last_time = current_time
        except asyncio.CancelledError:
            return

    async def handle_one(payload: Dict[str, Any], request_id: int) -> None:
        """Handle one request and log the response"""
        logger.debug("Starting handle_one function")
        request_start_time = time.perf_counter()
        try:
            logger.debug("Calling send_one_nonstream")
            latency_s, pt, ct, resp_json = await send_one_nonstream(client, payload)
            latency_ms = int(latency_s * 1000)
            ok = resp_json is not None
            
            logger.debug(
                f"send_one_nonstream completed: latency={latency_s}, pt={pt}, ct={ct}"
            )

            request_end_time = time.perf_counter()
            response_time_ms = (request_end_time - request_start_time) * 1000

            # Store data for summary
            requests_data.append(
                {
                    "request_id": request_id,
                    "ok": ok,
                    "latency_ms": latency_ms,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "input_throughput": pt / latency_s if latency_s > 0 else 0,
                    "output_throughput": ct / latency_s if latency_s > 0 else 0,
                    "total_throughput": ((pt + ct) / latency_s if latency_s > 0 else 0),
                    "response_time_ms": response_time_ms,
                }
            )

            # Write output line to JSONL
            logger.debug("Writing response to output file")
            with outputs_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "ok": ok,
                            "latency_ms": latency_ms,
                            "usage": {"prompt_tokens": pt, "completion_tokens": ct},
                            "input_throughput": pt / latency_s if latency_s > 0 else 0,
                            "output_throughput": ct / latency_s if latency_s > 0 else 0,
                            "total_throughput": (
                                (pt + ct) / latency_s if latency_s > 0 else 0
                            ),
                            "response": resp_json,
                        }
                    )
                    + "\n"
                )
            logger.debug("Response written to output file successfully")
        except Exception as e:
            import traceback
            logger.error(f"Exception in handle_one: {e}")
            traceback.print_exc()

    # Main send loop
    logger.info("Starting main send loop")
    rps_task = asyncio.create_task(rps_printer())
    try:
        while send_counter < args.total_requests:
            logger.debug(
                f"Main loop iteration: send_counter={send_counter}, total_requests={args.total_requests}"
            )
            try:
                # Get next payload from dataset iterator (should be fast since prefetched)
                logger.debug("Getting next payload from dataset")
                payload = next(dataset_iterator)
                logger.debug(f"Got payload: {type(payload)}")
                # Skip empty payloads
                if not payload:
                    logger.warning("Empty payload, skipping")
                    continue

            except StopIteration:
                logger.info("Completed Dataset Iterations")
                break
            except Exception as e:
                import traceback

                logger.error(f"Exception getting payload from dataset: {e}")
                traceback.print_exc()
                continue

            # Concurrency gating
            while len(in_flight) >= max_concurrency:
                logger.debug(
                    f"Concurrency limit reached ({len(in_flight)}/{max_concurrency}), waiting..."
                )
                done, in_flight = await asyncio.wait(
                    in_flight, return_when=asyncio.FIRST_COMPLETED
                )
                logger.debug("One request completed, continuing...")

            # Send request and track it
            logger.debug("Creating task for handle_one")
            send_counter += 1
            task = asyncio.create_task(handle_one(payload, send_counter))
            in_flight.add(task)
            task.add_done_callback(lambda t: in_flight.discard(t))
            
            logger.info(f"Sent request #{send_counter}, in_flight={len(in_flight)}")

            if interval > 0:
                # Sleep until next tick
                logger.debug(f"Sleeping for {interval} seconds")
                await asyncio.sleep(interval)

        # Drain remaining tasks
        if in_flight:
            print(f"Draining {len(in_flight)} remaining tasks...")
            await asyncio.gather(*in_flight, return_exceptions=True)
    finally:
        try:
            rps_task.cancel()
            await rps_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            import traceback
            print(f"Exception in cleanup: {e}")
            traceback.print_exc()
    
    end_time = time.perf_counter()
    duration_s = end_time - start_time
    
    # Generate and save summary
    if requests_data:
        df = pd.DataFrame(requests_data)
        
        # Latency calculations
        latency_values = df['latency_ms'].tolist()
        p50_latency = statistics.median(latency_values) if latency_values else 0
        p90_latency = statistics.quantiles(latency_values, n=10)[8] if len(latency_values) >= 10 else 0
        
        # Token calculations
        avg_prompt_tokens = df['prompt_tokens'].mean()
        avg_completion_tokens = df['completion_tokens'].mean()
        
        # Throughput calculations
        avg_input_throughput = df['input_throughput'].mean()
        avg_output_throughput = df['output_throughput'].mean()
        avg_total_throughput = df['total_throughput'].mean()
        
        # Overall metrics
        total_requests_completed = len(df)
        success_rate = (df['ok'].sum() / total_requests_completed) * 100 if total_requests_completed > 0 else 0
        
        summary = {
            "Total Requests": total_requests_completed,
            "Total Duration (s)": duration_s,
            "Success Rate (%)": success_rate,
            "Avg Latency (ms)": df['latency_ms'].mean(),
            "P50 Latency (ms)": p50_latency,
            "P90 Latency (ms)": p90_latency,
            "Avg Prompt Tokens": avg_prompt_tokens,
            "Avg Completion Tokens": avg_completion_tokens,
            "Avg Input Throughput (tokens/s)": avg_input_throughput,
            "Avg Output Throughput (tokens/s)": avg_output_throughput,
            "Avg Total Throughput (tokens/s)": avg_total_throughput,
        }
        
        # Save summary to CSV
        summary_df = pd.DataFrame([summary])
        csv_path = outputs_path.parent / (outputs_path.stem + "_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        print("\n--- Load Test Summary ---")
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to {csv_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multimodal load testing script for OpenAI-compatible API"
    )
    # Replaced host and port with Fireworks-specific arguments
    parser.add_argument(
        "--fireworks-api-key",
        required=True,
        help="API key for Fireworks AI",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Name of the model to use on Fireworks",
    )
    # The system prompt is now read from a file
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default="",
        help="Path to a file containing the system prompt",
    )
    parser.add_argument(
        "--qps", type=float, required=True, help="Target requests per second"
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        required=True,
        help="Total number of requests to send",
    )
    parser.add_argument(
        "--served-model-name",
        default="vision",
        help="Name used in OpenAI 'model' field",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing images for testing",
    )
    parser.add_argument(
        "--outputs_path",
        default="/tmp/mm_benchmark_outputs.jsonl",
        help="Per-request JSONL log path",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=None,
        help="Maximum concurrent requests (default: qps * 10)",
    )
    parser.add_argument(
        "--images-per-request",
        type=int,
        default=4,
        help="Number of images per request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--logprobs",
        action="store_true",
        help="Enable logprobs in response",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=0,
        help="Number of top logprobs to return",
    )
    return parser.parse_args()


def main() -> None:
    """Main function."""
    logger.info("Starting benchmark script")
    args = parse_args()
    logger.info(f"Parsed arguments: {args}")

    # Read system prompt from file
    system_prompt = ""
    if args.system_prompt_file:
        try:
            with open(args.system_prompt_file, "r") as f:
                system_prompt = f.read()
            logger.info("Successfully loaded system prompt from file")
        except FileNotFoundError:
            logger.error(f"Error: System prompt file not found at {args.system_prompt_file}")
            return
        except Exception as e:
            logger.error(f"Failed to read system prompt file: {e}")
            return

    # Create dataset
    logger.info("Creating dataset...")
    try:
        dataset = LoadTestingRandomDataset(
            image_dir=args.image_dir,
            model_name=args.model,
            total_requests=args.total_requests,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            logprobs=args.logprobs,
            top_logprobs=args.top_logprobs,
            system_prompt=system_prompt,
            images_per_request=args.images_per_request,
        )
        logger.info(f"Dataset created with {len(dataset)} prefetched requests")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        return

    dataset_iterator = iter(dataset)
    logger.info("Starting load test...")
    asyncio.run(run_load(args, dataset_iterator))
    logger.info("Load test completed")


if __name__ == "__main__":
    main()