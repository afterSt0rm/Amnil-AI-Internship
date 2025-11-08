import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import GPUtil
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import requests


class APIBenchmark:
    def __init__(self, base_url: str, test_image_path: str):
        self.base_url = base_url
        self.test_image_path = test_image_path
        self.results = []

    def single_request(self):
        """Perform single request and measure latency"""
        try:
            with open(self.test_image_path, "rb") as f:
                files = {"file": ("test.jpg", f, "image/jpeg")}
                start_time = time.time()
                response = requests.post(f"{self.base_url}/predict", files=files)
                end_time = time.time()

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "status": "success",
                        "total_time": (end_time - start_time) * 1000,  # ms
                        "inference_time": result["inference_time_ms"],
                        "response_size": len(response.content),
                    }
                else:
                    return {"status": "error", "error": response.text}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def latency_test(self, num_requests: int = 100):
        """Test latency with multiple sequential requests"""
        print(f"Running latency test with {num_requests} requests...")

        latencies = []
        inference_times = []
        successes = 0

        for i in range(num_requests):
            result = self.single_request()
            if result["status"] == "success":
                latencies.append(result["total_time"])
                inference_times.append(result["inference_time"])
                successes += 1

            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_requests} requests")

        if successes > 0:
            stats = {
                "total_requests": num_requests,
                "successful_requests": successes,
                "success_rate": successes / num_requests,
                "avg_latency": statistics.mean(latencies),
                "avg_inference_time": statistics.mean(inference_times),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "p95_latency": statistics.quantiles(latencies, n=20)[
                    18
                ],  # 95th percentile
                "std_latency": statistics.stdev(latencies),
            }

            print("\nLatency Test Results:")
            for key, value in stats.items():
                print(f"{key}: {value}")

            return stats
        else:
            print("No successful requests")
            return None

    def throughput_test(self, concurrent_users: int, duration: int = 30):
        """Test throughput with concurrent users"""
        print(
            f"Running throughput test with {concurrent_users} concurrent users for {duration}s..."
        )

        self._throughput_results = []
        self._stop_event = threading.Event()

        def worker(worker_id):
            while not self._stop_event.is_set():
                result = self.single_request()
                if result["status"] == "success":
                    self._throughput_results.append(
                        {
                            "worker_id": worker_id,
                            "timestamp": time.time(),
                            "latency": result["total_time"],
                        }
                    )

        # Start workers
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(worker, i) for i in range(concurrent_users)]

            # Run for specified duration
            time.sleep(duration)
            self._stop_event.set()

        end_time = time.time()

        # Calculate throughput
        total_time = end_time - start_time
        total_requests = len(self._throughput_results)
        rps = total_requests / total_time

        latencies = [r["latency"] for r in self._throughput_results]

        stats = {
            "concurrent_users": concurrent_users,
            "duration": duration,
            "total_requests": total_requests,
            "requests_per_second": rps,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "min_latency": min(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
        }

        print("\nThroughput Test Results:")
        for key, value in stats.items():
            print(f"{key}: {value}")

        return stats

    def load_test(self, user_pattern: list):
        """Test with varying load patterns"""
        print("Running load test with varying user patterns...")

        all_results = []

        for concurrent_users in user_pattern:
            print(f"\nTesting with {concurrent_users} concurrent users...")
            result = self.throughput_test(concurrent_users, duration=10)
            if result:
                result["concurrent_users"] = concurrent_users
                all_results.append(result)

            time.sleep(2)  # Cool down period

        return all_results

    def get_system_metrics(self):
        """Collect system resource usage"""
        try:
            response = requests.get(f"{self.base_url}/system-status")
            if response.status_code == 200:
                return response.json()["system_metrics"]
        except:
            return None

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite"""
        print("Starting comprehensive benchmark...")

        benchmark_results = {}

        # 1. Single request baseline
        print("\n1. Single Request Baseline")
        single_result = self.single_request()
        benchmark_results["single_request"] = single_result

        # 2. Latency test
        print("\n2. Latency Test")
        latency_results = self.latency_test(num_requests=50)
        benchmark_results["latency"] = latency_results

        # 3. Throughput tests
        print("\n3. Throughput Tests")
        throughput_results = []
        for users in [1, 5, 10, 20]:
            result = self.throughput_test(users, duration=15)
            if result:
                throughput_results.append(result)
        benchmark_results["throughput"] = throughput_results

        # 4. Load pattern test
        print("\n4. Load Pattern Test")
        load_results = self.load_test([1, 5, 10, 15, 20, 10, 5])
        benchmark_results["load_pattern"] = load_results

        # 5. System metrics
        print("\n5. System Metrics")
        system_metrics = self.get_system_metrics()
        benchmark_results["system_metrics"] = system_metrics

        # Save results
        with open("benchmark_results.json", "w") as f:
            json.dump(benchmark_results, f, indent=2)

        self.generate_report(benchmark_results)

        return benchmark_results

    def generate_report(self, results):
        """Generate a comprehensive benchmark report"""
        print("\n" + "=" * 50)
        print("BENCHMARK REPORT")
        print("=" * 50)

        if "latency" in results and results["latency"]:
            latency = results["latency"]
            print(f"\n📊 Latency Performance:")
            print(f"   Average Latency: {latency['avg_latency']:.2f}ms")
            print(f"   P95 Latency: {latency['p95_latency']:.2f}ms")
            print(f"   Success Rate: {latency['success_rate']:.1%}")

        if "throughput" in results:
            print(f"\n🚀 Throughput Performance:")
            for test in results["throughput"]:
                print(
                    f"   {test['concurrent_users']} users: {test['requests_per_second']:.2f} RPS"
                )

        if "system_metrics" in results and results["system_metrics"]:
            metrics = results["system_metrics"]
            print(f"\n💻 System Resources:")
            print(f"   CPU Usage: {metrics.get('cpu_percent', 'N/A')}%")
            print(f"   Memory Usage: {metrics.get('memory_percent', 'N/A')}%")

            if metrics.get("gpu_metrics"):
                for gpu_id, gpu in metrics["gpu_metrics"].items():
                    print(
                        f"   GPU {gpu_id}: {gpu['load']:.1f}% load, {gpu['memory_used']}MB used"
                    )


if __name__ == "__main__":
    # Initialize benchmark
    benchmark = APIBenchmark(
        base_url="http://localhost:8001",
        test_image_path="tests/Cherry Powdery Mildew.jpg",
    )

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
