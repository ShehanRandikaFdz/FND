"""
Performance Optimization for Credibility Analyzer
Optimizes speed and memory usage for real-time processing.
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc
import pickle
import joblib
from memory_profiler import profile
import warnings
warnings.filterwarnings('ignore')

from credibility_analyzer import CredibilityAnalyzer
from text_preprocessor import TextPreprocessor
from feature_extractor import AdvancedFeatureExtractor

class PerformanceOptimizer:
    """
    Optimizes performance of the credibility analyzer.
    """
    
    def __init__(self, analyzer: CredibilityAnalyzer):
        self.analyzer = analyzer
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = AdvancedFeatureExtractor()
        
        # Performance metrics
        self.performance_metrics = {
            'avg_prediction_time': 0.0,
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'throughput_per_second': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Caching system
        self.prediction_cache = {}
        self.cache_size_limit = 1000
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Batch processing queue
        self.batch_queue = queue.Queue()
        self.batch_size = 10
        self.batch_timeout = 1.0  # seconds
        
        # Model warmup flag
        self.models_warmed_up = False
    
    def warmup_models(self) -> Dict[str, float]:
        """Warm up models for faster inference."""
        print("🔥 Warming up models for optimal performance...")
        
        start_time = time.time()
        
        # Warmup texts
        warmup_texts = [
            "This is a test article for model warmup.",
            "Breaking news about technology developments.",
            "Scientists discover new treatment methods.",
            "Economic indicators show positive trends.",
            "Climate change research reveals new insights."
        ]
        
        warmup_times = []
        
        for text in warmup_texts:
            warmup_start = time.time()
            try:
                result = self.analyzer.analyze_credibility(text)
                warmup_time = time.time() - warmup_start
                warmup_times.append(warmup_time)
            except Exception as e:
                print(f"Warning: Warmup failed for text: {e}")
        
        total_warmup_time = time.time() - start_time
        avg_warmup_time = np.mean(warmup_times) if warmup_times else 0.0
        
        self.models_warmed_up = True
        
        print(f"✅ Models warmed up in {total_warmup_time:.2f}s (avg: {avg_warmup_time:.3f}s per prediction)")
        
        return {
            'total_warmup_time': total_warmup_time,
            'avg_warmup_time': avg_warmup_time,
            'warmup_samples': len(warmup_texts)
        }
    
    def benchmark_prediction_speed(self, test_texts: List[str], num_runs: int = 5) -> Dict[str, float]:
        """Benchmark prediction speed."""
        print(f"⚡ Benchmarking prediction speed with {len(test_texts)} texts, {num_runs} runs...")
        
        all_times = []
        
        for run in range(num_runs):
            run_times = []
            
            for text in test_texts:
                start_time = time.time()
                try:
                    result = self.analyzer.analyze_credibility(text)
                    prediction_time = time.time() - start_time
                    run_times.append(prediction_time)
                except Exception as e:
                    print(f"Warning: Prediction failed: {e}")
                    run_times.append(0.0)
            
            all_times.extend(run_times)
            
            # Clear cache between runs
            self.prediction_cache.clear()
        
        # Calculate statistics
        times_array = np.array(all_times)
        
        speed_metrics = {
            'avg_prediction_time': np.mean(times_array),
            'median_prediction_time': np.median(times_array),
            'min_prediction_time': np.min(times_array),
            'max_prediction_time': np.max(times_array),
            'std_prediction_time': np.std(times_array),
            'throughput_per_second': 1.0 / np.mean(times_array),
            'total_predictions': len(all_times)
        }
        
        print(f"✅ Speed benchmark completed:")
        print(f"   Average: {speed_metrics['avg_prediction_time']:.3f}s")
        print(f"   Throughput: {speed_metrics['throughput_per_second']:.1f} predictions/second")
        
        return speed_metrics
    
    def benchmark_memory_usage(self, test_texts: List[str]) -> Dict[str, float]:
        """Benchmark memory usage."""
        print("💾 Benchmarking memory usage...")
        
        process = psutil.Process()
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory usage during predictions
        memory_samples = []
        
        for i, text in enumerate(test_texts):
            # Measure memory before prediction
            memory_before = process.memory_info().rss / 1024 / 1024
            
            try:
                result = self.analyzer.analyze_credibility(text)
                
                # Measure memory after prediction
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_after - memory_before)
                
            except Exception as e:
                print(f"Warning: Memory benchmark failed: {e}")
            
            # Force garbage collection every 10 predictions
            if i % 10 == 0:
                gc.collect()
        
        # Calculate memory metrics
        memory_array = np.array(memory_samples)
        
        memory_metrics = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': process.memory_info().rss / 1024 / 1024,
            'avg_memory_per_prediction_mb': np.mean(memory_array),
            'max_memory_per_prediction_mb': np.max(memory_array),
            'memory_growth_mb': (process.memory_info().rss / 1024 / 1024) - initial_memory
        }
        
        print(f"✅ Memory benchmark completed:")
        print(f"   Initial: {memory_metrics['initial_memory_mb']:.1f} MB")
        print(f"   Peak: {memory_metrics['peak_memory_mb']:.1f} MB")
        print(f"   Growth: {memory_metrics['memory_growth_mb']:.1f} MB")
        
        return memory_metrics
    
    def benchmark_cpu_usage(self, test_texts: List[str], duration: float = 10.0) -> Dict[str, float]:
        """Benchmark CPU usage during predictions."""
        print(f"🖥️ Benchmarking CPU usage for {duration}s...")
        
        process = psutil.Process()
        cpu_samples = []
        
        start_time = time.time()
        text_index = 0
        
        while time.time() - start_time < duration:
            if text_index >= len(test_texts):
                text_index = 0
            
            # Measure CPU usage
            cpu_percent = process.cpu_percent()
            cpu_samples.append(cpu_percent)
            
            # Make prediction
            try:
                text = test_texts[text_index]
                result = self.analyzer.analyze_credibility(text)
            except Exception as e:
                print(f"Warning: CPU benchmark failed: {e}")
            
            text_index += 1
            time.sleep(0.1)  # Small delay to avoid overwhelming
        
        cpu_array = np.array(cpu_samples)
        
        cpu_metrics = {
            'avg_cpu_percent': np.mean(cpu_array),
            'max_cpu_percent': np.max(cpu_array),
            'min_cpu_percent': np.min(cpu_array),
            'std_cpu_percent': np.std(cpu_array),
            'samples_count': len(cpu_samples)
        }
        
        print(f"✅ CPU benchmark completed:")
        print(f"   Average: {cpu_metrics['avg_cpu_percent']:.1f}%")
        print(f"   Max: {cpu_metrics['max_cpu_percent']:.1f}%")
        
        return cpu_metrics
    
    def implement_prediction_caching(self, text: str, result: Dict[str, any]) -> None:
        """Implement caching for predictions."""
        # Create cache key (simple hash of text)
        cache_key = hash(text)
        
        # Check cache size limit
        if len(self.prediction_cache) >= self.cache_size_limit:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        # Store result in cache
        self.prediction_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def get_cached_prediction(self, text: str) -> Optional[Dict[str, any]]:
        """Get cached prediction if available."""
        cache_key = hash(text)
        
        if cache_key in self.prediction_cache:
            self.cache_hits += 1
            return self.prediction_cache[cache_key]['result']
        else:
            self.cache_misses += 1
            return None
    
    def batch_predict(self, texts: List[str], batch_size: int = 10) -> List[Dict[str, any]]:
        """Process multiple texts in batches for better throughput."""
        print(f"📦 Processing {len(texts)} texts in batches of {batch_size}...")
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_start = time.time()
            
            batch_results = []
            for text in batch:
                # Check cache first
                cached_result = self.get_cached_prediction(text)
                if cached_result:
                    batch_results.append(cached_result)
                else:
                    # Make new prediction
                    result = self.analyzer.analyze_credibility(text)
                    batch_results.append(result)
                    # Cache the result
                    self.implement_prediction_caching(text, result)
            
            batch_time = time.time() - batch_start
            results.extend(batch_results)
            
            print(f"   Batch {i//batch_size + 1}: {len(batch)} texts in {batch_time:.2f}s")
        
        return results
    
    def parallel_predict(self, texts: List[str], max_workers: int = 4) -> List[Dict[str, any]]:
        """Process texts in parallel using threading."""
        print(f"🔄 Processing {len(texts)} texts in parallel with {max_workers} workers...")
        
        def predict_single(text: str) -> Dict[str, any]:
            try:
                # Check cache first
                cached_result = self.get_cached_prediction(text)
                if cached_result:
                    return cached_result
                
                # Make new prediction
                result = self.analyzer.analyze_credibility(text)
                # Cache the result
                self.implement_prediction_caching(text, result)
                return result
            except Exception as e:
                return {'error': str(e)}
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(predict_single, texts))
        
        parallel_time = time.time() - start_time
        
        print(f"✅ Parallel processing completed in {parallel_time:.2f}s")
        print(f"   Throughput: {len(texts)/parallel_time:.1f} predictions/second")
        
        return results
    
    def optimize_text_preprocessing(self, texts: List[str]) -> List[str]:
        """Optimize text preprocessing for better performance."""
        print("🔧 Optimizing text preprocessing...")
        
        optimized_texts = []
        
        for text in texts:
            # Use faster preprocessing pipeline
            try:
                processed_text, _, _ = self.preprocessor.preprocess(text)
                optimized_texts.append(processed_text)
            except Exception as e:
                print(f"Warning: Preprocessing failed: {e}")
                optimized_texts.append(text)  # Use original text
        
        return optimized_texts
    
    def profile_memory_usage(self, text: str) -> Dict[str, float]:
        """Profile memory usage for a single prediction."""
        process = psutil.Process()
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Make prediction
        result = self.analyzer.analyze_credibility(text)
        
        # Final memory
        final_memory = process.memory_info().rss / 1024 / 1024
        
        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_delta_mb': final_memory - initial_memory
        }
    
    def run_comprehensive_performance_test(self, test_texts: List[str]) -> Dict[str, any]:
        """Run comprehensive performance testing."""
        print("🚀 Running Comprehensive Performance Test")
        print("=" * 60)
        
        # Warmup models
        warmup_results = self.warmup_models()
        
        # Benchmark speed
        speed_results = self.benchmark_prediction_speed(test_texts[:20])  # Use subset for speed
        
        # Benchmark memory
        memory_results = self.benchmark_memory_usage(test_texts[:10])  # Use subset for memory
        
        # Benchmark CPU
        cpu_results = self.benchmark_cpu_usage(test_texts[:5], duration=5.0)  # Shorter duration
        
        # Test batch processing
        print("\n📦 Testing batch processing...")
        batch_start = time.time()
        batch_results = self.batch_predict(test_texts[:20], batch_size=5)
        batch_time = time.time() - batch_start
        
        # Test parallel processing
        print("\n🔄 Testing parallel processing...")
        parallel_results = self.parallel_predict(test_texts[:20], max_workers=2)
        
        # Calculate cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0
        
        # Compile results
        performance_results = {
            'warmup_results': warmup_results,
            'speed_results': speed_results,
            'memory_results': memory_results,
            'cpu_results': cpu_results,
            'batch_processing': {
                'batch_time': batch_time,
                'batch_throughput': len(batch_results) / batch_time,
                'results_count': len(batch_results)
            },
            'parallel_processing': {
                'parallel_time': time.time() - time.time(),  # Will be calculated in parallel_predict
                'results_count': len(parallel_results)
            },
            'caching': {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': cache_hit_rate,
                'cache_size': len(self.prediction_cache)
            },
            'overall_performance_score': self._calculate_performance_score(
                speed_results, memory_results, cpu_results, cache_hit_rate
            )
        }
        
        return performance_results
    
    def _calculate_performance_score(self, speed_results: Dict[str, float], 
                                   memory_results: Dict[str, float],
                                   cpu_results: Dict[str, float],
                                   cache_hit_rate: float) -> float:
        """Calculate overall performance score."""
        # Weighted scoring based on importance
        speed_score = min(1.0, speed_results['throughput_per_second'] / 10.0)  # Target: 10 pred/s
        memory_score = max(0.0, 1.0 - memory_results['memory_growth_mb'] / 100.0)  # Target: <100MB growth
        cpu_score = max(0.0, 1.0 - cpu_results['avg_cpu_percent'] / 100.0)  # Target: <100% CPU
        cache_score = cache_hit_rate
        
        # Weighted average
        weights = {'speed': 0.4, 'memory': 0.3, 'cpu': 0.2, 'cache': 0.1}
        overall_score = (
            weights['speed'] * speed_score +
            weights['memory'] * memory_score +
            weights['cpu'] * cpu_score +
            weights['cache'] * cache_score
        )
        
        return overall_score
    
    def generate_optimization_recommendations(self, performance_results: Dict[str, any]) -> List[str]:
        """Generate optimization recommendations based on performance results."""
        recommendations = []
        
        # Speed recommendations
        speed_results = performance_results['speed_results']
        if speed_results['avg_prediction_time'] > 0.5:
            recommendations.append("Prediction time is high. Consider model optimization or caching.")
        
        if speed_results['throughput_per_second'] < 5:
            recommendations.append("Throughput is low. Consider batch processing or parallelization.")
        
        # Memory recommendations
        memory_results = performance_results['memory_results']
        if memory_results['memory_growth_mb'] > 50:
            recommendations.append("Memory usage is high. Consider garbage collection or model pruning.")
        
        # CPU recommendations
        cpu_results = performance_results['cpu_results']
        if cpu_results['avg_cpu_percent'] > 80:
            recommendations.append("CPU usage is high. Consider parallel processing or model optimization.")
        
        # Caching recommendations
        caching = performance_results['caching']
        if caching['cache_hit_rate'] < 0.3:
            recommendations.append("Cache hit rate is low. Consider increasing cache size or improving cache strategy.")
        
        # Overall performance
        overall_score = performance_results['overall_performance_score']
        if overall_score < 0.7:
            recommendations.append("Overall performance is below optimal. Consider comprehensive optimization.")
        
        if not recommendations:
            recommendations.append("Performance is optimal! No major optimizations needed.")
        
        return recommendations

def create_sample_test_texts() -> List[str]:
    """Create sample test texts for performance testing."""
    return [
        "Reuters reports that the Federal Reserve announced new interest rate policies today.",
        "BREAKING!!! You WON'T believe what scientists discovered! Doctors HATE this trick!",
        "According to a study published in Nature, researchers found evidence of water on Mars.",
        "URGENT: This shocking discovery will change everything! Government officials don't want you to know!",
        "The President signed the new healthcare bill today, according to official sources.",
        "Scientists develop new AI technology that could revolutionize medical diagnosis.",
        "Stock market reaches all-time high as investors show confidence in economic recovery.",
        "Climate change research reveals new insights into global warming patterns.",
        "Technology companies report record profits in the third quarter earnings.",
        "Police investigate suspicious activity in downtown area following reports.",
        "Young entrepreneurs launch innovative startup to address environmental challenges.",
        "Women leaders excel in crisis management, according to recent study findings.",
        "Immigration policies undergo comprehensive review by bipartisan committee.",
        "Religious communities unite to support local charity initiatives.",
        "Educational institutions adapt to new learning technologies and methods."
    ]

def main():
    """Test the performance optimizer."""
    print("🚀 Testing Performance Optimizer")
    print("=" * 50)
    
    # Initialize components
    analyzer = CredibilityAnalyzer()
    optimizer = PerformanceOptimizer(analyzer)
    
    # Create sample test texts
    test_texts = create_sample_test_texts()
    
    # Run comprehensive performance test
    results = optimizer.run_comprehensive_performance_test(test_texts)
    
    # Print summary
    print(f"\n📊 Performance Test Summary")
    print("=" * 60)
    
    print(f"🎯 Overall Performance Score: {results['overall_performance_score']:.3f}")
    
    print(f"\n⚡ Speed Results:")
    speed = results['speed_results']
    print(f"   Average Time: {speed['avg_prediction_time']:.3f}s")
    print(f"   Throughput: {speed['throughput_per_second']:.1f} pred/s")
    
    print(f"\n💾 Memory Results:")
    memory = results['memory_results']
    print(f"   Memory Growth: {memory['memory_growth_mb']:.1f} MB")
    print(f"   Peak Memory: {memory['peak_memory_mb']:.1f} MB")
    
    print(f"\n🖥️ CPU Results:")
    cpu = results['cpu_results']
    print(f"   Average CPU: {cpu['avg_cpu_percent']:.1f}%")
    print(f"   Max CPU: {cpu['max_cpu_percent']:.1f}%")
    
    print(f"\n📦 Caching Results:")
    caching = results['caching']
    print(f"   Cache Hit Rate: {caching['cache_hit_rate']:.1%}")
    print(f"   Cache Size: {caching['cache_size']}")
    
    print(f"\n💡 Optimization Recommendations:")
    recommendations = optimizer.generate_optimization_recommendations(results)
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")

if __name__ == "__main__":
    main()
