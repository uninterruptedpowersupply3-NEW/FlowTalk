#!/usr/bin/env python3
"""
Precompute VAE latents + text tokens into sharded cache files.
"""

import argparse
import time

from encoder_backend import (
    EncoderRuntimeConfig,
    FastMultimodalEncoder,
    autodetect_loader_workers,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-throughput multimodal precompute (images + tokens)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing image/text sidecars")
    parser.add_argument("--cache-dir", default=".latent_cache", help="Output shard cache directory")
    parser.add_argument("--batch-size", type=int, default=32, help="GPU encoding batch size")
    parser.add_argument("--max-size", type=int, default=256, help="Square resize size for VAE input")
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=512,
        help="Maximum prompt token length before truncation; cached tokens remain unpadded",
    )
    parser.add_argument("--keep-aspect-ratio", dest="keep_aspect_ratio", action="store_true", help="Preserve image aspect ratio with letterbox padding")
    parser.add_argument("--no-keep-aspect-ratio", dest="keep_aspect_ratio", action="store_false", help="Force square resize (distorts aspect ratio)")
    parser.set_defaults(keep_aspect_ratio=True)
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers (auto if omitted)")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Worker prefetch factor")
    parser.add_argument("--text-threads", type=int, default=8, help="Threads for tokenizer batch encode")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on VAE encoder")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for VAE encoder",
    )
    parser.add_argument("--use-cuda-graphs", action="store_true", help="Enable CUDA graph replay for fixed-size batches")
    parser.add_argument("--graph-batch-size", type=int, default=64, help="Static batch size used for CUDA graph capture")
    parser.add_argument(
        "--latent-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Stored latent dtype",
    )
    parser.add_argument(
        "--token-dtype",
        type=str,
        default="int32",
        choices=["int32", "int64"],
        help="Stored token dtype",
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=2.0,
        help="Shard rotation threshold in GB",
    )
    parser.add_argument("--flush-every", type=int, default=5000, help="Flush index every N encoded images")
    args = parser.parse_args()

    workers = autodetect_loader_workers(args.workers)
    shard_bytes = int(args.shard_size_gb * 1024 * 1024 * 1024)

    cfg = EncoderRuntimeConfig(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        max_size=args.max_size,
        max_text_length=args.max_text_length,
        keep_aspect_ratio=args.keep_aspect_ratio,
        num_workers=workers,
        prefetch_factor=args.prefetch_factor,
        use_compile=args.compile,
        compile_mode=args.compile_mode,
        use_cuda_graphs=args.use_cuda_graphs,
        cuda_graph_batch_size=args.graph_batch_size,
        text_threads=args.text_threads,
        latent_dtype=args.latent_dtype,
        token_dtype=args.token_dtype,
        shard_max_bytes=shard_bytes,
        flush_every=args.flush_every,
    )

    print("=" * 72)
    print("FAST MULTIMODAL PRECOMPUTE")
    print("=" * 72)
    print(f"Data Dir           : {args.data_dir}")
    print(f"Cache Dir          : {args.cache_dir}")
    print(f"Batch Size         : {args.batch_size}")
    print(f"Image Size         : {args.max_size}")
    print(f"Max Text Length    : {args.max_text_length} (unpadded)")
    print(f"Keep Aspect Ratio  : {args.keep_aspect_ratio}")
    print(f"Workers            : {workers}")
    print(f"Text Threads       : {args.text_threads}")
    print(f"Compile            : {args.compile} ({args.compile_mode})")
    print(f"CUDA Graphs        : {args.use_cuda_graphs} (bs={args.graph_batch_size})")
    print(f"Store Dtypes       : latent={args.latent_dtype}, token={args.token_dtype}")
    print(f"Shard Size         : {args.shard_size_gb:.2f} GB")
    print("=" * 72)

    t0 = time.time()
    encoder = FastMultimodalEncoder(cfg)
    stats = encoder.precompute_directory(args.data_dir)
    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print("PRECOMPUTE COMPLETE")
    print("=" * 72)
    print(f"Total images found : {stats['total_files']:,}")
    print(f"Encoded this run   : {stats['encoded']:,}")
    print(f"Skipped cached     : {stats['skipped']:,}")
    print(f"Cache total        : {stats['cached_total']:,}")
    print(f"Runtime            : {elapsed / 60.0:.2f} min")
    print(f"Throughput         : {stats['rate']:.2f} img/s")
    print("=" * 72)


if __name__ == "__main__":
    main()
