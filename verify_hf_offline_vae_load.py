import os


def main():
    # Force offline mode (no HTTP checks) for Hugging Face Hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # If anything tries to perform HTTP, make it fail loudly.
    try:
        import requests

        orig_request = requests.sessions.Session.request

        def _no_http(*args, **kwargs):
            raise RuntimeError("HTTP request attempted while HF_HUB_OFFLINE=1 (should not happen).")

        requests.sessions.Session.request = _no_http
    except Exception:
        orig_request = None

    try:
        from vae_module import FluxVAE

        _ = FluxVAE(local_files_only=True)
        print("OK: FluxVAE loaded with HF_HUB_OFFLINE=1 and local_files_only=True (no HTTP allowed).")
    finally:
        if orig_request is not None:
            requests.sessions.Session.request = orig_request


if __name__ == "__main__":
    main()

