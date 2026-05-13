import sys

def check_imports():
    results = []
    checks = [
        ("torch",         "torch"),
        ("cv2",           "opencv-python"),
        ("PIL",           "Pillow"),
        ("numpy",         "numpy"),
        ("openai",        "openai"),
        ("supervision",   "supervision"),
        ("dotenv",        "python-dotenv"),
    ]

    for module, pkg in checks:
        try:
            __import__(module)
            results.append((pkg, " PRESENT "))
        except ImportError:
            results.append(pkg, " MISSING MAN")

    print("\n Dependency Sanity Check ")
    for pkg, status in results:
        print(f"  {status}   {pkg}")

    #confidence check for the api keys
    print("\n=== Azure OpenAI Config ===")
    from config import (
        AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION
    )
    fields = {
        "ENDPOINT":   AZURE_OPENAI_ENDPOINT,
        "API_KEY":    AZURE_OPENAI_API_KEY,
        "DEPLOYMENT": AZURE_OPENAI_DEPLOYMENT,
        "API_VERSION":AZURE_OPENAI_API_VERSION,
    }
    for k, v in fields.items():
        status = "✓" if v else "✗ NOT SET"
        print(f"  {status}  {k}")

        #grounding dino
        print("\n=== Grounding DINO ===")
        try:
            from groundingdino.util.inference import load_model
            print("  ✓  groundingdino importable")
        except ImportError:
            print("  ⚠  groundingdino not installed yet (install in Step 5)")

if __name__ == "__main__":
    check_imports()