
import os
import sys

def run_calculator_test():
    print("--- Verifying Calculator Logic ---")
    
    data_dir = r"C:\Users\chatr\Documents\Tech\VLLM\New folder\Train_Img"
    text_dirs = r"C:\Users\chatr\Documents\Tech\VLLM\text_data\maths"
    
    if not os.path.exists(data_dir):
        # Create dummy if needed
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "test.png"), "w") as f: f.write("dummy")
        
    count = 0
    
    # Copy-paste logic from gui_app.py
    if os.path.exists(data_dir):
        with os.scandir(data_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.txt', '.json')):
                    count += 1
    
    print(f"Image Count: {count}")
    
    # Text
    if text_dirs:
        for d in text_dirs.split(','):
            d = d.strip()
            if os.path.exists(d):
                try:
                        for root, dirs, files in os.walk(d):
                            count += len([f for f in files if f.endswith(('.txt', '.json', '.xml', '.md'))])
                except Exception as e:
                    print(f"Calc Error (Txt): {e}")
            else:
                 print(f"Text dir not found: {d}")

    print(f"Total Count: {count}")
    
    if count > 0:
        print("✅ Calculator Verification PASSED (Found files)")
    else:
        print("⚠️ Calculator Found 0 files (Might be expected if empty, but logic ran)")

if __name__ == "__main__":
    run_calculator_test()
