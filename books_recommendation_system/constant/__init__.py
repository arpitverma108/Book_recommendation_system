import os

ROOT_DIR = os.getcwd()
# Main config file path
CONFIG_FOLDER_NAME = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_FOLDER_NAME,CONFIG_FILE_NAME)
print("CONFIG_FILE_PATH:", CONFIG_FILE_PATH)  # 👈 Full path to config.yaml

# Optional: Check if the file actually exists
if os.path.exists(CONFIG_FILE_PATH):
    print("✅ Config file exists!")
else:
    print("❌ Config file NOT found.")