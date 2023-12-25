import os

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

directories = ["videos/input", "videos/output"]

for directory in directories:
    create_directory(directory)

print("Setup complete.")
