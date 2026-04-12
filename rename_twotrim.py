import os
import shutil

repo_root = r"c:\Users\admin\Desktop\personal\compression-master"
exclude_dirs = {".git", "__pycache__", ".pytest_cache", ".venv", ".mypy_cache", ".VSCodeCounter", ".gemini", "benchmarks/data", ".system_generated"}
# don't mess with large binary/data files. We only care about code.
valid_extensions = {".py", ".md", ".json", ".toml", ".txt"}

replacements = {
    "twotrim": "twotrim",
    "TwoTrim": "TwoTrim",
    "TWOTRIM": "TWOTRIM",
    "Twotrim": "Twotrim",
}

print("Starting global string replacement...")
files_modified = 0

for root, dirs, files in os.walk(repo_root):
    # filter out excluded dirs
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    
    for file in files:
        if not any(file.endswith(ext) for ext in valid_extensions):
            continue
            
        file_path = os.path.join(root, file)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            continue
            
        original_content = content
        
        for old, new in replacements.items():
            content = content.replace(old, new)
            
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            files_modified += 1
            print(f"Modified: {file_path}")

print(f"Global string replacement complete. Modified {files_modified} files.")

old_dir = os.path.join(repo_root, "src", "twotrim")
new_dir = os.path.join(repo_root, "src", "twotrim")

if os.path.exists(old_dir):
    print(f"Renaming directory {old_dir} to {new_dir}...")
    os.rename(old_dir, new_dir)
    print("Directory renamed successfully.")
else:
    print(f"Directory {old_dir} not found. Skipped.")
    
print("Rename operation finished!")
