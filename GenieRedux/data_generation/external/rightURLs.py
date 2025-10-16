import os
import re

def fix_urls_file(file_path):
    """Fix the URLs file to remove duplicate parts both before and after the pipe"""
    print(f"Fixing file: {file_path}")
    
    # Read the original file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process the lines
    fixed_lines = []
    for line in lines:
        line = line.strip()
        
        # Skip header lines and separator lines
        if line.startswith('#') or line.startswith('=') or not line:
            fixed_lines.append(line)
            continue
        
        # Fix the problematic lines
        if '|' in line:
            parts = line.split('|')
            if len(parts) == 2:
                # Extract just the game name from the first part
                # The first part looks like: //archive.org/download/ni-roms/roms/Atari - 2600.zip/3-D Genesis
                # We want to extract just "3-D Genesis"
                first_part = parts[0]
                url_part = parts[1]
                
                # Extract the game name (everything after the last '/')
                if '/' in first_part:
                    game_name = first_part.split('/')[-1]
                else:
                    game_name = first_part
                
                # Clean up the URL part - remove the duplicate prefix
                # URL looks like: https://archive.org/download/ni-roms/roms/Atari%20-%202600.zip///archive.org/download/ni-roms/roms/Atari%20-%202600.zip/Journey%20Escape%20%28USA%29.zip
                # We want: https://archive.org/download/ni-roms/roms/Atari%20-%202600.zip/Journey%20Escape%20%28USA%29.zip
                if '///archive.org/download/' in url_part:
                    # Split and take the part after the triple slash
                    url_parts = url_part.split('///archive.org/download/')
                    if len(url_parts) > 1:
                        clean_url = f"https://archive.org/download/{url_parts[1]}"
                    else:
                        clean_url = url_part
                else:
                    clean_url = url_part
                
                # Write the fixed line
                fixed_line = f"{game_name}|{clean_url}"
                fixed_lines.append(fixed_line)
            else:
                # Keep the line as is if it doesn't have exactly 2 parts
                fixed_lines.append(line)
        else:
            # Keep lines without pipes as is
            fixed_lines.append(line)
    
    # Write the fixed content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + '\n')
    
    print(f"✓ Fixed {file_path}")

def fix_all_urls_files(directory_path):
    """Fix all URLs files in the directory"""
    print(f"Looking for URLs files in: {directory_path}")
    
    # Find all URLs files
    url_files = []
    for filename in os.listdir(directory_path):
        if filename.startswith('urls_') and filename.endswith('.txt'):
            url_files.append(os.path.join(directory_path, filename))
    
    if not url_files:
        print("No URLs files found!")
        return
    
    print(f"Found {len(url_files)} files to fix:")
    for file_path in url_files:
        print(f"  - {os.path.basename(file_path)}")
    
    # Fix each file
    for file_path in url_files:
        fix_urls_file(file_path)
    
    print("\n✓ All files have been fixed!")

def main():
    # Configuration
    OUTPUT_DIR = "/home/sdainelli/Aladdin/GenieRedux/data_generation/external/ROMs"
    
    # Check if directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Directory {OUTPUT_DIR} does not exist!")
        return
    
    # Fix all URLs files
    fix_all_urls_files(OUTPUT_DIR)
    
    # Show sample of fixed content
    print("\nSample of fixed content:")
    sample_file = os.path.join(OUTPUT_DIR, "urls_Atari2600.txt")
    if os.path.exists(sample_file):
        with open(sample_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Show the first 5 non-header lines
            count = 0
            for line in lines:
                if line.strip() and not line.startswith('#') and not line.startswith('='):
                    print(f"  {line.strip()}")
                    count += 1
                    if count >= 5:
                        break

if __name__ == "__main__":
    main()