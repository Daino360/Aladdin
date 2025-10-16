import os
import requests
import time
from urllib.parse import unquote

def create_session():
    """Create a requests session with manual authentication"""
    session = requests.Session()
    
    # Manual cookies from your browser
    manual_cookies = {
        'logged-in-sig': '1790324854%201758788854%20l%2F%2BA2M2uXjwF5HzXZlQdqYIfmsecxqJJb9pJDUde%2B5Q9rUcdpek5yDjpo01Ibk9NQVVHP407A0fDq%2FuJL1p38u7NwdynyCFSrFsDWMMcc49%2BDL5GRqSd285cMzMhiV7RPU%2F6lujYrKWNklN8TF7sJ%2FtER%2FgBAPcmR3D8zaYL%2BYG6MnHNl7pRC2SpWjDY8V%2BnGUj2IU%2F9J5dmqrQDH7g0qbQylXXz63CxT7LjknkJKJP34ZOAxSMi8Sfm%2F0gyff%2FyUkkW%2FvQOwmw7zdD3mDiOhL14dEYcj2FNxheJYpOG3ADgMWdW26q43KP3a1CpMlcQzcJCcqCfzCbCrEOwvH%2Fv5Q%3D%3D',
        'logged-in-user': 'stefano.dainelli%40edu.unifi.it',
    }
    
    session.cookies.update(manual_cookies)
    
    # Set common headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    })
    
    print("Manual authentication cookies set")
    return session

def load_failed_roms(failed_file_path):
    """Load the list of failed ROMs from the no_url_found file"""
    failed_roms = []
    try:
        with open(failed_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Check if line already has a URL (format: "Game-Console|url")
                    if '|' in line:
                        parts = line.split('|', 1)
                        game_console = parts[0].strip()
                        url = parts[1].strip()
                    else:
                        game_console = line
                        url = None
                    
                    # Extract game name and console from the format "GameName-Console"
                    if '-' in game_console:
                        parts = game_console.split('-', 1)
                        game_name = parts[0].strip()
                        console = parts[1].strip()
                        failed_roms.append({
                            'name': game_name,
                            'console': console,
                            'original_line': game_console,
                            'url': url
                        })
        print(f"Loaded {len(failed_roms)} failed ROMs from {failed_file_path}")
        return failed_roms
    except Exception as e:
        print(f"Error reading failed ROMs file: {e}")
        return []

def load_all_urls(urls_file_path):
    """Load all URLs from the urls_all_consoles.txt file"""
    url_data = {}
    try:
        with open(urls_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '|' in line:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        game_name = parts[0].strip()
                        url = parts[1].strip()
                        
                        # Try to extract console from game name (format: "Game Name (Console)")
                        console = "Unknown"
                        if '(' in game_name and ')' in game_name:
                            start_idx = game_name.rfind('(') + 1
                            end_idx = game_name.rfind(')')
                            if start_idx < end_idx:
                                console = game_name[start_idx:end_idx].strip()
                        
                        # Clean game name by removing console info
                        clean_game_name = game_name
                        if '(' in clean_game_name:
                            clean_game_name = clean_game_name.split('(')[0].strip()
                        
                        if console not in url_data:
                            url_data[console] = {}
                        
                        url_data[console][clean_game_name] = url
        
        # Print summary
        total_urls = sum(len(urls) for urls in url_data.values())
        print(f"Loaded {total_urls} URLs from {urls_file_path}")
        print("URLs by console:")
        for console, urls in url_data.items():
            print(f"  {console}: {len(urls)} URLs")
        
        return url_data
    except Exception as e:
        print(f"Error reading URLs file: {e}")
        return {}

def find_exact_url(game_name, console, url_data):
    """Find exact URL for a game in the URL data"""
    if console not in url_data:
        return None
    
    # Try exact match first
    if game_name in url_data[console]:
        return url_data[console][game_name]
    
    # Try normalized matching
    normalized_game = game_name.lower().replace(' ', '').replace('-', '').replace('_', '').replace(',', '').replace('.', '').replace("'", "")
    
    for stored_game_name, url in url_data[console].items():
        normalized_stored = stored_game_name.lower().replace(' ', '').replace('-', '').replace('_', '').replace(',', '').replace('.', '').replace("'", "")
        
        if normalized_game == normalized_stored:
            return url
        
        # Check if one contains the other
        if normalized_game in normalized_stored or normalized_stored in normalized_game:
            return url
    
    return None

def update_failed_list_with_urls(failed_roms, url_data):
    """Update failed ROMs list with found URLs and return updated list"""
    updated_failed_roms = []
    found_urls_count = 0
    
    print("\nUpdating failed list with URLs...")
    for game in failed_roms:
        # If URL already exists, keep it
        if game['url']:
            updated_failed_roms.append(game)
            continue
        
        # Try to find URL
        url = find_exact_url(game['name'], game['console'], url_data)
        if url:
            game['url'] = url
            found_urls_count += 1
            print(f"✓ Found URL for {game['name']} ({game['console']})")
        else:
            print(f"✗ No URL found for {game['name']} ({game['console']})")
        
        updated_failed_roms.append(game)
    
    print(f"Found URLs for {found_urls_count} out of {len(failed_roms)} failed ROMs")
    return updated_failed_roms

def save_updated_failed_list(failed_roms, output_path):
    """Save the updated failed list with URLs"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Failed ROMs with URLs (format: Game-Console|url)\n")
            for game in failed_roms:
                if game['url']:
                    f.write(f"{game['original_line']}|{game['url']}\n")
                else:
                    f.write(f"{game['original_line']}\n")
        print(f"✓ Updated failed list saved to: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving updated failed list: {e}")
        return False

def format_game_name_for_filename(game_name):
    """Format game name for safe filename"""
    safe_name = "".join(c for c in game_name if c.isalnum() or c in (' ', '-', '_'))
    return safe_name.replace(' ', '_')

def download_rom(session, game, url, output_dir):
    """Download a single ROM using exact URL"""
    # Create safe filename based on game name
    safe_filename = f"{format_game_name_for_filename(game['name'])}.zip"
    filepath = os.path.join(output_dir, safe_filename)
    
    # Skip if file already exists
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        print(f"✓ File already exists, skipping: {safe_filename} ({file_size:,} bytes)")
        return True
    
    try:
        # Extract the actual filename from URL for display
        url_filename = url.split('/')[-1]
        print(f"Downloading: {safe_filename}")
        print(f"  From URL: {url_filename}")
        
        # Download the file
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Check if we got a valid file
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            print(f"✗ Got HTML response instead of ROM file")
            return False
        
        # Get file size from headers
        content_length = response.headers.get('content-length')
        if content_length:
            file_size = int(content_length)
            print(f"  File size: {file_size:,} bytes")
        
        # Download the file
        downloaded_size = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
        
        # Verify file was downloaded properly
        actual_size = os.path.getsize(filepath)
        if actual_size > 1024:  # At least 1KB
            print(f"✓ Successfully downloaded: {safe_filename} ({actual_size:,} bytes)")
            return True
        else:
            # Remove invalid file
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"✗ Downloaded file too small: {actual_size} bytes")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Download failed: {e}")
        return False

def get_user_provided_url(game_name, console):
    """Ask user to provide URL for a failed download"""
    print(f"\n⚠️  Download failed for: {game_name} ({console})")
    print("Please provide the correct URL or press Enter to skip:")
    url = input("URL: ").strip()
    
    if url and (url.startswith('http://') or url.startswith('https://')):
        return url
    elif url:
        print("Invalid URL format. Should start with http:// or https://")
        return None
    else:
        print("Skipping this game.")
        return None

def main():
    # Configuration
    OUTPUT_DIR = "/home/sdainelli/Aladdin/GenieRedux/data_generation/external/ROMs"
    FAILED_ROMS_FILE = "/home/sdainelli/Aladdin/GenieRedux/data_generation/external/ROMs/no_url_found_20250929_131003.txt"
    ALL_URLS_FILE = "/home/sdainelli/Aladdin/GenieRedux/data_generation/external/ROMs/urls_all_consoles.txt"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load failed ROMs
    failed_roms = load_failed_roms(FAILED_ROMS_FILE)
    if not failed_roms:
        print("No failed ROMs found to retry. Exiting.")
        return
    
    # Load all URLs
    url_data = load_all_urls(ALL_URLS_FILE)
    if not url_data:
        print("No URLs found in the URLs file. Exiting.")
        return
    
    # Step 1: Update failed list with URLs
    updated_failed_roms = update_failed_list_with_urls(failed_roms, url_data)
    
    # Save updated failed list
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    updated_failed_file = os.path.join(OUTPUT_DIR, f"failed_roms_with_urls_{timestamp}.txt")
    save_updated_failed_list(updated_failed_roms, updated_failed_file)
    
    # Filter games that have URLs
    games_with_urls = [game for game in updated_failed_roms if game['url']]
    games_without_urls = [game for game in updated_failed_roms if not game['url']]
    
    print(f"\nGames with URLs: {len(games_with_urls)}")
    print(f"Games without URLs: {len(games_without_urls)}")
    
    if not games_with_urls:
        print("No games with URLs to download. Exiting.")
        return
    
    # Ask if user wants to proceed with download
    print(f"\nDo you want to proceed with downloading {len(games_with_urls)} games?")
    response = input("Enter 'y' to continue or any other key to exit: ").strip().lower()
    if response != 'y':
        print("Exiting.")
        return
    
    # Create authenticated session
    session = create_session()
    
    # Step 2: Download games with URLs
    successful_downloads = 0
    failed_downloads = []
    user_corrected_urls = []
    
    print(f"\nStarting download of {len(games_with_urls)} games with URLs...")
    print("=" * 60)
    
    start_time = time.time()
    
    for i, game in enumerate(games_with_urls, 1):
        print(f"\n[{i}/{len(games_with_urls)}] {game['name']} ({game['console']})")
        
        # Try to download with current URL
        if download_rom(session, game, game['url'], OUTPUT_DIR):
            successful_downloads += 1
        else:
            # Download failed, ask user for correct URL
            corrected_url = get_user_provided_url(game['name'], game['console'])
            if corrected_url:
                # Try again with user-provided URL
                print("Retrying with user-provided URL...")
                if download_rom(session, game, corrected_url, OUTPUT_DIR):
                    successful_downloads += 1
                    user_corrected_urls.append((game['original_line'], corrected_url))
                else:
                    failed_downloads.append(game['original_line'])
            else:
                failed_downloads.append(game['original_line'])
    
    # Print summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total games with URLs: {len(games_with_urls)}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {len(failed_downloads)}")
    print(f"Games without URLs: {len(games_without_urls)}")
    print(f"User-corrected URLs: {len(user_corrected_urls)}")
    print(f"Total time: {total_time:.1f} seconds")
    
    # Save results
    if successful_downloads > 0:
        success_file = os.path.join(OUTPUT_DIR, f"retry_success_{timestamp}.txt")
        with open(success_file, 'w', encoding='utf-8') as f:
            f.write(f"# Successfully downloaded {successful_downloads} games in retry\n")
        print(f"✓ Success summary saved to: {success_file}")
    
    if failed_downloads:
        failed_file = os.path.join(OUTPUT_DIR, f"retry_failed_{timestamp}.txt")
        with open(failed_file, 'w', encoding='utf-8') as f:
            f.write("# Failed downloads in retry\n")
            for game in failed_downloads:
                f.write(f"{game}\n")
        print(f"✗ Failed downloads saved to: {failed_file}")
    
    if games_without_urls:
        no_url_file = os.path.join(OUTPUT_DIR, f"retry_no_url_{timestamp}.txt")
        with open(no_url_file, 'w', encoding='utf-8') as f:
            f.write("# Games with no URL found\n")
            for game in games_without_urls:
                f.write(f"{game['original_line']}\n")
        print(f"✗ Games with no URL found saved to: {no_url_file}")
    
    if user_corrected_urls:
        corrected_file = os.path.join(OUTPUT_DIR, f"user_corrected_urls_{timestamp}.txt")
        with open(corrected_file, 'w', encoding='utf-8') as f:
            f.write("# User-corrected URLs (format: Game-Console|url)\n")
            for game_line, url in user_corrected_urls:
                f.write(f"{game_line}|{url}\n")
        print(f"✓ User-corrected URLs saved to: {corrected_file}")

if __name__ == "__main__":
    main()