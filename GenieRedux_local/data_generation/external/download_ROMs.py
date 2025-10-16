import os
import requests
import csv
import time
import re
from urllib.parse import unquote
from difflib import SequenceMatcher

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

def read_game_list_from_csv(csv_path):
    """Read game list from CSV file"""
    games = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                game_field = row['game'].strip()
                if not game_field:
                    continue
                    
                # Extract game name and console
                parts = game_field.split('-')
                if len(parts) >= 2:
                    game_name = parts[0].strip()
                    console = parts[1].strip()
                    games.append({
                        'name': game_name,
                        'console': console,
                        'original_line': game_field
                    })
        print(f"Successfully read {len(games)} games from CSV")
        return games
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def load_urls_from_files(output_dir):
    """Load all URLs from the url_*.txt files with game names"""
    url_data = {}
    for filename in os.listdir(output_dir):
        if filename.startswith('urls_') and filename.endswith('.txt'):
            console = filename[5:-4]  # Extract console name from "urls_Console.txt"
            filepath = os.path.join(output_dir, filename)
            
            console_data = {}
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '|' in line:
                            game_name, url = line.split('|', 1)
                            console_data[game_name] = url
                url_data[console] = console_data
                print(f"Loaded {len(console_data)} game URLs for {console}")
            except Exception as e:
                print(f"Error reading URL file {filename}: {e}")
    
    return url_data

def normalize_game_name(name):
    """Normalize game name for matching by removing common variations"""
    # Convert to lowercase
    name = name.lower()
    
    # Remove language codes and regional indicators in parentheses
    name = re.sub(r'\s*\([^)]*(fr|en|es|it|de|jp|us|eu|uk)[^)]*\)', '', name)
    
    # Remove language codes without parentheses
    name = re.sub(r'\s*(fr|en|es|it|de|jp|us|eu|uk)\s*$', '', name)
    
    # Remove punctuation and special characters
    name = re.sub(r'[^\w\s]', '', name)
    
    # Replace multiple spaces with single space and strip
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def calculate_similarity_score(game_name, stored_name):
    """Calculate similarity score between two game names"""
    # Normalize both names
    norm_game = normalize_game_name(game_name)
    norm_stored = normalize_game_name(stored_name)
    
    # Exact match after normalization
    if norm_game == norm_stored:
        return 1.0
    
    # Use SequenceMatcher for fuzzy matching
    similarity = SequenceMatcher(None, norm_game, norm_stored).ratio()
    
    # Bonus points if one contains the other
    if norm_game in norm_stored or norm_stored in norm_game:
        similarity += 0.2
    
    # Check if they start with the same words (priority for beginning match)
    game_words = norm_game.split()
    stored_words = norm_stored.split()
    
    if game_words and stored_words and game_words[0] == stored_words[0]:
        similarity += 0.1
    
    return min(similarity, 1.0)  # Cap at 1.0

def find_matching_url(game_name, console, url_data):
    """Find the best matching URL for a game in the URL files with improved matching"""
    if console not in url_data:
        return None
    
    candidates = []
    
    for stored_game_name, url in url_data[console].items():
        similarity = calculate_similarity_score(game_name, stored_game_name)
        
        if similarity > 0.6:  # Minimum similarity threshold
            candidates.append({
                'url': url,
                'similarity': similarity,
                'stored_name': stored_game_name
            })
    
    if not candidates:
        return None
    
    # Sort by similarity score (highest first)
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    best_match = candidates[0]
    
    # Print debug info for high-quality matches
    if best_match['similarity'] > 0.8:
        print(f"  Found match: '{best_match['stored_name']}' (similarity: {best_match['similarity']:.2f})")
    
    return best_match['url']

def ask_user_for_url(game_name, console):
    """Ask user to manually provide a URL for a game"""
    print(f"\n⚠️  No URL found for: {game_name} ({console})")
    print("Please provide a download URL manually (or press Enter to skip):")
    url = input("URL: ").strip()
    
    if url:
        # Basic URL validation
        if url.startswith(('http://', 'https://')):
            return url
        else:
            print("Invalid URL format. Please include http:// or https://")
            return ask_user_for_url(game_name, console)  # Recursively ask again
    else:
        return None

def format_game_name_for_filename(game_name):
    """Format game name for safe filename"""
    safe_name = "".join(c for c in game_name if c.isalnum() or c in (' ', '-', '_'))
    return safe_name.replace(' ', '_')

def download_rom(session, game, url, output_dir):
    """Download a single ROM using exact URL"""
    # Create safe filename based on game name from CSV
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

def check_existing_files(output_dir, games):
    """Check which games already exist in the output directory"""
    existing_files = set()
    for filename in os.listdir(output_dir):
        if filename.endswith('.zip'):
            # Remove .zip extension and normalize
            base_name = filename[:-4].replace('_', ' ').lower()
            existing_files.add(base_name)
    
    games_to_download = []
    skipped_games = []
    
    for game in games:
        safe_name = format_game_name_for_filename(game['name'])
        base_name = safe_name.replace('_', ' ').lower()
        
        if base_name in existing_files:
            skipped_games.append(game)
        else:
            games_to_download.append(game)
    
    print(f"Found {len(existing_files)} existing ROM files")
    print(f"Skipping {len(skipped_games)} games that already exist")
    print(f"Will download {len(games_to_download)} new games")
    
    return games_to_download, skipped_games

def filter_games_by_console(games, selected_consoles):
    """Filter games to only include selected consoles"""
    if not selected_consoles:
        return games
    
    filtered_games = [game for game in games if game['console'] in selected_consoles]
    print(f"Filtered to {len(filtered_games)} games from consoles: {', '.join(selected_consoles)}")
    return filtered_games

def get_available_consoles(games):
    """Get list of available consoles from the game list"""
    consoles = set(game['console'] for game in games)
    return sorted(consoles)

def retry_with_user_url(session, game, output_dir):
    """Ask user for URL and retry download"""
    print(f"\n🔄 Retrying download for: {game['name']} ({game['console']})")
    url = ask_user_for_url(game['name'], game['console'])
    
    if url:
        print(f"Trying user-provided URL: {url}")
        if download_rom(session, game, url, output_dir):
            return True, url
        else:
            print(f"✗ Download failed with user-provided URL")
            return False, url
    else:
        print(f"⏭️  Skipping game: {game['name']}")
        return False, None

def main():
    # Configuration
    OUTPUT_DIR = "/home/sdainelli/Aladdin/GenieRedux/data_generation/external/ROMs"
    # CSV_PATH = "/home/sdainelli/Aladdin/GenieRedux/data_generation/annotations/RetroAct_v0.1.csv"
    CSV_PATH = "/home/sdainelli/Aladdin/GenieRedux/data_generation/annotations/pl_games.csv"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read game list from CSV
    games = read_game_list_from_csv(CSV_PATH)
    if not games:
        print("No games found in CSV file. Exiting.")
        return
    
    # Load URLs from URL files
    print("\nLoading URLs from URL files...")
    url_files = load_urls_from_files(OUTPUT_DIR)
    
    if not url_files:
        print("No URL files found. Please run the URL scraping script first.")
        print("Expected files: urls_Sms.txt, urls_Nes.txt, etc.")
        return
    
    print(f"Loaded URLs for consoles: {', '.join(url_files.keys())}")
    
    # Count games by console
    console_counts = {}
    for game in games:
        console = game['console']
        console_counts[console] = console_counts.get(console, 0) + 1
    
    print("\nGames by console:")
    available_consoles = get_available_consoles(games)
    for console in available_consoles:
        count = console_counts.get(console, 0)
        url_count = len(url_files.get(console, []))
        print(f"  {console}: {count} games (URLs available: {url_count})")
    
    # Ask user which consoles to download
    print("\nWhich consoles do you want to download?")
    print("Options: all, or enter console names separated by commas (e.g., Nes,Genesis,Snes)")
    console_input = input("Enter selection: ").strip()
    
    if console_input.lower() == 'all':
        selected_consoles = []
        filtered_games = games
        print("Downloading games from all consoles")
    else:
        selected_consoles = [c.strip() for c in console_input.split(',') if c.strip()]
        # Validate console names
        invalid_consoles = [c for c in selected_consoles if c not in available_consoles]
        if invalid_consoles:
            print(f"Warning: Unknown consoles: {', '.join(invalid_consoles)}")
            selected_consoles = [c for c in selected_consoles if c in available_consoles]
        
        filtered_games = filter_games_by_console(games, selected_consoles)
        if not filtered_games:
            print("No games match the selected consoles. Exiting.")
            return
    
    # Check for existing files before downloading
    games_to_download, skipped_games = check_existing_files(OUTPUT_DIR, filtered_games)
    
    if not games_to_download:
        print("All games already exist in the download directory. Exiting.")
        return
    
    # Create authenticated session
    session = create_session()
    
    # Find URLs for each game and download
    successful_downloads = 0
    failed_downloads = []  # List of tuples (game_name, url)
    no_url_found = []
    user_provided_success = []
    
    print(f"\nStarting download of {len(games_to_download)} new games...")
    print("=" * 60)
    
    start_time = time.time()
    
    for i, game in enumerate(games_to_download, 1):
        print(f"\n[{i}/{len(games_to_download)}] {game['name']} ({game['console']})")
        
        # Find matching URL
        url = find_matching_url(game['name'], game['console'], url_files)
        
        if url:
            if download_rom(session, game, url, OUTPUT_DIR):
                successful_downloads += 1
            else:
                # Download failed with auto-found URL, ask user for alternative
                print(f"✗ Auto-found URL failed: {url}")
                success, user_url = retry_with_user_url(session, game, OUTPUT_DIR)
                if success:
                    successful_downloads += 1
                    user_provided_success.append((game['original_line'], user_url))
                else:
                    if user_url:
                        failed_downloads.append((game['original_line'], user_url))
                    else:
                        failed_downloads.append((game['original_line'], url))
        else:
            # No URL found automatically, ask user for URL
            print(f"✗ No matching URL found for {game['name']}")
            success, user_url = retry_with_user_url(session, game, OUTPUT_DIR)
            if success:
                successful_downloads += 1
                user_provided_success.append((game['original_line'], user_url))
            else:
                if user_url:
                    failed_downloads.append((game['original_line'], user_url))
                else:
                    no_url_found.append(game['original_line'])
    
    # Print summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total games in CSV: {len(games)}")
    print(f"Games from selected consoles: {len(filtered_games)}")
    print(f"Already existed: {len(skipped_games)}")
    print(f"Attempted to download: {len(games_to_download)}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"  - Auto-found URLs: {successful_downloads - len(user_provided_success)}")
    print(f"  - User-provided URLs: {len(user_provided_success)}")
    print(f"Failed downloads: {len(failed_downloads)}")
    print(f"No URL found: {len(no_url_found)}")
    print(f"Total time: {total_time:.1f} seconds")
    
    # Save failed and missing URLs
    if failed_downloads or no_url_found or user_provided_success:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if failed_downloads:
            failed_file = os.path.join(OUTPUT_DIR, f"failed_downloads_{timestamp}.txt")
            with open(failed_file, 'w', encoding='utf-8') as f:
                f.write("# Failed downloads (format: game_name|url)\n")
                for game_line, url in failed_downloads:
                    f.write(f"{game_line}|{url}\n")
            print(f"✗ Failed downloads saved to: {failed_file}")
        
        if no_url_found:
            no_url_file = os.path.join(OUTPUT_DIR, f"no_url_found_{timestamp}.txt")
            with open(no_url_file, 'w', encoding='utf-8') as f:
                f.write("# Games with no matching URL found\n")
                for game in no_url_found:
                    f.write(f"{game}\n")
            print(f"✗ Games with no URL found saved to: {no_url_file}")
        
        if user_provided_success:
            user_urls_file = os.path.join(OUTPUT_DIR, f"user_provided_urls_{timestamp}.txt")
            with open(user_urls_file, 'w', encoding='utf-8') as f:
                f.write("# Successfully downloaded with user-provided URLs (format: game_name|url)\n")
                for game_line, url in user_provided_success:
                    f.write(f"{game_line}|{url}\n")
            print(f"✓ User-provided successful URLs saved to: {user_urls_file}")

if __name__ == "__main__":
    main()