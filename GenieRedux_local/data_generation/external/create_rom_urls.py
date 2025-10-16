import os
import requests
import time
from urllib.parse import unquote, quote
from bs4 import BeautifulSoup

def create_session():
    """Create a requests session with manual authentication"""
    session = requests.Session()
    
    # Manual cookies from your browser
    manual_cookies = {
        'logged-in-sig': '1790324854%201758788854%20l%2F%2BA2M2uXjwF5HzXZlQdqYIfmsecxqJJb9pJDUde%2B5Q9rUcdpek5yDjpo01Ibk9NQVVHP407A0fDq%2FuJL1p38u7NwdynyCFSrFsDWMMcc49%2BDL5GRqSd285cMzMhiV7RPU%2F6lujYrKWNklN8TF7sJ%2FtER%2FgBAPcmR3D8zaYL%2BYG6MnHNl7pRC2SpWjDY8V%2BnGUj2IU%2F9J5dmqrQDH7g0qbQylXXz63CxT7LjknkJKJP34ZOAxSMi8Sfm%2F0gyff%2FyUkkW%2FvQOwmw7zdD3mDiOhL14dEYcj2FNxheJYpOG3ADgMWdW26q43KP3a1CpMlcQzcJCcqCfzCbCrEOwvH%2Fv5Q%3D%3D',
        'logged-in-user': 'stefano.dainelli%40edu.unifi.it',
    }
    
    session.cookies.update(manual_cookies)
    
    # Set common headers to mimic a real browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    print("Manual authentication cookies set")
    return session

def get_console_url_mapping():
    """Map console names to their archive identifiers"""
    return {
        'Sms': 'Sega%20-%20Master%20System%20-%20Mark%20III',
        'Nes': 'Nintendo%20-%20Nintendo%20Entertainment%20System', 
        'Genesis': 'Sega%20-%20Mega%20Drive%20-%20Genesis',
        'Snes': 'Nintendo%20-%20Super%20Nintendo%20Entertainment%20System',
        'Atari2600': 'Atari%20-%202600',
        'GameBoy': 'Nintendo%20-%20Game%20Boy',
        '32x': 'Sega%20-%2032X'
    }

def get_console_directory_url(console, collection_name):
    """Get the directory listing URL for a console"""
    console_mapping = get_console_url_mapping()
    if console not in console_mapping:
        return None
    
    console_folder = console_mapping[console]
    # New URL format based on the example
    return f"https://ia804508.us.archive.org/view_archive.php?archive=/26/items/{collection_name}/roms/{console_folder}.zip"

def extract_game_name_from_filename(filename):
    """Extract clean game name from filename"""
    # Remove .zip extension
    if filename.endswith('.zip'):
        filename = filename[:-4]
    
    # Remove region information in parentheses
    import re
    # Remove (USA), (Europe), etc.
    clean_name = re.sub(r'\s*\([^)]*\)\s*', ' ', filename)
    # Remove (Rev A), (Beta), etc.
    clean_name = re.sub(r'\s*\[[^\]]*\]\s*', ' ', clean_name)
    # Remove (Unl) etc.
    clean_name = re.sub(r'\s*\(Unl\)\s*', ' ', clean_name)
    
    # Clean up extra spaces and trim
    clean_name = ' '.join(clean_name.split())
    
    return clean_name

def scrape_rom_urls(session, console, collection_name):
    """Scrape all ROM URLs from the console's directory listing page"""
    directory_url = get_console_directory_url(console, collection_name)
    if not directory_url:
        print(f"Unknown console: {console}")
        return {}
    
    print(f"Scraping ROM URLs for {console}...")
    print(f"URL: {directory_url}")
    
    try:
        response = session.get(directory_url, timeout=30)
        response.raise_for_status()
        
        # Parse the HTML to find all file links
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all file links - looking for .zip files in the directory listing
        rom_data = {}  # {game_name: url}
        seen_urls = set()
        
        console_folder = get_console_url_mapping()[console]
        
        # Look for file links in the directory listing
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Look for .zip files that are actual ROM files (not the main archive)
            if (href.endswith('.zip') and 
                not href.startswith('../') and 
                not href.startswith('http') and
                'view_archive.php' not in href):
                
                # Construct the clean download URL - use the standard archive.org download format
                url = f"https://archive.org/download/{collection_name}/roms/{console_folder}.zip/{href}"
                
                # Skip if we've seen this URL before
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                # Extract filename and game name
                filename = unquote(href)
                game_name = extract_game_name_from_filename(filename)
                
                # Store the data
                rom_data[game_name] = url
        
        print(f"Found {len(rom_data)} unique ROMs for {console}")
        return rom_data
        
    except Exception as e:
        print(f"Error scraping {console}: {e}")
        return {}

def save_urls_to_file(rom_data, console, output_dir):
    """Save the URLs to a text file with game names"""
    filename = f"urls_{console}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# ROM URLs for {console}\n")
        f.write(f"# Total games: {len(rom_data)}\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Collection: ni-roms\n")
        f.write("# Format: game_name|download_url\n")
        f.write("=" * 80 + "\n")
        
        # Sort by game name for easier reading
        for game_name, url in sorted(rom_data.items()):
            # Write only the game name before the pipe, not the URL part
            f.write(f"{game_name}|{url}\n")
    
    print(f"✓ Saved {len(rom_data)} game URLs to: {filepath}")
    
    # Show first few entries as sample
    if rom_data:
        print("Sample entries:")
        for i, (game_name, url) in enumerate(list(rom_data.items())[:3]):
            print(f"  {game_name}|{url}")
    
    return filepath

def save_combined_urls_to_file(all_rom_data, output_dir):
    """Save all URLs to a single combined file"""
    filename = "urls_all_consoles.txt"
    filepath = os.path.join(output_dir, filename)
    
    all_entries = []
    for console_data in all_rom_data.values():
        all_entries.extend(console_data.items())
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# ROM URLs for all consoles\n")
        f.write(f"# Total games: {len(all_entries)}\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Collection: ni-roms\n")
        f.write("# Format: game_name|download_url\n")
        f.write("=" * 80 + "\n")
        
        # Sort by game name
        for game_name, url in sorted(all_entries):
            # Write only the game name before the pipe, not the URL part
            f.write(f"{game_name}|{url}\n")
    
    print(f"✓ Saved {len(all_entries)} game URLs to combined file: {filepath}")
    return filepath

def get_available_consoles():
    """Get list of available consoles"""
    return ['Sms', 'Nes', 'Genesis', 'Snes', 'Atari2600', 'GameBoy', '32x']

def test_url_scraping(session):
    """Test the URL scraping with the provided example"""
    test_url = "https://ia804508.us.archive.org/view_archive.php?archive=/26/items/ni-roms/roms/Atari%20-%202600.zip"
    print("Testing URL scraping with provided example...")
    print(f"Test URL: {test_url}")
    
    try:
        response = session.get(test_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Print some debug info about the page structure
        print(f"Page title: {soup.title.string if soup.title else 'No title'}")
        
        # Find all links for analysis
        zip_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '.zip' in href and not href.startswith('http') and not href.startswith('../'):
                zip_links.append(href)
        
        print(f"Found {len(zip_links)} .zip file links")
        if zip_links:
            print("Sample .zip links:")
            for link in zip_links[:5]:
                print(f"  {link}")
                
            # Show how URLs would be constructed
            print("\nSample constructed download URLs:")
            for link in zip_links[:3]:
                constructed_url = f"https://archive.org/download/ni-roms/roms/Atari%20-%202600.zip/{link}"
                print(f"  {constructed_url}")
                
    except Exception as e:
        print(f"Test failed: {e}")

def main():
    # Configuration
    COLLECTION_NAME = "ni-roms"  # Updated based on the example URL
    OUTPUT_DIR = "/home/sdainelli/Aladdin/GenieRedux/data_generation/external/ROMs"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get available consoles
    available_consoles = get_available_consoles()
    
    print("Available consoles:")
    for i, console in enumerate(available_consoles, 1):
        print(f"  {i}. {console}")
    
    # Ask user which consoles to scrape
    print("\nWhich consoles do you want to scrape URLs for?")
    print("Options: all, or enter console names separated by commas (e.g., Nes,Genesis,Snes)")
    console_input = input("Enter selection: ").strip()
    
    if console_input.lower() == 'all':
        consoles_to_scrape = available_consoles
        print("Scraping URLs for all consoles")
    else:
        consoles_to_scrape = [c.strip() for c in console_input.split(',') if c.strip()]
        # Validate console names
        invalid_consoles = [c for c in consoles_to_scrape if c not in available_consoles]
        if invalid_consoles:
            print(f"Warning: Unknown consoles: {', '.join(invalid_consoles)}")
            consoles_to_scrape = [c for c in consoles_to_scrape if c in available_consoles]
        
        if not consoles_to_scrape:
            print("No valid consoles selected. Exiting.")
            return
    
    # Create authenticated session
    session = create_session()
    
    # Optional: Test with the provided URL first
    test_scraping = input("Do you want to test scraping first? (y/n): ").strip().lower()
    if test_scraping == 'y':
        test_url_scraping(session)
        print("\n" + "="*50 + "\n")
    
    # Scrape URLs for selected consoles and save to files
    print(f"Scraping ROM URLs from archive.org...")
    print("=" * 50)
    
    all_rom_data = {}
    total_games = 0
    successful_consoles = []
    
    for console in consoles_to_scrape:
        rom_data = scrape_rom_urls(session, console, COLLECTION_NAME)
        if rom_data:
            save_urls_to_file(rom_data, console, OUTPUT_DIR)
            all_rom_data[console] = rom_data
            total_games += len(rom_data)
            successful_consoles.append(console)
        else:
            print(f"✗ Failed to scrape URLs for {console}")
        print()
    
    # Create combined file if multiple consoles were scraped
    if len(successful_consoles) > 1:
        save_combined_urls_to_file(all_rom_data, OUTPUT_DIR)
    
    # Print summary
    print(f"{'='*50}")
    print("SCRAPING SUMMARY")
    print(f"{'='*50}")
    print(f"Consoles processed: {len(successful_consoles)}/{len(consoles_to_scrape)}")
    print(f"Total unique games found: {total_games}")
    
    if successful_consoles:
        print(f"\nCreated files:")
        for console in successful_consoles:
            print(f"  ✓ urls_{console}.txt")
        
        if len(successful_consoles) > 1:
            print(f"  ✓ urls_all_consoles.txt (combined)")
        
        print(f"\nAll files saved to: {OUTPUT_DIR}")
        print(f"\nFile format: game_name|download_url")

if __name__ == "__main__":
    main()