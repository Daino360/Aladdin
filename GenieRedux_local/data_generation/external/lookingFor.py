import os
import subprocess
import re

def extract_first_name(game_entry):
    """
    Extract the first name from game entry before the second uppercase letter.
    The pattern looks for the first word before a second capital letter appears.
    """
    # Split by hyphen to get the first part
    first_part = game_entry.split('-')[0]
    
    # Find where the second uppercase letter appears
    uppercase_count = 0
    for i, char in enumerate(first_part):
        if char.isupper():
            uppercase_count += 1
            if uppercase_count == 2:
                return first_part[:i]
    
    # If no second uppercase found, return the whole first part
    return first_part

def search_in_directory(directory, search_term):
    """Search for files containing the search term in the specified directory using find command."""
    try:
        # Use find command to search for files containing the search term
        result = subprocess.run([
            'find', directory, 
            '-type', 'f', 
            '-iname', f'*{search_term}*'
        ], capture_output=True, text=True)
        
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except Exception as e:
        print(f"Error searching for {search_term}: {e}")
        return []

def main():
    games_list = """SuperStarWars-Snes,side free pl
SuperStarWarsTheEmpireStrikesBack-Snes,side free pl
SuperTrollIslands-Snes,side free pl
SuperTurrican-Snes,side free pl
SuperTurrican2-Snes,side free pl
SuperWidget-Snes,side free pl
SuperWonderBoy-Sms,side free pl
SwampThing-Nes,side free pl
SwordOfSodan-Genesis,side free pl
SydOfValis-Genesis,side free pl
SylvesterAndTweetyInCageyCapers-Genesis,side free pl
TakahashiMeijinNoBugutteHoney-Nes,side free pl
TargetEarth-Genesis,side free pl
TazMania-Genesis,side free pl
TazMania-Sms,side free pl
TeenageMutantNinjaTurtles-Nes,side free pl
TeenageMutantNinjaTurtlesIIITheManhattanProject-Nes,3p free pl
TeenageMutantNinjaTurtlesIITheArcadeGame-Nes,3p free pl
Terminator-Genesis,side free pl
Terminator-Sms,side free pl
Terminator2JudgmentDay-Nes,side free pl
TetsuwanAtom-Nes,side free pl
ThunderFox-Genesis,side free pl
TimeZone-Nes,side free pl
TinHead-Genesis,side free pl
TinyToonAdventures-Nes,side free pl
TinyToonAdventuresBusterBustsLoose-Snes,side free pl
TinyToonAdventuresBustersHiddenTreasure-Genesis,side free pl
Toki-Nes,side free pl
TomAndJerry-Snes,side free pl
TotallyRad-Nes,side free pl
TotalRecall-Nes,side free pl
ToxicCrusaders-Nes,3p free pl
TreasureMaster-Nes,side free pl
TrollsInCrazyland-Nes,side free pl
TroubleShooter-Genesis,side free pl
Turrican-Genesis,side free pl
UniversalSoldier-Genesis,side free pl
UruseiYatsuraLumNoWeddingBell-Nes,side free pl
Vectorman-Genesis,side free pl
Vectorman2-Genesis,side free pl
WaniWaniWorld-Genesis,side free pl
Wardner-Genesis,side free pl
WaynesWorld-Nes,side free pl
WereBackADinosaursStory-Snes,side free pl
Widget-Nes,side free pl
WizardsAndWarriors-Nes,side free pl
WiznLiz-Genesis,side free pl
Wolfchild-Genesis,side free pl
Wolfchild-Sms,side free pl
Wolfchild-Snes,side free pl
WolverineAdamantiumRage-Genesis,side free pl
WonderBoyInMonsterWorld-Sms,side free pl
WrathOfTheBlackManta-Nes,side free pl
WreckingCrew-Nes,side free pl
Xexyz-Nes,side free pl
XMenMojoWorld-Sms,side free pl
YoukaiClub-Nes,side free pl
YoukaiDouchuuki-Nes,side free pl
YoungIndianaJonesChronicles-Nes,side free pl
ZeroTheKamikazeSquirrel-Genesis,side free pl
ZeroTheKamikazeSquirrel-Snes,side free pl
ZoolNinjaOfTheNthDimension-Genesis,side free pl
ZoolNinjaOfTheNthDimension-Sms,side free pl
ZoolNinjaOfTheNthDimension-Snes,side free pl"""

    # Directory to search in
    roms_directory = "/home/sdainelli/Aladdin/GenieRedux/data_generation/external/ROMs"
    
    # Check if directory exists
    if not os.path.exists(roms_directory):
        print(f"Directory '{roms_directory}' does not exist!")
        return
    
    print(f"Searching in directory: {roms_directory}")
    print("=" * 50)
    
    found_games = []
    not_found_games = []
    
    # Process each game in the list
    for game_line in games_list.split('\n'):
        if not game_line.strip():
            continue
            
        first_name = extract_first_name(game_line)
        print(f"Searching for: {first_name}")
        
        # Search for files containing the first name
        results = search_in_directory(roms_directory, first_name)
        
        if results:
            found_games.append((first_name, results))
            print(f"✓ FOUND: {first_name}")
            for result in results:
                print(f"  - {result}")
        else:
            not_found_games.append(first_name)
            print(f"✗ NOT FOUND: {first_name}")
        
        print("-" * 30)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Found: {len(found_games)} games")
    print(f"Not found: {len(not_found_games)} games")
    
    if found_games:
        print("\nFound games:")
        for game_name, paths in found_games:
            print(f"  {game_name}: {len(paths)} file(s)")
    
    if not_found_games:
        print("\nGames not found:")
        for game_name in not_found_games:
            print(f"  {game_name}")

if __name__ == "__main__":
    main()