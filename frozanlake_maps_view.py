import gymnasium as gym

def print_map(map_layout):
    char_to_text = {
        'S': '\033[97mStart\033[0m',  # Branco
        'F': '\033[92mCaminho\033[0m',  # Verde
        'H': '\033[91mLago\033[0m',  # Vermelho
        'G': '\033[90mEnd\033[0m'  # Preto
    }
    
    for row in map_layout:
        print(" | ".join(char_to_text[char] for char in row))
        print("-" * (len(row) * 8 - 1))


train_maps = [
    ["SFFF", "HFFF", "HFHH", "HFFG"],
    ["SFFF", "FHFH", "FHFF", "HFFG"],
    ["SHHH", "FHFF", "FFFF", "HFFG"],
    ["SFFH", "FHFF", "FFFF", "HFFG"],
    ["SFHH", "FHFF", "FFFF", "HHHG"],
]

test_maps = [
    ["SFHH", "FFFH", "FHHH", "FFFG"],
    ["SFFH", "FFFF", "FHHF", "HHHG"]
]

print("Train Maps:")
for i, map_layout in enumerate(train_maps):
    print(f"Map {i+1}:")
    print_map(map_layout)
    print()

print("Test Maps:")
for i, map_layout in enumerate(test_maps):
    print(f"Map {i+1}:")
    print_map(map_layout)
    print()