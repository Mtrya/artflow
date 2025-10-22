"""
Dataset curation script for WikiArt with multi-model captions.

This script creates a curated subset of the huggan/wikiart dataset with captions
generated from multiple vision-language models. The pipeline:
1. Downloads huggan/wikiart dataset
2. Filters for specific artistic styles and known artists/genres  
3. Generates captions using multiple VLM models in parallel
4. Removes images that lack captions
5. Uploads final dataset to HuggingFace Hub

Caption generation uses 4 different approaches:
- Qwen-VL Plus with direct captioning prompt
- Qwen-VL Plus with reverse image prompt (text-to-image style)
- GPT 5 Mini with direct captioning prompt
- Template constructed by style, genre and artist
"""

from datasets import load_dataset
import sys
import os
from typing import List, Dict

# Add src/data to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from get_captions import call_parallel, CAPTIONERS
except ImportError:
    from .get_captions import call_parallel, CAPTIONERS

MAX_RETRIES = 3
BATCH_SIZE = 6
TEST_MODE = True

# Mapping dictionaries for integer indices to string names
ARTIST_MAP = {
    0: "Unknown Artist", 1: "Boris Kustodiev", 2: "Camille Pissarro", 3: "Childe Hassam",
    4: "Claude Monet", 5: "Edgar Degas", 6: "Eugene Boudin", 7: "Gustave Dore", 8: "Ilya Repin",
    9: "Ivan Aivazovsky", 10: "Ivan Shishkin", 11: "John Singer Sargent", 12: "Marc Chagall",
    13: "Martiros Saryan", 14: "Nicholas Roerich", 15: "Pablo Picasso", 16: "Paul Cezanne",
    17: "Pierre Auguste Renoir", 18: "Pyotr Konchalovsky", 19: "Raphael Kirchner", 20: "Rembrandt",
    21: "Salvador Dali", 22: "Vincent Van Gogh", 23: "Hieronymus Bosch", 24: "Leonardo da-vinci",
    25: "Albrecht Durer", 26: "Edouard Cortes", 27: "Sam Francis", 28: "Juan Gris",
    29: "Lucas Cranach the Elder", 30: "Paul Gauguin", 31: "Konstantin Makovsky", 32: "Egon Schiele",
    33: "Thomas Eakins", 34: "Gustave Moreau", 35: "Francisco Goya", 36: "Edvard Munch",
    37: "Henri Matisse", 38: "Fra Angelico", 39: "Maxime Maufra", 40: "Jan Matejko",
    41: "Mstislav Dobuzhinsky", 42: "Alfred Sisley", 43: "Mary Cassatt", 44: "Gustave Loiseau",
    45: "Fernando Botero", 46: "Zinaida Serebriakova", 47: "Georges Seurat", 48: "Isaac Levitan",
    49: "Joaquín Sorolla", 50: "Jacek Malczewski", 51: "Berthe Morisot", 52: "Andy Warhol",
    53: "Arkhip Kuindzhi", 54: "Niko Pirosmani", 55: "James Tissot", 56: "Vasily Polenov",
    57: "Valentin Serov", 58: "Pietro Perugino", 59: "Pierre Bonnard", 60: "Ferdinand Hodler",
    61: "Bartolome Esteban Murillo", 62: "Giovanni Boldini", 63: "Henri Martin", 64: "Gustav Klimt",
    65: "Vasily Perov", 66: "Odilon Redon", 67: "Tintoretto", 68: "Gene Davis", 69: "Raphael",
    70: "John Henry Twachtman", 71: "Henri de Toulouse-Lautrec", 72: "Antoine Blanchard",
    73: "David Burliuk", 74: "Camille Corot", 75: "Konstantin Korovin", 76: "Ivan Bilibin",
    77: "Titian", 78: "Maurice Prendergast", 79: "Edouard Manet", 80: "Peter Paul Rubens",
    81: "Aubrey Beardsley", 82: "Paolo Veronese", 83: "Joshua Reynolds", 84: "Kuzma Petrov-Vodkin",
    85: "Gustave Caillebotte", 86: "Lucian Freud", 87: "Michelangelo", 88: "Dante Gabriel Rossetti",
    89: "Felix Vallotton", 90: "Nikolay Bogdanov-Belsky", 91: "Georges Braque", 92: "Vasily Surikov",
    93: "Fernand Leger", 94: "Konstantin Somov", 95: "Katsushika Hokusai", 96: "Sir Lawrence Alma-Tadema",
    97: "Vasily Vereshchagin", 98: "Ernst Ludwig Kirchner", 99: "Mikhail Vrubel", 100: "Orest Kiprensky",
    101: "William Merritt Chase", 102: "Aleksey Savrasov", 103: "Hans Memling", 104: "Amedeo Modigliani",
    105: "Ivan Kramskoy", 106: "Utagawa Kuniyoshi", 107: "Gustave Courbet", 108: "William Turner",
    109: "Theo van Rysselberghe", 110: "Joseph Wright", 111: "Edward Burne-Jones", 112: "Koloman Moser",
    113: "Viktor Vasnetsov", 114: "Anthony van Dyck", 115: "Raoul Dufy", 116: "Frans Hals",
    117: "Hans Holbein the Younger", 118: "Ilya Mashkov", 119: "Henri Fantin-Latour", 120: "M.C. Escher",
    121: "El Greco", 122: "Mikalojus Ciurlionis", 123: "James McNeill Whistler", 124: "Karl Bryullov",
    125: "Jacob Jordaens", 126: "Thomas Gainsborough", 127: "Eugene Delacroix", 128: "Canaletto"
}

GENRE_MAP = {
    0: "abstract painting", 1: "cityscape", 2: "genre painting", 3: "illustration",
    4: "landscape", 5: "nude painting", 6: "portrait", 7: "religious painting",
    8: "sketch and study", 9: "still life", 10: "Unknown Genre"
}

STYLE_MAP = {
    0: "abstract expressionism", 1: "action painting", 2: "analytical cubism", 3: "art nouveau",
    4: "baroque", 5: "color field painting", 6: "contemporary realism", 7: "cubism",
    8: "early renaissance", 9: "expressionism", 10: "fauvism", 11: "high renaissance",
    12: "impressionism", 13: "mannerism late renaissance", 14: "minimalism", 15: "naive art primitivism",
    16: "new realism", 17: "northern renaissance", 18: "pointillism", 19: "pop art",
    20: "post impressionism", 21: "realism", 22: "rococo", 23: "romanticism",
    24: "symbolism", 25: "synthetic cubism", 26: "ukiyo-e"
}

# Step 1. Load dataset
ds = load_dataset("huggan/wikiart")["train"]


# Step 2. Filter for a subset
STYLES = [
    "high renaissance", "impressionism", "post impressionism", 
    "baroque", "northern renaissance", "romanticism", 
    "abstract expressionism", "ukiyo-e", "art nouveau", 
    "expressionism", "fauvism", "symbolism", "analytical cubism", 
    "pointillism", "early renaissance", "realism", "rococo", 
    "pop art", "mannerism late renaissance"
    ]

def filter_dataset(batch):
    """Filter function for batched processing."""
    mask = []
    for i in range(len(batch["artist"])):
        keep = (
            ARTIST_MAP[batch["artist"][i]] != "Unknown Artist"
            and GENRE_MAP[batch["genre"][i]] != "Unknown Genre"
            and STYLE_MAP[batch["style"][i]] in STYLES
        )
        mask.append(keep)
    return mask

print("Filtering original dataset...")
if TEST_MODE:
    filtered_ds = ds.select(range(1080))
else:
    filtered_ds = ds.filter(filter_dataset, batched=True)

print(f"Filtered dataset length: {len(ds)} -> {len(filtered_ds)}")

# Step 3. Initial caption generation
def add_captions(batch):
    """Add captions to a batch of examples using multiple VLM models"""
    images = batch["image"]

    # Extract existing captions for all images
    if batch.get(f"{CAPTIONERS[0]["name"]}-caption") is None: # first round, no existing captions, need to initialize the caption keys
        batch["wikiart-caption"] = [None for _ in range(len(images))]
        for c in CAPTIONERS:
            batch[f"{c["name"]}-caption"] = [None for _ in range(len(images))]
        existing_captions = None
    else:
        existing_captions: List[Dict[str,str]] = []
        for i in range(len(images)):
            # For example, existing = {"qwen-direct-caption": "<caption_content>"}
            existing = {f"{c["name"]}-caption": batch.get(f"{c["name"]}-caption")[i] for c in CAPTIONERS}
            existing_captions.append(existing)

    # Call VLM to caption
    captions_batch = call_parallel(images, existing_captions)

    # Add wikiart captions and distribute VLM results
    for i in range(len(images)):
        batch["wikiart-caption"][i] = f"{STYLE_MAP[batch['style'][i]]} {GENRE_MAP[batch['genre'][i]]} by {ARTIST_MAP[batch['artist'][i]]}"
        for captioner in CAPTIONERS:
            batch[f"{captioner['name']}-caption"][i] = captions_batch[i].get(captioner["name"])

    return batch

print("Generating initial captions...")
captioned_ds = filtered_ds.map(add_captions, batched=True, batch_size=BATCH_SIZE)

# Step 4. Remove images that have None captions
print("Removing images with persistent None captions...")

def has_all_captions(example):
    """Check if example has all captions (no None values)"""
    return all(example.get(f"{c['name']}-caption") is not None for c in CAPTIONERS)

final_ds = captioned_ds.filter(has_all_captions)
removed_count = len(captioned_ds) - len(final_ds)

if removed_count > 0:
    print(f"Removed {removed_count} images that still had None captions after {MAX_RETRIES} retries")
    print(f"Final dataset size: {len(final_ds)}")

total_captions = len(final_ds) * len(CAPTIONERS)
print(f"Total captions generated: {total_captions}")
print(f"Captioners used: {[c['name'] for c in CAPTIONERS]}")

# Step 5. Push to HuggingFace
print("Uploading final dataset to HuggingFace...")
dataset_length = len(final_ds) // 1000
final_ds.push_to_hub(f"kaupane/wikiart-captions-{dataset_length}k")

print("Dataset curation complete!")
