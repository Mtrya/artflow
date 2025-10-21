"""
Dataset curation script for WikiArt with multi-model captions.

This script creates a curated subset of the huggan/wikiart dataset with captions
generated from multiple vision-language models. The pipeline:
1. Downloads huggan/wikiart dataset
2. Filters for specific artistic styles and known artists/genres  
3. Generates captions using multiple VLM models in parallel
4. Retries missing captions automatically
5. Removes images that still lack captions after retries
6. Uploads final dataset to HuggingFace Hub

Caption generation uses 4 different approaches:
- Qwen-VL Plus with direct captioning prompt
- Qwen-VL Plus with reverse image prompt (text-to-image style)
- GPT-5 Mini with direct captioning prompt
- Gemini 2.5 Flash with direct captioning prompt
"""

from datasets import load_dataset
import sys
import os

# Add src/data to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from get_captions import call_parallel, CAPTIONERS
except ImportError:
    from .get_captions import call_parallel, CAPTIONERS

MAX_RETRIES = 3

# Mapping dictionaries for integer indices to string names
ARTIST_MAP = {
    0: "Unknown Artist", 1: "boris-kustodiev", 2: "camille-pissarro", 3: "childe-hassam",
    4: "claude-monet", 5: "edgar-degas", 6: "eugene-boudin", 7: "gustave-dore", 8: "ilya-repin",
    9: "ivan-aivazovsky", 10: "ivan-shishkin", 11: "john-singer-sargent", 12: "marc-chagall",
    13: "martiros-saryan", 14: "nicholas-roerich", 15: "pablo-picasso", 16: "paul-cezanne",
    17: "pierre-auguste-renoir", 18: "pyotr-konchalovsky", 19: "raphael-kirchner", 20: "rembrandt",
    21: "salvador-dali", 22: "vincent-van-gogh", 23: "hieronymus-bosch", 24: "leonardo-da-vinci",
    25: "albrecht-durer", 26: "edouard-cortes", 27: "sam-francis", 28: "juan-gris",
    29: "lucas-cranach-the-elder", 30: "paul-gauguin", 31: "konstantin-makovsky", 32: "egon-schiele",
    33: "thomas-eakins", 34: "gustave-moreau", 35: "francisco-goya", 36: "edvard-munch",
    37: "henri-matisse", 38: "fra-angelico", 39: "maxime-maufra", 40: "jan-matejko",
    41: "mstislav-dobuzhinsky", 42: "alfred-sisley", 43: "mary-cassatt", 44: "gustave-loiseau",
    45: "fernando-botero", 46: "zinaida-serebriakova", 47: "georges-seurat", 48: "isaac-levitan",
    49: "joaquín-sorolla", 50: "jacek-malczewski", 51: "berthe-morisot", 52: "andy-warhol",
    53: "arkhip-kuindzhi", 54: "niko-pirosmani", 55: "james-tissot", 56: "vasily-polenov",
    57: "valentin-serov", 58: "pietro-perugino", 59: "pierre-bonnard", 60: "ferdinand-hodler",
    61: "bartolome-esteban-murillo", 62: "giovanni-boldini", 63: "henri-martin", 64: "gustav-klimt",
    65: "vasily-perov", 66: "odilon-redon", 67: "tintoretto", 68: "gene-davis", 69: "raphael",
    70: "john-henry-twachtman", 71: "henri-de-toulouse-lautrec", 72: "antoine-blanchard",
    73: "david-burliuk", 74: "camille-corot", 75: "konstantin-korovin", 76: "ivan-bilibin",
    77: "titian", 78: "maurice-prendergast", 79: "edouard-manet", 80: "peter-paul-rubens",
    81: "aubrey-beardsley", 82: "paolo-veronese", 83: "joshua-reynolds", 84: "kuzma-petrov-vodkin",
    85: "gustave-caillebotte", 86: "lucian-freud", 87: "michelangelo", 88: "dante-gabriel-rossetti",
    89: "felix-vallotton", 90: "nikolay-bogdanov-belsky", 91: "georges-braque", 92: "vasily-surikov",
    93: "fernand-leger", 94: "konstantin-somov", 95: "katsushika-hokusai", 96: "sir-lawrence-alma-tadema",
    97: "vasily-vereshchagin", 98: "ernst-ludwig-kirchner", 99: "mikhail-vrubel", 100: "orest-kiprensky",
    101: "william-merritt-chase", 102: "aleksey-savrasov", 103: "hans-memling", 104: "amedeo-modigliani",
    105: "ivan-kramskoy", 106: "utagawa-kuniyoshi", 107: "gustave-courbet", 108: "william-turner",
    109: "theo-van-rysselberghe", 110: "joseph-wright", 111: "edward-burne-jones", 112: "koloman-moser",
    113: "viktor-vasnetsov", 114: "anthony-van-dyck", 115: "raoul-dufy", 116: "frans-hals",
    117: "hans-holbein-the-younger", 118: "ilya-mashkov", 119: "henri-fantin-latour", 120: "m.c.-escher",
    121: "el-greco", 122: "mikalojus-ciurlionis", 123: "james-mcneill-whistler", 124: "karl-bryullov",
    125: "jacob-jordaens", 126: "thomas-gainsborough", 127: "eugene-delacroix", 128: "canaletto"
}

GENRE_MAP = {
    0: "abstract_painting", 1: "cityscape", 2: "genre_painting", 3: "illustration",
    4: "landscape", 5: "nude_painting", 6: "portrait", 7: "religious_painting",
    8: "sketch_and_study", 9: "still_life", 10: "Unknown Genre"
}

STYLE_MAP = {
    0: "Abstract_Expressionism", 1: "Action_painting", 2: "Analytical_Cubism", 3: "Art_Nouveau",
    4: "Baroque", 5: "Color_Field_Painting", 6: "Contemporary_Realism", 7: "Cubism",
    8: "Early_Renaissance", 9: "Expressionism", 10: "Fauvism", 11: "High_Renaissance",
    12: "Impressionism", 13: "Mannerism_Late_Renaissance", 14: "Minimalism", 15: "Naive_Art_Primitivism",
    16: "New_Realism", 17: "Northern_Renaissance", 18: "Pointillism", 19: "Pop_Art",
    20: "Post_Impressionism", 21: "Realism", 22: "Rococo", 23: "Romanticism",
    24: "Symbolism", 25: "Synthetic_Cubism", 26: "Ukiyo_e"
}

# Step 1. Load dataset
ds = load_dataset("huggan/wikiart")["train"].shuffle(100).select(range(200))


# Step 2. Filter for a subset
STYLES = [
    "High_Renaissance", "Impressionism", "Post_Impressionism", 
    "Baroque", "Northern_Renaissance", "Romanticism", 
    "Abstract_Expressionism", "Ukiyo_e", "Art_Nouveau", 
    "Expressionism", "Fauvism", "Symbolism", "Analytical_Cubism", 
    "Pointillism", "Early_Renaissance", "Realism", "Rococo", 
    "Pop_Art", "Mannerism_Late_Renaissance"
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
filtered_ds = ds.filter(filter_dataset, batched=True)
print(f"Filtered dataset length: {len(ds)} -> {len(filtered_ds)}")

# Step 3. Initial caption generation
def add_captions(example):
    """Add captions to a single example using multiple VLM models."""
    existing = {f"{c['name']}-caption": example.get(f"{c['name']}-caption") for c in CAPTIONERS}
    try:
        captions = call_parallel(example["image"], existing)
    except Exception:
        captions = existing

    example["wikiart-caption"] = f"{STYLE_MAP[example['style']]} {GENRE_MAP[example['genre']]} by {ARTIST_MAP[example['artist']]}"
    for captioner in CAPTIONERS:
        example[f"{captioner['name']}-caption"] = captions.get(captioner["name"])

    return example

print("Generating initial captions...")
captioned_ds = filtered_ds.map(add_captions)

# Step 4. Retry missing captions
for retry_round in range(MAX_RETRIES):
    missing_count = sum(1 for x in captioned_ds if any(x.get(f"{c['name']}-caption") is None for c in CAPTIONERS))

    if missing_count == 0:
        print(f"All captions completed after {retry_round} retries")
        break

    print(f"Retry round {retry_round + 1}: {missing_count} images need captions")
    captioned_ds = captioned_ds.map(add_captions)

    if retry_round < MAX_RETRIES - 1:
        import time
        time.sleep(5)

# Step 5. Remove images that still have None captions
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

# Step 6. Push to HuggingFace
print("Uploading final dataset to HuggingFace...")
dataset_length = len(final_ds) // 1000
final_ds.push_to_hub(f"kaupane/wikiart-captions-{dataset_length}k")

print("Dataset curation complete!")
