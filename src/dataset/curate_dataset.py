"""
Dataset curation script for WikiArt with multi-model captions.

This script creates a curated subset of the huggan/wikiart dataset with captions
generated from multiple vision-language models. The pipeline:
1. Downloads huggan/wikiart dataset
2. Generates captions using multiple VLM models in parallel
3. Removes images that lack captions
4. Uploads final dataset to HuggingFace Hub

Caption generation uses multiple different approaches with multiple models:
- Direct captioning prompt
- Spatial relationship prompt
- Reverse image prompt (text-to-image style)
- Template constructed by style, genre and artist
"""

import time
import os
import json
import argparse
from typing import List, Dict, Optional, Tuple, Set

from datasets import load_dataset, Dataset, concatenate_datasets

try:
    from get_captions import call_parallel, CAPTIONERS_QWEN, CAPTIONERS_MISTRAL
except ImportError:
    from .get_captions import call_parallel, CAPTIONERS_QWEN, CAPTIONERS_MISTRAL

SLEEP_TIME = 1  # to avoid rate limit
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "curate_progress.json")

# Mapping dictionaries for integer indices to string names
ARTIST_MAP = {
    0: "Unknown Artist",
    1: "Boris Kustodiev",
    2: "Camille Pissarro",
    3: "Childe Hassam",
    4: "Claude Monet",
    5: "Edgar Degas",
    6: "Eugene Boudin",
    7: "Gustave Dore",
    8: "Ilya Repin",
    9: "Ivan Aivazovsky",
    10: "Ivan Shishkin",
    11: "John Singer Sargent",
    12: "Marc Chagall",
    13: "Martiros Saryan",
    14: "Nicholas Roerich",
    15: "Pablo Picasso",
    16: "Paul Cezanne",
    17: "Pierre Auguste Renoir",
    18: "Pyotr Konchalovsky",
    19: "Raphael Kirchner",
    20: "Rembrandt",
    21: "Salvador Dali",
    22: "Vincent Van Gogh",
    23: "Hieronymus Bosch",
    24: "Leonardo da-vinci",
    25: "Albrecht Durer",
    26: "Edouard Cortes",
    27: "Sam Francis",
    28: "Juan Gris",
    29: "Lucas Cranach the Elder",
    30: "Paul Gauguin",
    31: "Konstantin Makovsky",
    32: "Egon Schiele",
    33: "Thomas Eakins",
    34: "Gustave Moreau",
    35: "Francisco Goya",
    36: "Edvard Munch",
    37: "Henri Matisse",
    38: "Fra Angelico",
    39: "Maxime Maufra",
    40: "Jan Matejko",
    41: "Mstislav Dobuzhinsky",
    42: "Alfred Sisley",
    43: "Mary Cassatt",
    44: "Gustave Loiseau",
    45: "Fernando Botero",
    46: "Zinaida Serebriakova",
    47: "Georges Seurat",
    48: "Isaac Levitan",
    49: "Joaquín Sorolla",
    50: "Jacek Malczewski",
    51: "Berthe Morisot",
    52: "Andy Warhol",
    53: "Arkhip Kuindzhi",
    54: "Niko Pirosmani",
    55: "James Tissot",
    56: "Vasily Polenov",
    57: "Valentin Serov",
    58: "Pietro Perugino",
    59: "Pierre Bonnard",
    60: "Ferdinand Hodler",
    61: "Bartolome Esteban Murillo",
    62: "Giovanni Boldini",
    63: "Henri Martin",
    64: "Gustav Klimt",
    65: "Vasily Perov",
    66: "Odilon Redon",
    67: "Tintoretto",
    68: "Gene Davis",
    69: "Raphael",
    70: "John Henry Twachtman",
    71: "Henri de Toulouse-Lautrec",
    72: "Antoine Blanchard",
    73: "David Burliuk",
    74: "Camille Corot",
    75: "Konstantin Korovin",
    76: "Ivan Bilibin",
    77: "Titian",
    78: "Maurice Prendergast",
    79: "Edouard Manet",
    80: "Peter Paul Rubens",
    81: "Aubrey Beardsley",
    82: "Paolo Veronese",
    83: "Joshua Reynolds",
    84: "Kuzma Petrov-Vodkin",
    85: "Gustave Caillebotte",
    86: "Lucian Freud",
    87: "Michelangelo",
    88: "Dante Gabriel Rossetti",
    89: "Felix Vallotton",
    90: "Nikolay Bogdanov-Belsky",
    91: "Georges Braque",
    92: "Vasily Surikov",
    93: "Fernand Leger",
    94: "Konstantin Somov",
    95: "Katsushika Hokusai",
    96: "Sir Lawrence Alma-Tadema",
    97: "Vasily Vereshchagin",
    98: "Ernst Ludwig Kirchner",
    99: "Mikhail Vrubel",
    100: "Orest Kiprensky",
    101: "William Merritt Chase",
    102: "Aleksey Savrasov",
    103: "Hans Memling",
    104: "Amedeo Modigliani",
    105: "Ivan Kramskoy",
    106: "Utagawa Kuniyoshi",
    107: "Gustave Courbet",
    108: "William Turner",
    109: "Theo van Rysselberghe",
    110: "Joseph Wright",
    111: "Edward Burne-Jones",
    112: "Koloman Moser",
    113: "Viktor Vasnetsov",
    114: "Anthony van Dyck",
    115: "Raoul Dufy",
    116: "Frans Hals",
    117: "Hans Holbein the Younger",
    118: "Ilya Mashkov",
    119: "Henri Fantin-Latour",
    120: "M.C. Escher",
    121: "El Greco",
    122: "Mikalojus Ciurlionis",
    123: "James McNeill Whistler",
    124: "Karl Bryullov",
    125: "Jacob Jordaens",
    126: "Thomas Gainsborough",
    127: "Eugene Delacroix",
    128: "Canaletto",
}

GENRE_MAP = {
    0: "abstract painting",
    1: "cityscape",
    2: "genre painting",
    3: "illustration",
    4: "landscape",
    5: "nude painting",
    6: "portrait",
    7: "religious painting",
    8: "sketch and study",
    9: "still life",
    10: "painting of unknown genre",
}

STYLE_MAP = {
    0: "abstract expressionism",
    1: "action painting",
    2: "analytical cubism",
    3: "art nouveau",
    4: "baroque",
    5: "color field painting",
    6: "contemporary realism",
    7: "cubism",
    8: "early renaissance",
    9: "expressionism",
    10: "fauvism",
    11: "high renaissance",
    12: "impressionism",
    13: "mannerism late renaissance",
    14: "minimalism",
    15: "naive art primitivism",
    16: "new realism",
    17: "northern renaissance",
    18: "pointillism",
    19: "pop art",
    20: "post impressionism",
    21: "realism",
    22: "rococo",
    23: "romanticism",
    24: "symbolism",
    25: "synthetic cubism",
    26: "ukiyo-e",
}

# Create reverse mappings for filtering
ARTIST_MAP_REVERSE = {name.lower(): idx for idx, name in ARTIST_MAP.items()}
GENRE_MAP_REVERSE = {name.lower(): idx for idx, name in GENRE_MAP.items()}
STYLE_MAP_REVERSE = {name.lower(): idx for idx, name in STYLE_MAP.items()}


def parse_filter_input(
    filter_input: Optional[str], name_map: Dict[str, int], filter_type: str
) -> Set[int]:
    """Parse and validate filter input, returning a set of valid indices."""
    if not filter_input:
        return set()

    names = [name.strip().lower() for name in filter_input.split(",")]
    valid_indices = set()
    invalid_names = []

    for name in names:
        if name in name_map:
            valid_indices.add(name_map[name])
        else:
            invalid_names.append(name)

    if invalid_names:
        print(
            f"Warning: Invalid {filter_type} names provided: {invalid_names}. "
            f"Valid {filter_type}s: {[k for k in name_map.keys() if k != 'unknown artist']}[:5]..."
        )

    return valid_indices


def apply_filtering(
    ds: Dataset, artists: Optional[str], genres: Optional[str], styles: Optional[str]
) -> Dataset:
    """Apply filtering based on artist, genre, and style specifications."""
    artist_indices = parse_filter_input(artists, ARTIST_MAP_REVERSE, "artist")
    genre_indices = parse_filter_input(genres, GENRE_MAP_REVERSE, "genre")
    style_indices = parse_filter_input(styles, STYLE_MAP_REVERSE, "style")

    # Log filter information
    if artist_indices:
        artist_names = [k for k, v in ARTIST_MAP_REVERSE.items() if v in artist_indices]
        print(f"Filtering for artists: {artist_names}")
    if genre_indices:
        genre_names = [k for k, v in GENRE_MAP_REVERSE.items() if v in genre_indices]
        print(f"Filtering for genres: {genre_names}")
    if style_indices:
        style_names = [k for k, v in STYLE_MAP_REVERSE.items() if v in style_indices]
        print(f"Filtering for styles: {style_names}")

    # Create batched filter function
    def batch_filter_fn(batch):
        artist_matches = [
            not artist_indices or artist in artist_indices for artist in batch["artist"]
        ]
        genre_matches = [
            not genre_indices or genre in genre_indices for genre in batch["genre"]
        ]
        style_matches = [
            not style_indices or style in style_indices for style in batch["style"]
        ]

        # Combine all conditions with AND logic
        combined_matches = [
            am and gm and sm
            for am, gm, sm in zip(artist_matches, genre_matches, style_matches)
        ]

        return combined_matches

    # Apply filtering with batching
    filtered_ds = ds.filter(batch_filter_fn, batched=True, batch_size=1000)
    print(f"Dataset filtered: {len(ds)} -> {len(filtered_ds)} examples")

    return filtered_ds


def save_checkpoint(processed_ds: Dataset, next_idx: int) -> None:
    """Save current processing progress"""
    import shutil

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint_data_path = os.path.join(CHECKPOINT_DIR, "processed_data")
    temp_path = os.path.join(CHECKPOINT_DIR, "processed_data.tmp")

    # Save to temporary location first
    processed_ds.save_to_disk(temp_path)

    # Remove old checkpoint if exists
    if os.path.exists(checkpoint_data_path):
        shutil.rmtree(checkpoint_data_path)

    # Atomically move temp to final location
    os.rename(temp_path, checkpoint_data_path)

    # Save metadata
    checkpoint_info = {
        "next_start_idx": next_idx,
        "timestamp": time.time(),
        "total_processed": len(processed_ds),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint_info, f, indent=2)

    print(
        f"Checkpoint saved: {len(processed_ds)} examples processed, next start: {next_idx}"
    )


def load_checkpoint() -> Tuple[Optional[Dataset], int]:
    """Load checkpoint from disk, returns (processed_dataset, next_start_idx)"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "processed_data")

    if not os.path.exists(CHECKPOINT_FILE):
        return None, 0

    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint_info = json.load(f)

    if os.path.exists(checkpoint_path):
        processed_ds = Dataset.load_from_disk(checkpoint_path)
        # Detach from memory-mapped files to allow overwriting
        processed_ds = processed_ds.flatten_indices()
        print(
            f"Resuming from checkpoint: {checkpoint_info['total_processed']} examples already processed"
        )
        return processed_ds, checkpoint_info["next_start_idx"]

    return None, 0


def add_captions(batch: Dict, captioners: List[Dict[str, str]]):
    """Add captions to a batch of examples using multiple VLM models"""
    images = batch["image"]

    # Extract existing captions for all images
    if (
        batch.get(f"{captioners[0]['name']}") is None
    ):  # first round, no existing captions, need to initialize the caption keys
        batch["wikiart-caption"] = [None for _ in range(len(images))]
        for c in captioners:
            batch[f"{c['name']}"] = [None for _ in range(len(images))]
        existing_captions = None
    else:
        existing_captions: List[Dict[str, str]] = []
        for i in range(len(images)):
            # For example, existing = {"qwen-direct": "<caption_content>"}
            existing = {
                f"{c['name']}": batch.get(f"{c['name']}")[i] for c in captioners
            }
            existing_captions.append(existing)

    # Call VLM to caption
    captions_batch = call_parallel(images, captioners, existing_captions)
    time.sleep(SLEEP_TIME)  # avoid rate limit

    # Add wikiart captions and distribute VLM results
    for i in range(len(images)):
        batch["wikiart-caption"][i] = (
            f"{STYLE_MAP[batch['style'][i]]} {GENRE_MAP[batch['genre'][i]]} by {ARTIST_MAP[batch['artist'][i]]}"
        )
        for captioner in captioners:
            batch[f"{captioner['name']}"][i] = captions_batch[i].get(captioner["name"])

    return batch


def has_all_captions(example: Dict, captioners: List[Dict[str, str]]):
    """Check if example has all captions (no None values)"""
    return all(example.get(f"{c['name']}") is not None for c in captioners)


def curate_dataset(
    range_limit: Optional[int],
    batch_size: int,
    chunk_size: int,
    hf_username: str,
    model: Optional[str],
    resume_from: Optional[int] = None,
    max_retries: int = 3,
    artists: Optional[str] = None,
    genres: Optional[str] = None,
    styles: Optional[str] = None,
) -> None:
    """Main pipeline: load dataset, generate captions, filter, and upload to HuggingFace"""
    # Step 1. Load dataset
    ds = load_dataset("huggan/wikiart")["train"]

    # Apply filtering if specified
    if artists or genres or styles:
        ds = apply_filtering(ds, artists, genres, styles)

    # Apply range limit if specified
    if range_limit is not None:
        actual_limit = min(range_limit, len(ds))
        actual_limit = max(1, actual_limit)  # ensure at least 1
        print(f"Limiting dataset to first {actual_limit} examples...")
        ds = ds.select(range(actual_limit))

    if model.lower() == "qwen":
        captioners = CAPTIONERS_QWEN
    elif model.lower() == "mistral":
        captioners = CAPTIONERS_MISTRAL
    else:
        raise ValueError("Model set other than 'qwen' and 'mistral' not defined.")

    # Step 2. Checkpoint-based caption generation with chunked processing
    print("Checking for existing checkpoint...")
    processed_ds, start_idx = load_checkpoint()

    # Handle --resume-from argument to override checkpoint
    if resume_from is not None:
        print(f"Forcing resume from index {resume_from}.")
        if processed_ds is not None:
            # If checkpoint has more data than resume_from, truncate it
            if resume_from < len(processed_ds):
                print(
                    f"Truncating checkpoint data from {len(processed_ds)} to {resume_from} examples."
                )
                processed_ds = processed_ds.select(range(resume_from))
                # Save the truncated dataset as the new checkpoint
                save_checkpoint(processed_ds, resume_from)
            elif resume_from > len(processed_ds):
                print(
                    f"Warning: --resume-from ({resume_from}) is greater than checkpoint size ({len(processed_ds)})."
                )

        start_idx = resume_from

    if processed_ds is None:
        processed_ds = ds.select(range(0))  # empty dataset

    total = len(ds)
    print(f"Starting caption generation from index {start_idx}/{total}...")

    for chunk_start in range(start_idx, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)

        print(f"\nProcessing chunk [{chunk_start}:{chunk_end}] ({chunk_end}/{total})")

        # Process this chunk
        chunk = ds.select(range(chunk_start, chunk_end))
        captioned_chunk = chunk.map(
            lambda batch: add_captions(batch, captioners),
            batched=True,
            batch_size=batch_size,
        )

        # Merge with already processed data
        processed_ds = concatenate_datasets([processed_ds, captioned_chunk])

        # Save checkpoint
        save_checkpoint(processed_ds, chunk_end)

    captioned_ds = processed_ds
    print("\nCaption generation complete!")

    # Step 2.5: Retry examples with missing captions
    print("\nChecking for examples with missing captions...")

    for attempt in range(1, max_retries + 1):
        # Split dataset into complete and incomplete
        complete_ds = captioned_ds.filter(lambda x: has_all_captions(x, captioners))
        incomplete_ds = captioned_ds.filter(
            lambda x: not has_all_captions(x, captioners)
        )

        if len(incomplete_ds) == 0:
            print("All examples have complete captions!")
            break

        print(
            f"\nRetry attempt {attempt}/{max_retries}: {len(incomplete_ds)} examples missing captions"
        )

        # Retry caption generation for incomplete examples
        retried_ds = incomplete_ds.map(
            lambda batch: add_captions(batch, captioners),
            batched=True,
            batch_size=batch_size,
        )

        # Merge back: complete examples + retried examples
        captioned_ds = concatenate_datasets([complete_ds, retried_ds])

        # Save checkpoint after retry
        save_checkpoint(captioned_ds, len(ds))

    print("\nRetry phase complete!")

    # Step 3. Remove images that still have None captions after retries
    print("Removing images that failed all retry attempts...")

    final_ds = captioned_ds.filter(lambda x: has_all_captions(x, captioners))

    total_captions = len(final_ds) * len(captioners)
    print(f"Total captions generated: {total_captions}")
    print(f"Captioners used: {[c['name'] for c in captioners]}")

    # Step 4. Push to HuggingFace
    print("Uploading final dataset to HuggingFace...")
    dataset_length = len(final_ds) // 1000
    final_ds.push_to_hub(f"{hf_username}/wikiart-captions-{dataset_length}k")

    print("Dataset curation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curate WikiArt dataset with VLM-generated captions"
    )
    parser.add_argument(
        "--range",
        type=int,
        default=None,
        metavar="N",
        dest="range_limit",
        help="Limit dataset to first N examples (clipped to [1, dataset_length])",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="Batch size for VLM captioning (default: 4)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        metavar="N",
        help="Checkpoint save frequency in number of examples (default: 100)",
    )
    parser.add_argument(
        "--hf-username",
        type=str,
        required=True,
        metavar="USERNAME",
        help="HuggingFace username for dataset upload",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        metavar="N",
        help="Set of captioner model and prompts (available: ['qwen','mistral'], default: 'qwen')",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=None,
        metavar="INDEX",
        help="Force resumption from a specific index, discarding any corrupted progress beyond it.",
    )
    parser.add_argument(
        "--artists",
        type=str,
        default=None,
        metavar="ARTISTS",
        help="Comma-separated list of artists to include (e.g., 'van gogh, monet, picasso')",
    )
    parser.add_argument(
        "--genres",
        type=str,
        default=None,
        metavar="GENRES",
        help="Comma-separated list of genres to include (e.g., 'portrait, landscape, still life')",
    )
    parser.add_argument(
        "--styles",
        type=str,
        default=None,
        metavar="STYLES",
        help="Comma-separated list of styles to include (e.g., 'impressionism, cubism, realism')",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        metavar="N",
        help="Maximum number of retry attempts for examples with missing captions (default: 3)",
    )

    args = parser.parse_args()

    curate_dataset(
        range_limit=args.range_limit,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        hf_username=args.hf_username,
        model=args.model,
        resume_from=args.resume_from,
        max_retries=args.max_retries,
        artists=args.artists,
        genres=args.genres,
        styles=args.styles,
    )
