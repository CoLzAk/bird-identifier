# api/species_utils.py
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional


def load_species_with_thumbnails() -> List[Dict]:
    """
    Load species data from species.json and encode thumbnails as base64.

    Returns:
        List of species dictionaries with base64-encoded thumbnails
    """
    species_file = Path(__file__).parent / "species" / "species.json"

    if not species_file.exists():
        return []

    try:
        with open(species_file, 'r', encoding='utf-8') as f:
            species_list = json.load(f)

        # Load thumbnails and convert to base64
        for species in species_list:
            thumbnail_path = species.get('thumbnail')
            if thumbnail_path:
                # Construct full path
                full_path = Path(__file__).parent / thumbnail_path

                # Load and encode image
                if full_path.exists():
                    try:
                        with open(full_path, 'rb') as img_file:
                            image_data = img_file.read()
                            # Encode as base64
                            base64_image = base64.b64encode(image_data).decode('utf-8')

                            # Determine image format from file extension
                            ext = full_path.suffix.lower()
                            mime_type = {
                                '.jpg': 'image/jpeg',
                                '.jpeg': 'image/jpeg',
                                '.png': 'image/png',
                                '.gif': 'image/gif',
                                '.webp': 'image/webp'
                            }.get(ext, 'image/jpeg')

                            # Store as data URI
                            species['thumbnail_base64'] = f"data:{mime_type};base64,{base64_image}"
                    except Exception as e:
                        print(f"⚠️ Failed to load thumbnail for {species['name']}: {e}")
                        species['thumbnail_base64'] = None
                else:
                    species['thumbnail_base64'] = None
            else:
                species['thumbnail_base64'] = None

        return species_list

    except Exception as e:
        print(f"⚠️ Failed to load species data: {e}")
        return []
