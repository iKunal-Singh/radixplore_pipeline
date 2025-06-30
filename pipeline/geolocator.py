import json
import os
import re
import time

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
except ImportError:
    raise ImportError("geopy not found. Please run 'pip install -r requirements.txt'.")


def geocode_location(location_name: str, geolocator) -> list:
    """Geocodes a location name using Nominatim, returning a list of candidates."""
    try:
        locations = geolocator.geocode(location_name, exactly_one=False, limit=3)
        time.sleep(1) # Adhere to Nominatim's usage policy
        if not locations: return []
        return [{"name": loc.address, "latitude": loc.latitude, "longitude": loc.longitude} for loc in locations]
    except (GeocoderTimedOut, GeocoderUnavailable):
        print(f"Warning: Geocoding service timed out for '{location_name}'.")
        return []
    except Exception as e:
        print(f"An error occurred during geocoding for '{location_name}': {e}")
        return []

def simulate_llm_location_extraction(context: str) -> list:
    """Simulates an LLM call to extract location names using regex."""
    potential_locations = re.findall(r'\b([A-Z][A-Za-z\'-]+(?: [A-Z][A-Za-z\'-]+)*)\b', context)
    stop_words = {"The", "A", "An", "This", "Project", "Company"}
    locations = [loc for loc in potential_locations if loc not in stop_words and len(loc) > 2]
    if "WA" in context and "Western Australia" not in locations:
        locations.append("Western Australia")
    return list(set(locations))

def simulate_llm_disambiguation(context: str, candidates: list) -> dict:
    """Simulates an LLM call to choose the best coordinates and provide confidence."""
    if not candidates:
        return {"chosen_location": None, "coordinates": None, "geolocation_confidence": 0.0,
                "justification": "No location candidates were provided to the simulation."}

    best_candidate, max_score = None, -1
    for cand in candidates:
        score = 0
        cand_name_lower = cand['name'].lower()
        if 'western australia' in context.lower() and 'western australia' in cand_name_lower: score += 0.8
        elif 'wa' in context and 'western australia' in cand_name_lower: score += 0.8
        if 'australia' in context.lower() and 'australia' in cand_name_lower: score += 0.5
        if score > max_score: max_score, best_candidate = score, cand

    if best_candidate and max_score > 0.5:
        confidence = min(round(max_score, 4), 1.0)
        return {"chosen_location": best_candidate['name'], "coordinates": [best_candidate['latitude'], best_candidate['longitude']], "geolocation_confidence": confidence, "justification": f"Simulation selected candidate based on high contextual match (score: {confidence})."}
    
    first_cand = candidates[0]
    return {"chosen_location": first_cand['name'], "coordinates": [first_cand['latitude'], first_cand['longitude']], "geolocation_confidence": 0.25, "justification": "Context was ambiguous. Simulation defaulted to the first available result."}

def run_geolocation_pipeline(input_file: str, output_file: str):
    """Reads NER output, geocodes, disambiguates, and writes final enriched output."""
    print("\n--- Starting Geolocation Pipeline ---")
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found. Halting."); return
    
    if os.path.exists(output_file): os.remove(output_file)

    geolocator = Nominatim(user_agent=f"RadiXploreChallenge_Pipeline/{time.time()}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        records = [json.loads(line) for line in f_in]

    for i, record in enumerate(records):
        context = record['context_sentence']
        print(f"Geolocating record {i+1}/{len(records)}: {record['project_name'][:40]}...")

        location_names = simulate_llm_location_extraction(context)
        if not location_names: continue

        all_candidates = []
        for name in location_names:
            all_candidates.extend(geocode_location(name, geolocator))
        
        if not all_candidates: continue
        
        final_choice = simulate_llm_disambiguation(context, all_candidates)
        record.update(final_choice)
        
        with open(output_file, 'a', encoding='utf-8') as f_out:
            json.dump(record, f_out); f.write('\n')

    print(f"\nGeolocation pipeline complete. Final output saved to '{output_file}'.")
