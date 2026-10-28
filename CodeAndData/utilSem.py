import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import numpy as np

SESSION = requests.Session()

# Retries for transient errors
retries = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
SESSION.mount("https://", HTTPAdapter(max_retries=retries))

HEADERS = {
    "User-Agent": "MyWikidataLookupScript/1.0 (your.email@example.com)"
}

# Load sentence transformer model (this will download on first run)
print("Loading semantic similarity model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and effective model
print("Model loaded!\n")

def get_wikidata_id(name, language="en", limit=10):
    """
    Search Wikidata by name and return ALL matches (list of dicts with QID, label, description).
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": language,
        "format": "json",
        "type": "item",
        "limit": limit,
        "search": name,
    }
    resp = SESSION.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("search", [])
    
    if not results:
        return []
    
    # Return all results as a list of dicts
    return [
        {
            "id": item["id"],
            "label": item.get("label", ""),
            "description": item.get("description", ""),
            "match": item.get("match", {})
        }
        for item in results
    ]

def fuzzy_match_score(search_term, label):
    """
    Calculate fuzzy string matching score between search term and label (0-1).
    Higher score means better match.
    """
    return SequenceMatcher(None, search_term.lower(), label.lower()).ratio()

def phrase_contains_label(phrase, label):
    """
    Check if the phrase contains the label as a substring (case-insensitive).
    Returns True if label is mentioned in the phrase.
    """
    return label.lower() in phrase.lower()

def is_generic_entity(description):
    """
    Check if entity is a generic type (given name, family name, etc.)
    These should be penalized unless the context specifically asks for them.
    """
    generic_types = [
        'given name', 'family name', 'surname', 'first name',
        'male given name', 'female given name', 'unisex given name'
    ]
    return any(generic in description.lower() for generic in generic_types)

def calculate_label_embedding_similarity(user_phrase, label):
    """
    Calculate semantic similarity between the user phrase and the entity label.
    This helps identify if the label is conceptually related to the phrase.
    """
    phrase_embedding = model.encode([user_phrase])
    label_embedding = model.encode([label])
    similarity = cosine_similarity(phrase_embedding, label_embedding).flatten()[0]
    return float(similarity)

def calculate_combined_similarity(search_term, user_phrase, results, 
                                  label_weight=0.2, 
                                  semantic_weight=0.5,
                                  label_phrase_weight=0.3):
    """
    Calculate combined similarity score using:
    1. Label matching (fuzzy string match with search term) - 20%
    2. Semantic similarity (between user phrase and description) - 50%
    3. Label-phrase similarity (semantic match between phrase and label) - 30%
    
    Plus penalties for generic entities and bonuses for phrase mentions.
    
    Returns results sorted by combined score (highest first).
    """
    if not results:
        return []
    
    # Filter out items without descriptions
    valid_results = [r for r in results if r['description']]
    
    if not valid_results:
        print("Warning: No items with descriptions found.")
        return []
    
    # Get all descriptions
    descriptions = [item['description'] for item in valid_results]
    
    # Encode user phrase and all descriptions into embeddings
    user_embedding = model.encode([user_phrase])
    description_embeddings = model.encode(descriptions)
    
    # Calculate cosine similarity between user phrase and each description
    semantic_similarities = cosine_similarity(user_embedding, description_embeddings).flatten()
    
    # Calculate combined scores
    for i, item in enumerate(valid_results):
        # 1. Label matching score (0-1) - exact match with search term
        label_score = fuzzy_match_score(search_term, item['label'])
        
        # 2. Semantic similarity score (description vs phrase)
        semantic_score = float(semantic_similarities[i])
        semantic_score_normalized = (semantic_score + 1) / 2
        
        # 3. Label-phrase similarity - does the phrase mention or relate to this label?
        label_phrase_score = calculate_label_embedding_similarity(user_phrase, item['label'])
        label_phrase_score_normalized = (label_phrase_score + 1) / 2
        
        # Bonus: If the phrase explicitly contains the label, significant boost
        phrase_mention_bonus = 0.0
        if phrase_contains_label(user_phrase, item['label']):
            phrase_mention_bonus = 0.4  # Major boost for direct mentions
        
        # Penalty: Generic entities (given names, family names) get penalized
        generic_penalty = 0.0
        if is_generic_entity(item['description']):
            generic_penalty = 0.3  # Reduce score for generic entities
        
        # Combined score with weights, bonuses, and penalties
        combined_score = (
            label_weight * label_score + 
            semantic_weight * semantic_score_normalized +
            label_phrase_weight * label_phrase_score_normalized +
            phrase_mention_bonus -
            generic_penalty
        )
        
        # Store all scores
        item['label_score'] = label_score
        item['semantic_similarity'] = semantic_score
        item['label_phrase_similarity'] = label_phrase_score
        item['phrase_mention_bonus'] = phrase_mention_bonus
        item['generic_penalty'] = generic_penalty
        item['combined_score'] = combined_score
    
    # Sort by combined score (descending)
    valid_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return valid_results

# Example
if __name__ == "__main__":
    try:
        # Get search term from user
        search_term = input("Enter a search term (e.g., 'Hercules'): ").strip()
        if not search_term:
            print("No search term provided.")
            exit()
        
        # Get user phrase for similarity comparison
        user_phrase = input("Enter a phrase to compare (e.g., 'strong warrior from mythology'): ").strip()
        if not user_phrase:
            print("No comparison phrase provided.")
            exit()
        
        print(f"\nSearching for '{search_term}'...")
        results = get_wikidata_id(search_term, limit=10)
        
        if results:
            print(f"Found {len(results)} result(s). Calculating combined scores...\n")
            
            # Calculate combined similarities with adjusted weights
            # label_weight: How much the search term matches the label
            # semantic_weight: How much the description relates to the phrase
            # label_phrase_weight: How much the label itself relates to the phrase
            ranked_results = calculate_combined_similarity(
                search_term, 
                user_phrase, 
                results, 
                label_weight=0.2,        # 20% - exact label matching
                semantic_weight=0.5,      # 50% - description similarity
                label_phrase_weight=0.3   # 30% - label mentioned in phrase
            )
            
            if ranked_results:
                print(f"Results ranked by combined score:")
                print(f"Phrase: '{user_phrase}'")
                print("=" * 130)
                for i, item in enumerate(ranked_results, 1):
                    bonus_str = f" +Bonus: {item['phrase_mention_bonus']:.2f}" if item['phrase_mention_bonus'] > 0 else ""
                    penalty_str = f" -Penalty: {item['generic_penalty']:.2f}" if item['generic_penalty'] > 0 else ""
                    
                    print(f"{i}. {item['label']} ({item['id']}) - Combined Score: {item['combined_score']:.4f}{bonus_str}{penalty_str}")
                    print(f"   Label Match: {item['label_score']:.4f} | Desc Similarity: {item['semantic_similarity']:.4f} | Label-Phrase: {item['label_phrase_similarity']:.4f}")
                    print(f"   Description: {item['description']}")
                    print("-" * 130)
            else:
                print("No valid results with descriptions to compare.")
        else:
            print("No results found.")
            
    except requests.HTTPError as e:
        print("HTTP error:", e, "\nBody:", e.response.text[:400])
    except requests.RequestException as e:
        print("Request failed:", e)
    except Exception as e:
        print("Error:", e)