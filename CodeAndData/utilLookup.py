

import requests
import json
from typing import Optional, List, Dict


class WikidataPropertyLookup:
    """Helper class for looking up Wikidata properties"""
    
    BASE_URL = "https://www.wikidata.org/w/api.php"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikidataPropertyLookup/1.0'
        })
    
    def get_property_name(self, property_id: str, language: str = "en") -> Optional[Dict]:
        """
        Get property name and description from property ID (e.g., P31)
        
        Args:
            property_id: Wikidata property ID (e.g., "P31")
            language: Language code (default: "en")
            
        Returns:
            Dictionary with label, description, and aliases
        """
        params = {
            'action': 'wbgetentities',
            'ids': property_id,
            'format': 'json',
            'props': 'labels|descriptions|aliases'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'entities' in data and property_id in data['entities']:
                entity = data['entities'][property_id]
                
                result = {
                    'id': property_id,
                    'label': entity.get('labels', {}).get(language, {}).get('value', 'N/A'),
                    'description': entity.get('descriptions', {}).get(language, {}).get('value', 'N/A'),
                    'aliases': [alias['value'] for alias in entity.get('aliases', {}).get(language, [])]
                }
                return result
            else:
                print(f"Property {property_id} not found")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
    
    def search_property_id(self, property_name: str, language: str = "en", limit: int = 10) -> List[Dict]:
        """
        Search for property ID by property name
        
        Args:
            property_name: Name of the property to search for
            language: Language code (default: "en")
            limit: Maximum number of results (default: 10)
            
        Returns:
            List of matching properties with their IDs, labels, and descriptions
        """
        params = {
            'action': 'wbsearchentities',
            'search': property_name,
            'language': language,
            'type': 'property',
            'format': 'json',
            'limit': limit
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if 'search' in data:
                for item in data['search']:
                    results.append({
                        'id': item.get('id'),
                        'label': item.get('label', 'N/A'),
                        'description': item.get('description', 'N/A'),
                        'url': item.get('concepturi', '')
                    })
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching: {e}")
            return []
    
    def get_multiple_properties(self, property_ids: List[str], language: str = "en") -> Dict:
        """
        Get information for multiple properties at once
        
        Args:
            property_ids: List of property IDs
            language: Language code (default: "en")
            
        Returns:
            Dictionary mapping property IDs to their information
        """
        params = {
            'action': 'wbgetentities',
            'ids': '|'.join(property_ids),
            'format': 'json',
            'props': 'labels|descriptions'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = {}
            if 'entities' in data:
                for prop_id, entity in data['entities'].items():
                    results[prop_id] = {
                        'label': entity.get('labels', {}).get(language, {}).get('value', 'N/A'),
                        'description': entity.get('descriptions', {}).get(language, {}).get('value', 'N/A')
                    }
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return {}


def print_property_info(info: Dict):
    """Pretty print property information"""
    print(f"\n{'='*60}")
    print(f"Property ID: {info['id']}")
    print(f"Label: {info['label']}")
    print(f"Description: {info['description']}")
    if info.get('aliases'):
        print(f"Aliases: {', '.join(info['aliases'])}")
    print(f"{'='*60}\n")


def print_search_results(results: List[Dict]):
    """Pretty print search results"""
    sata={"none"}
    aga=""
    print(f"\n{'='*60}")
    aga+=f"\n{'='*60}"
    print(f"Found {len(results)} results:")
    aga+=f"Found {len(results)} results:"
    print(f"{'='*60}")
    aga+=f"{'='*60}"
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['id']} - {result['label']}")
        aga+=f"\n{i}. {result['id']} - {result['label']}"
        sata.add(result['id'])
        print(f"   Description: {result['description']}")
        aga+=f"   Description: {result['description']}"
        print(f"   URL: {result['url']}")
        aga+=f"   URL: {result['url']}"
    print(f"\n{'='*60}\n")
    aga+=f"\n{'='*60}\n"
    
    return aga,sata

def print2_search_results(results: List[Dict]):
    """Pretty print search results"""
    sata={"none"}
    aga=""
    #print(f"\n{'='*60}")
    aga+=f"\n{'='*60}"
    #print(f"Found {len(results)} results:")
    aga+=f"Found {len(results)} results:"
    #print(f"{'='*60}")
    aga+=f"{'='*60}"
    for i, result in enumerate(results, 1):
        #print(f"\n{i}. {result['id']} - {result['label']}")
        aga+=f"\n{i}. {result['id']} - {result['label']}"
        sata.add(result['id'])
        #print(f"   Description: {result['description']}")
        aga+=f"   Description: {result['description']}"
        #print(f"   URL: {result['url']}")
        aga+=f"   URL: {result['url']}"
    #print(f"\n{'='*60}\n")
    aga+=f"\n{'='*60}\n"
    
    return aga,sata
