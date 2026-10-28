import json
from rdflib.plugins.sparql import prepareQuery
from rdflib import URIRef
from rdflib.paths import Path
from rdflib.term import Variable, Literal, BNode
def extract_entities_from_query(sparql_query):
    """
    Extract entities (subjects and objects) from a single SPARQL query using RDFLib.
    Entities are collected only when they are NOT variables (i.e., concrete URIRefs or Literals).
    Blank nodes are ignored.

    Args:
        sparql_query (str): SPARQL query string

    Returns:
        dict: {
            'subject_uris': [str, ...],
            'object_uris': [str, ...],
            'object_literals': [str, ...]
        }
    """
    try:
        # Reuse the same common prefixes pattern you have
        common_prefixes = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
"""
        if 'PREFIX' not in sparql_query.upper():
            sparql_query = common_prefixes + sparql_query
        else:
            for line in common_prefixes.strip().split('\n'):
                prefix_name = line.split(':')[0].replace('PREFIX', '').strip()
                if f'PREFIX {prefix_name}:' not in sparql_query and f'prefix {prefix_name}:' not in sparql_query.lower():
                    sparql_query = line + '\n' + sparql_query

        parsed = prepareQuery(sparql_query)

        subject_uris = set()
        object_uris = set()
        object_literals = set()

        def collect_entity(term, position):
            """
            Add a term to the appropriate set based on its type and triple position.
            Skip variables and blank nodes; accept URIRefs; accept Literals for objects.
            """
            # Variables -> skip
            if isinstance(term, Variable):
                return
            # Blank nodes -> skip
            if isinstance(term, BNode):
                return
            # Concrete URIs
            if isinstance(term, URIRef):
                if position == 's':
                    subject_uris.add(str(term))
                elif position == 'o':
                    object_uris.add(str(term))
                return
            # Literals only allowed in object position
            if position == 'o' and isinstance(term, Literal):
                # store lexical form; you can also keep language/datatype if you want
                object_literals.add(str(term))

        def find_entities(node):
            # CompValue-like nodes
            if hasattr(node, 'get') or hasattr(node, '__getitem__'):
                if hasattr(node, 'get'):
                    triples = node.get('triples')
                else:
                    try:
                        triples = node['triples']
                    except (KeyError, TypeError):
                        triples = None

                if triples:
                    for triple in triples:
                        # triple shape: (subject, predicate, object, ...)
                        if len(triple) >= 3:
                            s, _, o = triple[0], triple[1], triple[2]
                            collect_entity(s, 's')
                            collect_entity(o, 'o')

                # Recurse all values
                if hasattr(node, 'values'):
                    for value in node.values():
                        if value is not None:
                            if isinstance(value, (list, tuple)):
                                for item in value:
                                    find_entities(item)
                            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, URIRef, Literal)):
                                find_entities(value)

            # Regular objects
            elif hasattr(node, '__dict__'):
                for attr_value in node.__dict__.values():
                    if attr_value is not None:
                        if isinstance(attr_value, (list, tuple)):
                            for item in attr_value:
                                find_entities(item)
                        elif hasattr(attr_value, '__dict__') and not isinstance(attr_value, (str, int, URIRef, Literal)):
                            find_entities(attr_value)

        find_entities(parsed.algebra)

        return {
            'subject_uris': sorted(subject_uris),
            'object_uris': sorted(object_uris),
            'object_literals': sorted(object_literals),
        }

    except Exception as e:
        print(f"Error parsing query for entities: {e}")
        return {
            'subject_uris': [],
            'object_uris': [],
            'object_literals': [],
        }

def extract2_predicates_from_query(sparql_query):
    """
    Extract predicates from a single SPARQL query using RDFLib
    
    Args:
        sparql_query (str): SPARQL query string
        
    Returns:
        list: List of unique predicates found in the query
    """
    try:
        # Add common Wikidata prefixes if not already present
        common_prefixes = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>

"""
        # Only add prefixes if they're not already in the query
        if 'PREFIX' not in sparql_query.upper():
            sparql_query = common_prefixes + sparql_query
        else:
            # Add missing prefixes
            for line in common_prefixes.strip().split('\n'):
                prefix_name = line.split(':')[0].replace('PREFIX', '').strip()
                if f'PREFIX {prefix_name}:' not in sparql_query and f'prefix {prefix_name}:' not in sparql_query.lower():
                    sparql_query = line + '\n' + sparql_query
        
        # Parse the query
        parsed = prepareQuery(sparql_query)
        
        predicates = set()
        
        def extract_from_path(pred):
            """Extract predicates from URIs and property paths"""
            # Direct URI predicate
            if isinstance(pred, URIRef):
                predicates.add(str(pred))
            # Property path
            elif isinstance(pred, Path):
                # Handle different path types
                if hasattr(pred, 'arg'):
                    extract_from_path(pred.arg)
                if hasattr(pred, 'args'):
                    for arg in pred.args:
                        extract_from_path(arg)
                if hasattr(pred, 'path'):
                    extract_from_path(pred.path)
        
        # Traverse the algebra tree to find predicates
        def find_predicates(node):
            # CompValue objects are dict-like
            if hasattr(node, 'get') or hasattr(node, '__getitem__'):
                # Check for triples in BGP (Basic Graph Pattern)
                if hasattr(node, 'get'):
                    triples = node.get('triples')
                else:
                    try:
                        triples = node['triples']
                    except (KeyError, TypeError):
                        triples = None
                
                if triples:
                    for triple in triples:
                        if len(triple) >= 2:
                            pred = triple[1]
                            extract_from_path(pred)
                
                # Recursively explore all values
                if hasattr(node, 'values'):
                    for value in node.values():
                        if value is not None:
                            if isinstance(value, (list, tuple)):
                                for item in value:
                                    find_predicates(item)
                            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, URIRef)):
                                find_predicates(value)
            
            # Regular objects
            elif hasattr(node, '__dict__'):
                for attr_value in node.__dict__.values():
                    if attr_value is not None:
                        if isinstance(attr_value, (list, tuple)):
                            for item in attr_value:
                                find_predicates(item)
                        elif hasattr(attr_value, '__dict__') and not isinstance(attr_value, (str, int, URIRef)):
                            find_predicates(attr_value)
        
        find_predicates(parsed.algebra)
        
        return sorted(list(predicates))
    
    except Exception as e:
        print(f"Error parsing query: {e}")
        return []


def extract_predicates_from_query(sparql_query):
    """
    Extract predicates from a single SPARQL query using RDFLib
    
    Args:
        sparql_query (str): SPARQL query string
        
    Returns:
        list: List of unique predicates found in the query
    """
    try:
        # Add common Wikidata prefixes if not already present
        common_prefixes = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bd: <http://www.bigdata.com/rdf#>

"""
        # Only add prefixes if they're not already in the query
        if 'PREFIX' not in sparql_query.upper():
            sparql_query = common_prefixes + sparql_query
        else:
            # Add missing prefixes
            for line in common_prefixes.strip().split('\n'):
                prefix_name = line.split(':')[0].replace('PREFIX', '').strip()
                if f'PREFIX {prefix_name}:' not in sparql_query and f'prefix {prefix_name}:' not in sparql_query.lower():
                    sparql_query = line + '\n' + sparql_query
        
        # Parse the query
        parsed = prepareQuery(sparql_query)
        
        predicates = set()
        
        def extract_from_path(pred):
            """Extract predicates from URIs and property paths"""
            # Direct URI predicate
            if isinstance(pred, URIRef):
                predicates.add(str(pred))
            # Property path
            elif isinstance(pred, Path):
                # Handle different path types
                if hasattr(pred, 'arg'):
                    extract_from_path(pred.arg)
                if hasattr(pred, 'args'):
                    for arg in pred.args:
                        extract_from_path(arg)
                if hasattr(pred, 'path'):
                    extract_from_path(pred.path)
        
        # Traverse the algebra tree to find predicates
        def find_predicates(node):
            # CompValue objects are dict-like
            if hasattr(node, 'get') or hasattr(node, '__getitem__'):
                # Check for triples in BGP (Basic Graph Pattern)
                if hasattr(node, 'get'):
                    triples = node.get('triples')
                else:
                    try:
                        triples = node['triples']
                    except (KeyError, TypeError):
                        triples = None
                
                if triples:
                    for triple in triples:
                        if len(triple) >= 2:
                            pred = triple[1]
                            extract_from_path(pred)
                
                # Recursively explore all values
                if hasattr(node, 'values'):
                    for value in node.values():
                        if value is not None:
                            if isinstance(value, (list, tuple)):
                                for item in value:
                                    find_predicates(item)
                            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, URIRef)):
                                find_predicates(value)
            
            # Regular objects
            elif hasattr(node, '__dict__'):
                for attr_value in node.__dict__.values():
                    if attr_value is not None:
                        if isinstance(attr_value, (list, tuple)):
                            for item in attr_value:
                                find_predicates(item)
                        elif hasattr(attr_value, '__dict__') and not isinstance(attr_value, (str, int, URIRef)):
                            find_predicates(attr_value)
        
        find_predicates(parsed.algebra)
        
        return sorted(list(predicates))
    
    except Exception as e:
        print(f"Error parsing query: {e}")
        return []


def extract_predicates_from_json(json_file_path):
    """
    Extract predicates from all SPARQL queries in a JSON file
    
    Args:
        json_file_path (str): Path to JSON file containing SPARQL queries
        
    Returns:
        tuple: (results dict, all_predicates list)
    """
    try:
        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        results = {}
        all_predicates = set()
        
        # Process each query
        for query_id, sparql_query in queries.items():
            print(f"\nProcessing query {query_id}...")
            predicates = extract_predicates_from_query(sparql_query)
            results[query_id] = predicates
            all_predicates.update(predicates)
            
            if predicates:
                print(f"  Found {len(predicates)} predicate(s):")
                for pred in predicates:
                    print(f"    - {pred}")
            else:
                print("  No predicates found or error parsing query")
        
        return results, sorted(list(all_predicates))
    
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found")
        return {}, []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{json_file_path}'")
        return {}, []
    except Exception as e:
        print(f"Error processing file: {e}")
        return {}, []


def save_results(results, all_predicates, output_file='predicates_output.json'):
    """
    Save extracted predicates to a JSON file
    
    Args:
        results (dict): Dictionary of predicates per query
        all_predicates (list): List of all unique predicates
        output_file (str): Output file path
    """
    output = {
        'predicates_by_query': results,
        'all_unique_predicates': all_predicates,
        'statistics': {
            'total_queries': len(results),
            'total_unique_predicates': len(all_predicates)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    # Input JSON file containing SPARQL queries
    # input_file = "query.json"
    #
    # print(f"Reading SPARQL queries from '{input_file}'...")
    # print("=" * 70)
    #
    # # Extract predicates
    # results, all_predicates = extract_predicates_from_json(input_file)
    #
    # # Print summary
    # print("\n" + "=" * 70)
    # print("SUMMARY")
    # print("=" * 70)
    # print(f"Total queries processed: {len(results)}")
    # print(f"Total unique predicates: {len(all_predicates)}")
    #
    # if all_predicates:
    #     print("\nAll unique predicates found:")
    #     for i, pred in enumerate(all_predicates, 1):
    #         print(f"  {i}. {pred}")
    #
    # # Save results
    # save_results(results, all_predicates)


    query_id=1;
    results=[]
    all_predicates=[]
    sparql_query="SELECT (COUNT(DISTINCT ?article) AS ?count)\nWHERE {?article wdt:P31/wdt:P279* wd:Q95074}"

    predicates = extract_predicates_from_query(sparql_query)


    
    if predicates:
        print(f"  Found {len(predicates)} predicate(s):")
        for pred in predicates:
            print(f"    - {pred}")
    else:
        print("  No predicates found or error parsing query")

    print("\n✓ Processing complete!")
    
    
    
    sparql_query = "SELECT (COUNT(DISTINCT ?article) AS ?count) WHERE { ?article wdt:P31/wdt:P279* wd:Q95074 }"

    predicates = extract_predicates_from_query(sparql_query)
    entities = extract_entities_from_query(sparql_query)

    if predicates:
        print(f"Found {len(predicates)} predicate(s):")
        for p in predicates:
            print(f"  - {p}")
    else:
        print("No predicates found")

    print("\nEntities (non-variables):")
    print("Subjects (URIs):")
    for s in entities['subject_uris']:
        print(f"  - {s}")
    print("Objects (URIs):")
    for o in entities['object_uris']:
        print(f"  - {o}")
    print("Objects (literals):")
    for lit in entities['object_literals']:
        print(f"  - {lit}")
