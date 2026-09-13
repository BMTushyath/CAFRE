import json
import os
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError
import ollama

logger = logging.getLogger(__name__)

# Pydantic Schema for Ontology Validation
class CKEOntology(BaseModel):
    domain: str = Field(..., description="The name of the domain (e.g., Finance, Healthcare)")
    protected_attributes: List[str] = Field(..., description="List of attributes protected by law or ethics in this domain.")
    historical_biases: List[str] = Field(..., description="Examples of historical bias risks in this domain.")
    known_proxies: Dict[str, List[str]] = Field(..., description="Mapping of protected attributes to common proxy variables.")

class CKEGenerator:
    """
    Context Knowledge Engine (CKE) Generator.
    Autonomously generates and caches domain-specific fairness rules using a local LLM.
    """
    def __init__(self, cache_dir: str = None, model_name: str = 'llama3.1'):
        self.model_name = model_name
        
        if cache_dir is None:
            # Default cache dir is CAFRE/knowledge/ontologies/
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.cache_dir = os.path.join(base_dir, 'ontologies')
        else:
            self.cache_dir = cache_dir
            
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, domain: str) -> str:
        safe_domain = domain.lower().replace(" ", "_").replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe_domain}_ontology.json")

    def _load_cached_ontology(self, domain: str) -> Optional[Dict]:
        path = self._get_cache_path(domain)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Re-validate structure upon loading
                    validated = CKEOntology(**data)
                    logger.info(f"Loaded cached CKE ontology for domain: {domain}")
                    return validated.model_dump()
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Cached CKE ontology for {domain} is invalid. Regenerating. Error: {e}")
        return None

    def _generate_ontology_with_llm(self, domain: str) -> Dict:
        """
        Invokes local LLM deterministically to generate the domain ontology.
        """
        logger.info(f"Generating new CKE ontology for domain: {domain} using {self.model_name}")
        
        prompt = f"""
You are an expert AI fairness researcher and legal compliance auditor.
Your task is to generate a comprehensive fairness ontology for the domain: "{domain}".
You MUST output ONLY a valid JSON object matching exactly this schema:
{{
    "domain": "{domain}",
    "protected_attributes": ["list", "of", "strings"],
    "historical_biases": ["list", "of", "strings describing historical bias issues"],
    "known_proxies": {{
        "protected_attribute_1": ["proxy_variable_1", "proxy_variable_2"]
    }}
}}
Output nothing else. No markdown formatting, no explanations. Just raw JSON.
"""
        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0.0,
                'seed': 42 # Ensure deterministic output
            }
        )
        
        content = response.get('message', {}).get('content', '')
        
        # Clean up potential markdown formatting if the LLM ignores instructions
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        try:
            parsed_json = json.loads(content.strip())
            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(f"LLM produced invalid JSON for {domain}: {content}")
            raise ValueError(f"Failed to parse LLM output as JSON: {e}")

    def get_ontology(self, domain: str) -> Dict:
        """
        Main entrypoint. Checks cache, generates if missing, validates, caches, and returns.
        """
        if not domain or domain.lower() == "unknown":
            logger.warning("Cannot generate CKE ontology for 'Unknown' domain.")
            return {}

        # 1. Check cache
        cached = self._load_cached_ontology(domain)
        if cached:
            return cached

        # 2. Invoke LLM if not cached
        raw_ontology = self._generate_ontology_with_llm(domain)

        # 3. Validation Pipeline
        try:
            validated_ontology = CKEOntology(**raw_ontology)
        except ValidationError as e:
            logger.error(f"LLM generated CKE ontology failed structural validation: {e}")
            raise ValueError(f"CKE ontology validation failed for {domain}: {e}")

        # 4. Duplicate Detection (Internal constraint checks)
        # Ensure protected attributes in known_proxies actually exist in the protected_attributes list
        for protected_attr in validated_ontology.known_proxies.keys():
            if protected_attr not in validated_ontology.protected_attributes:
                logger.warning(f"LLM generated CKE proxy for unknown attribute '{protected_attr}'. Auto-adding to protected list.")
                validated_ontology.protected_attributes.append(protected_attr)
                
        # Deduplicate lists preserving order to avoid test flakiness
        validated_ontology.protected_attributes = list(dict.fromkeys(validated_ontology.protected_attributes))

        # 5. Cache validated ontology
        final_dict = validated_ontology.model_dump()
        path = self._get_cache_path(domain)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(final_dict, f, indent=4)
            
        logger.info(f"Successfully generated and cached CKE ontology for {domain}")
        return final_dict
