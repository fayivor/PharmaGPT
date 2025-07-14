"""PubMed retriever for pharmaceutical literature."""

import logging
import time
from typing import List, Dict, Any, Optional
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from dataclasses import dataclass
from config import config

logger = logging.getLogger(__name__)


@dataclass
class PubMedArticle:
    """PubMed article representation."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: str
    doi: Optional[str] = None
    keywords: List[str] = None
    mesh_terms: List[str] = None


class PubMedRetriever:
    """Retriever for PubMed articles using NCBI E-utilities."""
    
    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        """Initialize PubMed retriever.
        
        Args:
            api_key: NCBI API key for higher rate limits.
            email: Email for NCBI (required for API usage).
        """
        self.api_key = api_key or config.PUBMED_API_KEY
        self.email = email or "pharmagpt@example.com"  # Replace with actual email
        self.base_url = config.PUBMED_BASE_URL
        self.session = requests.Session()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_interval = 0.34 if self.api_key else 1.0  # 3/sec with key, 1/sec without
        
        logger.info("Initialized PubMed retriever")
    
    def search(self, query: str, max_results: int = 20, filters: Optional[Dict[str, str]] = None) -> List[str]:
        """Search PubMed and return PMIDs.
        
        Args:
            query: Search query.
            max_results: Maximum number of results.
            filters: Additional search filters.
            
        Returns:
            List of PMIDs.
        """
        # Build search parameters
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'xml',
            'tool': 'PharmaGPT',
            'email': self.email
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        # Add filters
        if filters:
            for key, value in filters.items():
                if key == 'date_range':
                    params['datetype'] = 'pdat'
                    params['mindate'] = value.get('start', '')
                    params['maxdate'] = value.get('end', '')
                elif key == 'article_type':
                    params['term'] += f' AND {value}[pt]'
                elif key == 'journal':
                    params['term'] += f' AND "{value}"[ta]'
        
        try:
            self._rate_limit()
            
            url = f"{self.base_url}esearch.fcgi"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            pmids = [id_elem.text for id_elem in root.findall('.//Id')]
            
            logger.info(f"Found {len(pmids)} PMIDs for query: {query[:50]}...")
            return pmids
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def fetch_articles(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch full article details for given PMIDs.
        
        Args:
            pmids: List of PubMed IDs.
            
        Returns:
            List of PubMedArticle objects.
        """
        if not pmids:
            return []
        
        # Batch PMIDs (max 200 per request)
        batch_size = 200
        articles = []
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            batch_articles = self._fetch_batch(batch_pmids)
            articles.extend(batch_articles)
        
        logger.info(f"Fetched {len(articles)} articles")
        return articles
    
    def _fetch_batch(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch a batch of articles.
        
        Args:
            pmids: List of PMIDs to fetch.
            
        Returns:
            List of PubMedArticle objects.
        """
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'rettype': 'abstract',
            'tool': 'PharmaGPT',
            'email': self.email
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            self._rate_limit()
            
            url = f"{self.base_url}efetch.fcgi"
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            
            return self._parse_articles(response.content)
            
        except Exception as e:
            logger.error(f"Failed to fetch articles: {e}")
            return []
    
    def _parse_articles(self, xml_content: bytes) -> List[PubMedArticle]:
        """Parse XML response into PubMedArticle objects.
        
        Args:
            xml_content: XML response content.
            
        Returns:
            List of parsed articles.
        """
        articles = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article_elem in root.findall('.//PubmedArticle'):
                try:
                    article = self._parse_single_article(article_elem)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to parse XML: {e}")
        
        return articles
    
    def _parse_single_article(self, article_elem) -> Optional[PubMedArticle]:
        """Parse a single article element.
        
        Args:
            article_elem: XML element for the article.
            
        Returns:
            PubMedArticle object or None if parsing fails.
        """
        try:
            # Extract PMID
            pmid_elem = article_elem.find('.//PMID')
            if pmid_elem is None:
                return None
            pmid = pmid_elem.text
            
            # Extract title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            # Extract abstract
            abstract_parts = []
            for abstract_elem in article_elem.findall('.//AbstractText'):
                if abstract_elem.text:
                    abstract_parts.append(abstract_elem.text)
            abstract = " ".join(abstract_parts)
            
            # Extract authors
            authors = []
            for author_elem in article_elem.findall('.//Author'):
                last_name = author_elem.find('LastName')
                first_name = author_elem.find('ForeName')
                if last_name is not None:
                    author_name = last_name.text
                    if first_name is not None:
                        author_name += f", {first_name.text}"
                    authors.append(author_name)
            
            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract publication date
            pub_date_elem = article_elem.find('.//PubDate')
            pub_date = ""
            if pub_date_elem is not None:
                year = pub_date_elem.find('Year')
                month = pub_date_elem.find('Month')
                day = pub_date_elem.find('Day')
                
                if year is not None:
                    pub_date = year.text
                    if month is not None:
                        pub_date += f"-{month.text}"
                        if day is not None:
                            pub_date += f"-{day.text}"
            
            # Extract DOI
            doi = None
            for id_elem in article_elem.findall('.//ArticleId'):
                if id_elem.get('IdType') == 'doi':
                    doi = id_elem.text
                    break
            
            # Extract MeSH terms
            mesh_terms = []
            for mesh_elem in article_elem.findall('.//MeshHeading/DescriptorName'):
                if mesh_elem.text:
                    mesh_terms.append(mesh_elem.text)
            
            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=pub_date,
                doi=doi,
                mesh_terms=mesh_terms
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse article: {e}")
            return None
    
    def _rate_limit(self) -> None:
        """Implement rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def search_and_fetch(self, query: str, max_results: int = 20, filters: Optional[Dict[str, str]] = None) -> List[PubMedArticle]:
        """Search and fetch articles in one call.
        
        Args:
            query: Search query.
            max_results: Maximum number of results.
            filters: Additional search filters.
            
        Returns:
            List of PubMedArticle objects.
        """
        pmids = self.search(query, max_results, filters)
        return self.fetch_articles(pmids)
