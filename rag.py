import os 
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np
from typing import List, Tuple, Dict, Optional
import re
import pandas as pd
from dataclasses import dataclass
import PyPDF2  
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter


# Load env variables 
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

@dataclass
class TableData:
    """Data class to store extracted table information."""
    content: pd.DataFrame
    context: str
    table_type: str
    location: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None


class CustomSECQueryEngine:
    """custom query engine with manual cosine similarity."""
    def __init__(self, documents: List[Document], tables: List[TableData], top_k: int = 5):
        self.documents = documents
        self.tables = tables
        self.top_k = top_k
        self.embed_model = OpenAIEmbedding()
        self.llm = OpenAI()
        self.node_parser = SentenceSplitter(chunk_size=3000)
        self.doc_embeddings = None
        self.table_embeddings = None
        self.doc_chunks = []
        self.chunk_to_doc_map = []
        self.initialize_embeddings()

    def initialize_embeddings(self):
        """generate embeddings for all documents and tables."""
        self.doc_chunks = []
        self.chunk_to_doc_map = []
        
        for doc_idx, doc in enumerate(self.documents):
            nodes = self.node_parser.get_nodes_from_documents([doc])
            for node in nodes:
                self.doc_chunks.append(node.text)
                self.chunk_to_doc_map.append(doc_idx)
            
        self.doc_embeddings = np.array([
            self.embed_model.get_text_embedding(chunk)
            for chunk in self.doc_chunks
        ])
        
        # Table embeddings
        table_texts = [
            f"{table.table_type}: {table.context}\n{table.content.to_string()}"
            for table in self.tables
        ]
        self.table_embeddings = np.array([
            self.embed_model.get_text_embedding(text)
            for text in table_texts
        ]) if table_texts else np.array([])

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)

    def get_top_k_similar(self, query_embedding: np.ndarray, 
                         is_financial_query: bool) -> List[Tuple[str, float, bool]]:
        """get top-k similar documents and tables."""
        similarities = []
        
        # Document similarities
        for idx, doc_embedding in enumerate(self.doc_embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append(('doc', idx, similarity))
        
        # Table similarities
        if is_financial_query and len(self.table_embeddings) > 0:
            for idx, table_embedding in enumerate(self.table_embeddings):
                similarity = self.cosine_similarity(query_embedding, table_embedding)
                # Boost table similarity for financial queries
                similarities.append(('table', idx, similarity * 1.2))
        
        # Sort by similarity score in descending order
        return sorted(similarities, key=lambda x: x[2], reverse=True)[:self.top_k]

    def query(self, question: str) -> str:
        """query documents and tables using cosine similarity."""
        # check if question is about financial data
        financial_patterns = [
            r'(?i)(financial|numbers|amount|total|sum|difference)',
            r'(?i)(table|statement|balance sheet|income statement)'
        ]
        is_financial_query = any(re.search(pattern, question) 
                               for pattern in financial_patterns)
        
        # generate embedding for the query
        query_embedding = self.embed_model.get_text_embedding(question)
        
        # get top-k similar documents and tables
        top_matches = self.get_top_k_similar(query_embedding, is_financial_query)
        
        # prepare context from top matches
        context_parts = []
        for type_, idx, sim in top_matches:
            if type_ == 'doc':
                # Use the chunk text and map back to original document
                doc_idx = self.chunk_to_doc_map[idx]
                context_parts.append(
                    f"[Document Similarity: {sim:.4f}]\n{self.doc_chunks[idx]}"
                )
            else:  # table
                table = self.tables[idx]
                context_parts.append(
                    f"[Table Similarity: {sim:.4f}]\nTable Type: {table.table_type}\n"
                    f"Context: {table.context}\nData:\n{table.content.to_string()}"
                )
        
        context = "\n\n".join(context_parts)
        
        
        prompt = f"""You are a financial analyst specialized in analyzing 10-K SEC filings. 
        Based on the following context from the 10-K documents and tables, please answer this question:
        
        Question: {question}
        
        Context:
        {context}
        
        If the information is not available in the context or you're unsure, please explicitly state so.
        Focus on providing factual information from the filings only.
        When referring to financial data, please be specific about the numbers and their context."""
        
        # Use LLM instead of embedding model for completion
        response = self.llm.complete(prompt)
        return response.text




def process_documents(directory: str) -> Tuple[List[Document], List[TableData]]:
    """Load and process SEC documents with table extraction."""
    documents = []
    tables = []
    
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.pdf'):  # Check for PDF files
                continue
                
            file_path = os.path.join(root, file)
            
            try:
                # Read PDF file
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ""
                    # Extract text from each page
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
                
                # Split content into sections based on common SEC filing headers
                sections = re.split(r'\n(?=ITEM\s+\d+\.)', content)
                
                for section in sections:
                    # Extract tables using pattern matching
                    table_matches = re.finditer(
                        r'((?:\s*[-+\d.,()$%]+\s+)+)\n', 
                        section
                    )
                    
                    current_pos = 0
                    for match in table_matches:
                        # Add text before table as document
                        text_before = section[current_pos:match.start()].strip()
                        if text_before:
                            documents.append(Document(text=text_before))
                        
                        # Process potential table content
                        table_text = match.group(1)
                        try:
                            # Convert table text to DataFrame
                            rows = [row.split() for row in table_text.strip().split('\n')]
                            if len(rows) > 1:  # Ensure we have at least 2 rows for a table
                                df = pd.DataFrame(rows[1:], columns=rows[0])
                                
                                # Get context (text before table)
                                context = text_before[-200:] if text_before else ""
                                
                                # Determine table type based on content and context
                                table_type = "Financial Table"
                                if any(keyword in context.lower() for keyword in 
                                     ['balance sheet', 'assets', 'liabilities']):
                                    table_type = "Balance Sheet"
                                elif any(keyword in context.lower() for keyword in 
                                       ['income', 'revenue', 'earnings']):
                                    table_type = "Income Statement"
                                
                                # Create TableData object
                                table_data = TableData(
                                    content=df,
                                    context=context,
                                    table_type=table_type,
                                    location=file_path,
                                    metadata={'section': section[:100]}
                                )
                                tables.append(table_data)
                        
                        except Exception as e:
                            # If table parsing fails, treat as regular text
                            documents.append(Document(text=table_text))
                        
                        current_pos = match.end()
                    
                    # Add remaining text as document
                    remaining_text = section[current_pos:].strip()
                    if remaining_text:
                        documents.append(Document(text=remaining_text))
                        
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
    
    return documents, tables



def main():
    # Process documents and extract tables
    documents, tables = process_documents("./data")
    
    # Create custom query engine with table awareness and cosine similarity
    query_engine = CustomSECQueryEngine(documents, tables)
    
    # Example queries
    questions = [
        "What was the total revenue for the most recent fiscal year?",
        "Show me the company's current assets from the balance sheet.",
        "What are the main business risks mentioned in the filing?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)
        response = query_engine.query(question)
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()
