import streamlit as st
import os
import base64
from typing import List, Dict, Any
import tempfile
import asyncio
import concurrent.futures
from functools import partial
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import json
import re

# Import required libraries
from unstructured.partition.pdf import partition_pdf
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

def setup_api_keys():
    """Setup API keys from Streamlit secrets"""
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
    except KeyError as e:
        st.error(f"""
        Missing API key in secrets.toml: {str(e)}
        Please add the following to your .streamlit/secrets.toml file:
        
        GOOGLE_API_KEY = "your_google_api_key"
        GROQ_API_KEY = "your_groq_api_key"
        LANGCHAIN_API_KEY = "your_langchain_api_key"
        LANGCHAIN_TRACING_V2 = "true"
        """)
        st.stop()

def setup_retriever():
    """Initialize the multi-vector retriever"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embeddings)
    store = InMemoryStore()
    
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id",
    )

async def process_text_async(text: Any, text_chain: Any) -> str:
    """Process a single text chunk asynchronously"""
    try:
        return await text_chain.ainvoke(text)
    except Exception as e:
        st.warning(f"Error processing text: {str(e)}")
        return ""

async def process_image_async(img: str, image_chain: Any) -> str:
    """Process a single image asynchronously"""
    try:
        return await image_chain.ainvoke({"image": f"data:image/jpeg;base64,{img}"})
    except Exception as e:
        st.warning(f"Error processing image: {str(e)}")
        return ""

async def summarize_content(texts: List, tables: List, images: List) -> Dict:
    """Generate summaries for different content types asynchronously"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize models
    status_text.text("Initializing AI models...")
    groq_model = ChatGroq(temperature=0.5, model="llama-3.2-11b-vision-preview", max_tokens=300)
    gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    progress_bar.progress(20)

    summaries = {
        "text_summaries": [],
        "table_summaries": [],
        "image_summaries": []
    }

    # Setup prompts and chains
    text_prompt = ChatPromptTemplate.from_template("""
    Summarize the following text concisely:
    {content}
    """)
    text_chain = {"content": lambda x: x} | text_prompt | groq_model | StrOutputParser()
    
    image_prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {"type": "text", "text": "Describe this image in detail, focusing on technical aspects."},
            {"type": "image_url", "image_url": {"url": "{image}"}}
        ])
    ])
    image_chain = image_prompt | gemini_model | StrOutputParser()

    # Process text content
    if texts:
        status_text.text("Analyzing text content...")
        text_tasks = [process_text_async(text, text_chain) for text in texts]
        summaries["text_summaries"] = await asyncio.gather(*text_tasks)
        for i in range(len(texts)):
            progress_bar.progress(20 + (30 * (i + 1) // len(texts)))
            st.write(f"✓ Processed text section {i+1}/{len(texts)}")

    # Process tables
    if tables:
        status_text.text("Analyzing tables...")
        table_tasks = [process_text_async(table, text_chain) for table in tables]
        summaries["table_summaries"] = await asyncio.gather(*table_tasks)
        for i in range(len(tables)):
            progress_bar.progress(50 + (25 * (i + 1) // len(tables)))
            st.write(f"✓ Processed table {i+1}/{len(tables)}")

    # Process images
    if images:
        status_text.text("Analyzing images...")
        image_tasks = [process_image_async(img, image_chain) for img in images]
        summaries["image_summaries"] = await asyncio.gather(*image_tasks)
        for i in range(len(images)):
            progress_bar.progress(75 + (25 * (i + 1) // len(images)))
            st.write(f"✓ Processed image {i+1}/{len(images)}")

    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    return summaries

def process_pdf_chunk(chunk: Any) -> tuple:
    """Process a single PDF chunk"""
    texts = []
    tables = []
    images = []
    
    if "CompositeElement" in str(type(chunk)):
        texts.append(chunk)
        # Extract images from composite elements
        for el in chunk.metadata.orig_elements:
            if "Image" in str(type(el)):
                images.append(el.metadata.image_base64)
    elif "Table" in str(type(chunk)):
        tables.append(chunk)
    
    return texts, tables, images

def process_pdf(uploaded_file) -> tuple:
    """Process uploaded PDF and extract content in parallel"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create a temporary file
    status_text.text("Creating temporary file...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    # Extract content using unstructured
    status_text.text("Extracting content from PDF...")
    progress_bar.progress(20)
    
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
    )

    # Process chunks in parallel
    status_text.text("Processing content in parallel...")
    progress_bar.progress(40)
    
    all_texts = []
    all_tables = []
    all_images = []
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_pdf_chunk, chunks))
        
        for texts, tables, images in results:
            all_texts.extend(texts)
            all_tables.extend(tables)
            all_images.extend(images)

    # Show content statistics
    st.info(f"""Found:
    - {len(all_texts)} text sections
    - {len(all_tables)} tables
    - {len(all_images)} images
    """)

    progress_bar.progress(100)
    status_text.text("Content extraction complete!")
    os.unlink(file_path)  # Clean up temp file
    return all_texts, all_tables, all_images

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda _: "I apologize, but I'm currently experiencing high traffic. Please try again in a few moments."
)
async def generate_answer(gemini: Any, question: str, context: str) -> str:
    """Generate answer with retry logic"""
    try:
        response = await gemini.ainvoke([
            HumanMessage(content=f"""
            Based on the following context, answer the question: {question}
            
            Context: {context}
            """)
        ])
        return response.content
    except Exception as e:
        st.warning("Rate limit reached. Waiting before retry...")
        raise e

def add_documents_to_retriever(retriever: MultiVectorRetriever, texts: List, tables: List, summaries: Dict):
    """Add processed documents to the retriever"""
    status_text = st.empty()
    status_text.text("Adding documents to retriever...")
    
    # Add text content
    for i, (text, summary) in enumerate(zip(texts, summaries["text_summaries"])):
        doc_id = f"text_{i}"
        doc = Document(
            page_content=str(text),
            metadata={
                "doc_id": doc_id,
                "source": "text",
                "summary": summary
            }
        )
        # Add to vectorstore
        retriever.vectorstore.add_documents([doc])
        # Add to docstore
        retriever.docstore.mset([(doc_id, doc)])

    # Add table content
    for i, (table, summary) in enumerate(zip(tables, summaries["table_summaries"])):
        doc_id = f"table_{i}"
        doc = Document(
            page_content=str(table),
            metadata={
                "doc_id": doc_id,
                "source": "table",
                "summary": summary
            }
        )
        # Add to vectorstore
        retriever.vectorstore.add_documents([doc])
        # Add to docstore
        retriever.docstore.mset([(doc_id, doc)])

    status_text.text("Documents added to retriever!")

def extract_references(texts: List) -> List[str]:
    """Extract and process references from text content"""
    references = []
    
    # Common patterns for references
    reference_indicators = [
        "References:",
        "Bibliography:",
        "Further reading:",
        "See also:",
        "Related articles:",
        "Sources:",
    ]
    
    for text in texts:
        content = str(text)
        # Check if this section contains references
        if any(indicator.lower() in content.lower() for indicator in reference_indicators):
            # Split by newlines to separate individual references
            lines = content.split('\n')
            for line in lines:
                # Skip empty lines and headers
                if line.strip() and not any(indicator in line for indicator in reference_indicators):
                    references.append(line.strip())
    
    return references

async def summarize_references(references: List[str], groq_model: Any) -> List[Dict]:
    """Summarize each reference with its topic and relevance"""
    if not references:
        return []
    
    status_text = st.empty()
    status_text.text("Analyzing references...")
    
    reference_prompt = ChatPromptTemplate.from_template("""
    For this reference: {reference}
    Provide a brief analysis in JSON format with these fields:
    - title: The main title or topic
    - type: The type of reference (article, book, website, etc.)
    - relevance: A brief note about its relevance to the main document
    Keep each field concise (max 2 sentences).
    """)
    
    reference_chain = {"reference": lambda x: x} | reference_prompt | groq_model | StrOutputParser()
    
    summaries = []
    for i, ref in enumerate(references):
        try:
            summary = await reference_chain.ainvoke(ref)
            summaries.append(summary)
            st.write(f"✓ Processed reference {i+1}/{len(references)}")
        except Exception as e:
            st.warning(f"Failed to process reference: {str(e)}")
    
    status_text.text("Reference analysis complete!")
    return summaries

def analyze_document_structure(texts: List, tables: List, images: List) -> Dict:
    """Analyze the document structure and create a hierarchical map"""
    structure = {
        "sections": [],
        "internal_links": [],
        "cross_references": [],
        "hierarchy": {}
    }
    
    current_section = None
    current_level = 0
    
    # Common section indicators
    section_patterns = [
        r"^(?:Chapter|Section)\s+\d+",
        r"^\d+\.\d*\s+",
        r"^[IVXLCDM]+\.",
        r"^[A-Z]\.",
    ]
    
    for i, text in enumerate(texts):
        content = str(text)
        lines = content.split('\n')
        
        # Analyze first line for potential section header
        if lines and any(re.match(pattern, lines[0].strip()) for pattern in section_patterns):
            section = {
                "title": lines[0].strip(),
                "index": i,
                "content_type": "text",
                "subsections": [],
                "links": [],
                "references": []
            }
            
            # Detect section level based on formatting
            if re.match(r"^\d+\.\d+\.\d+", lines[0]):
                level = 3
            elif re.match(r"^\d+\.\d+", lines[0]):
                level = 2
            else:
                level = 1
                
            # Add to hierarchy
            if level == 1:
                structure["hierarchy"][section["title"]] = {"subsections": [], "content": []}
                current_section = section["title"]
            elif current_section:
                structure["hierarchy"][current_section]["subsections"].append(section["title"])
            
            structure["sections"].append(section)
            
        # Look for links and references
        for line in lines:
            # Find internal links (e.g., "see Section 2.1")
            internal_links = re.findall(r"see (?:Section|Chapter|Fig\.|Table)\s+[\d\.]+", line)
            if internal_links:
                if current_section:
                    structure["hierarchy"][current_section]["content"].append({
                        "type": "internal_link",
                        "text": internal_links
                    })
                structure["internal_links"].extend(internal_links)
            
            # Find cross-references
            cross_refs = re.findall(r"\[[\d,\s-]+\]", line)
            if cross_refs:
                structure["cross_references"].extend(cross_refs)
                
    # Map tables and images to sections
    for i, table in enumerate(tables):
        if current_section:
            structure["hierarchy"][current_section]["content"].append({
                "type": "table",
                "index": i
            })
            
    for i, image in enumerate(images):
        if current_section:
            structure["hierarchy"][current_section]["content"].append({
                "type": "image",
                "index": i
            })
    
    return structure

def display_document_structure(structure: Dict):
    """Display the document structure analysis in a readable format"""
    st.header("📑 Document Structure Analysis")
    
    # Basic statistics
    st.subheader("Document Statistics")
    st.write(f"- Number of main sections: {len(structure['sections'])}")
    st.write(f"- Internal links found: {len(structure['internal_links'])}")
    st.write(f"- Cross-references: {len(structure['cross_references'])}")
    
    # Section Hierarchy
    st.subheader("Section Hierarchy")
    for section, details in structure["hierarchy"].items():
        st.markdown(f"### {section}")
        if details["subsections"]:
            st.markdown("**Subsections:**")
            for subsection in details["subsections"]:
                st.markdown(f"- {subsection}")
        
        if details["content"]:
            st.markdown("**Content Elements:**")
            for item in details["content"]:
                if item["type"] == "internal_link":
                    st.markdown(f"🔗 Links to: {', '.join(item['text'])}")
                elif item["type"] == "table":
                    st.markdown(f"📊 Table {item['index'] + 1}")
                elif item["type"] == "image":
                    st.markdown(f"🖼️ Figure {item['index'] + 1}")
    
    # Internal Navigation
    if structure["internal_links"]:
        st.subheader("Internal Navigation")
        st.markdown("**Document Cross-References:**")
        for link in structure["internal_links"]:
            st.markdown(f"- {link}")

async def main():
    st.title("Multi-modal PDF Analyzer")
    st.write("Upload a PDF file for comprehensive analysis")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.info(f"Processing: {uploaded_file.name}")
        
        # Setup phase
        with st.spinner("Setting up environment..."):
            setup_api_keys()
            retriever = setup_retriever()

        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["PDF Processing", "Document Structure", "Content Analysis", "Q&A"])
        
        with tab1:
            # Process PDF
            texts, tables, images = process_pdf(uploaded_file)
        
        with tab2:
            # Document Structure Analysis
            structure = analyze_document_structure(texts, tables, images)
            display_document_structure(structure)
            
            # Add structure info to retriever
            doc = Document(
                page_content=str(structure),
                metadata={
                    "doc_id": "structure",
                    "source": "structure_analysis",
                    "type": "document_structure"
                }
            )
            retriever.vectorstore.add_documents([doc])
            retriever.docstore.mset([("structure", doc)])
        
        with tab3:
            # Content Analysis
            summaries = await summarize_content(texts, tables, images)
            
            # Add documents to retriever
            add_documents_to_retriever(retriever, texts, tables, summaries)

            # Display results
            if summaries["text_summaries"]:
                st.subheader("📄 Text Content")
                for i, summary in enumerate(summaries["text_summaries"]):
                    with st.expander(f"Text Section {i+1}"):
                        st.write(summary)

            if summaries["table_summaries"]:
                st.subheader("📊 Table Content")
                for i, summary in enumerate(summaries["table_summaries"]):
                    with st.expander(f"Table {i+1}"):
                        st.write(summary)

            if summaries["image_summaries"]:
                st.subheader("🖼️ Image Content")
                for i, (image, summary) in enumerate(zip(images, summaries["image_summaries"])):
                    with st.expander(f"Image {i+1}"):
                        st.image(f"data:image/jpeg;base64,{image}")
                        st.write(summary)

            # References section
            st.subheader("📚 References")
            references = extract_references(texts)
            if references:
                reference_summaries = await summarize_references(references, groq_model)
                for i, (ref, summary) in enumerate(zip(references, reference_summaries)):
                    with st.expander(f"Reference {i+1}"):
                        st.text("Original reference:")
                        st.write(ref)
                        st.text("Analysis:")
                        try:
                            summary_dict = json.loads(summary)
                            st.write("Title:", summary_dict.get("title"))
                            st.write("Type:", summary_dict.get("type"))
                            st.write("Relevance:", summary_dict.get("relevance"))
                        except json.JSONDecodeError:
                            st.write(summary)
            else:
                st.info("No references found in the document.")
        
        with tab4:
            # Q&A Section
            st.header("🤔 Ask Questions")
            question = st.text_input("Ask a question about the document:")
            if question:
                with st.spinner("Generating answer..."):
                    try:
                        gemini = ChatGoogleGenerativeAI(
                            model="gemini-1.5-flash",
                            temperature=0.3,
                            max_retries=3,
                            timeout=30
                        )
                        docs = retriever.get_relevant_documents(question)
                        
                        context = ' '.join(str(doc.page_content) for doc in docs)
                        response = await generate_answer(gemini, question, context)
                        
                        st.write("Answer:", response)
                    except Exception as e:
                        st.error("""
                        Rate limit reached. Please wait a moment before asking another question.
                        
                        Tips:
                        - Wait 30-60 seconds before trying again
                        - Try rephrasing your question
                        - Break complex questions into simpler ones
                        """)

if __name__ == "__main__":
    asyncio.run(main())