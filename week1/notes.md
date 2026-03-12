Day 1 — LangChain Basics

Document Loader — reads a file (PDF, txt etc) and converts it into LangChain Document objects with page_content and metadata


Text Splitter — breaks large documents into smaller chunks because LLMs have token limits and can't process entire documents at once


RecursiveCharacterTextSplitter — smarter splitter, tries to split on paragraphs first, then sentences, then words. Preserves meaning better than basic splitter.


Day 2 — Chunking

chunk_size=500 — each chunk is max 500 characters


chunk_overlap=50 — last 50 characters of chunk 1 repeat at start of chunk 2. Prevents losing context at boundaries.


Why overlap matters — if an answer sits between two chunks, overlap ensures it's not cut off and lost during retrieval

Day 3 Chromadb:

Embeddings — converts text into a list of numbers (vector) that captures meaning. Similar meaning = similar numbers.

ChromaDB — stores those vectors locally so you can search them later

similarity_search — converts your query into a vector, finds closest chunks in the database