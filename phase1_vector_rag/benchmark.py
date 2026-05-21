# This will be the RAGAS code
# TODO:
# 1. Check if everything is running fine
### 1.1 Ingestion (PDF parsing and chunks) - DONE (Just change some configurations like: The name of the collection, the chunk size and overlap, and the directory to be ingested (I want to ingest a full directory))
### 1.2 Retriever (querying Qdrant and returning the retrieved nodes) - DONE (Just change the query and the parameters like: top_k and node_threshold) (I will check if i can do better)
### 1.3 Generator (building the prompt and generating the answer) - DONE (Just change the prompt template and the model used for generation) (I need to check how to use MARITACA)
# 2. Run a test!
### 2.1 Insert some documents in the folder
### 2.2 Make some retrievals and check the retrieved chunks
### 2.3 Run the full RAG pipeline and check the final answer (Lets try to make some small terminal interface so we can keep asking)
# 3. Benchmark
### 3.1 RAGAS Benchmark, lets just create it as a simple script where we can change the query and the parameters and check the retrieved chunks and the final answer.
### 3.2 We can after that create the logic to run the full questions on the dataset and save the answers
### 3.3 We can after that run the benchmark again and see if there is any improvement to do!