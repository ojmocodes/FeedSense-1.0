# FeedSense-1.0
### 
plan

Instead of using a "multi-story" query process, use nlp to convert query into a "desired date" given the user message and the current date, this will have to formatted in a consistent syntax and accurately. Then from the output of this NLP, a vectorstore can be selected, where the titles of the vector stores are the date. it can use simple python logic for processing the "12/03/2023" string and classifying it in one of the vector stores. This allows for one-hop querying for any desired period of time.

also have talked to someone and they recommend vector search with neo4j knowledge graph
https://www.youtube.com/watch?v=bRD09ndyJNs

###