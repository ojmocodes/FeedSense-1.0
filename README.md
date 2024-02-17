# FeedSense-1.0
### 
plan

Instead of using a "multi-story" query process, use nlp to convert query into a "desired date" given the user message and the current date, this will have to formatted in a consistent syntax and accurately. Then from the output of this NLP, a vectorstore can be selected, where the titles of the vector stores are the date. it can use simple python logic for processing the "12/03/2023" string and classifying it in one of the vector stores. This allows for one-hop querying for any desired period of time.

also have talked to someone and they recommend vector search with neo4j knowledge graph
https://www.youtube.com/watch?v=bRD09ndyJNs
Look into Neo4j use-cases and in-house recommender systems

###

###

Blake notes

- Ontology/schema with data is IP
- Extract, Transform, Load
- Develop metagraph, a consistent ontology, define initially alongside data, can use LinkML w Python, 
- Extract (source->correct schema->NLP entity recognition/relation extraction in Spacy -> store in easily accesible space, s3, minio (OSstorage) as TileDB array w metadata/schema)
- Nodes more simple, relationships need domain expertise

Questions:
- Become clear on pipeline
- How once have ontology can you add on new pieces of data, how to iteratively add more nodes/relationships
- What problems faced going from research paper to entities/relationships as defined by schema
- Problems/best practices of the english -> cypher -> cypher -> english pipeline, alternatives and possibilties combining vector search and direct cypher
- Geometric deep learning (have tutorials in neo4j) (standard deep learning connnecting inputs to outputs), network analysis, simpler methods also very useful

- What are limits of node properties
- what specific model do english->cypher->cypher->english

said english->cypher new, what is reliable way to be able to ask "how many cows had more than x average milk solids from my whole herd"

in E->C system prompt are you system prompting all the entities and relationships

standard questions are easy to implement, can cover 95% of questions, can have drop downs with pre-defined cypher
non-standard questions do with english to cypher
return sources as metadata with the chunks

###