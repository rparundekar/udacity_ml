# Machine Learning Engineer Nanodegree
## Capstone Proposal: Understanding 'Things' using Semantic Graph Classification
Rahul Parundekar  
January 27th, 2017

## Proposal

### Domain Background
The world around us contains different types of things (e.g. people, places, objects, ideas, etc.). Predominantly, these things are defined by their attributes like shape, color, etc. These things are also defined by the "roles" that they play in their relationships with other things. For example, Washington D.C. is a place and U.S.A is a country. But they have a relationship of Washington D.C. being the capital of USA, which adds extra meaning to Washington D.C. This same role is played by Paris for France. 

I want to investigate machine learning on semantic data like attributes and relationships of things. Many domains use grahps to represent their information because nodes, properties and edges of graphs are very well suited to describe the attributes and relationships of things in the domain. For example:
a) Spoken systems, the output of Natural Language Processing is a parse tree [https://en.wikipedia.org/wiki/Parse_tree] 
b) Social networks are graphs.
c) High level semantic information in images are graphs of arrangements of things [https://arxiv.org/pdf/1405.0312.pdf] 
d) The arrangement of objects on the road for autonomous driving is a graph.
e) The browsing data of prodiucts, usage graphs, etc. is a graph (e.g. browing products, plan of actions, etc.)

In the classic sense, Machine Learning focuses on specific kinds of understanding - classification, clustering, regression, etc. [2] The algorightims in these deal with feature vectors (e.g. the features used for classification, etc.) and are aimed at essentially discriminating between different types of input to produce some output. To make decisions based on the state of the world an A.I. Agent can read from the world using sensors etc., can easily perform a classification task once it learns the relation between the data to its output decision. For the most part, the feature vectors used in such cases as input encode the attributes of the things, BUT not necessarily the relationships between things. And while the designer of the inputs and outputs of the algorithms may manually craft features to represent some of these relationships, the Agent has no automatic way of comprehending and using these relationships.

Can we use machine learning to make Agents better understand Things, including their attributes AND their relationships?. If we are able to inspect the attributes and relationships of the things together and infer their roles, find its types, etc. our agent can act on those. If an Agent is able to classify things by understanding its semantic relationships, we could in the future generalize it to an Agent that can act on the meaning of the things. 

### Problem Statement
Given the semantic data about things (i.e. their attributes and relationships), can we identify their types (e.g. hierarchy of classes) or categories (e.g. roles it plays) interest?  

For example, if you look at [6], Achilies has been put into the categories - demigods, people of trojan war, characters in Illead, etc. What makes him part of those categories? Can we learn the definitions of these based on the attributes and relationships of Achilies?

### Datasets and Inputs
An Ontology [3] is one formalism of to represent semantic data. It specifies the facts about the things' attributes and their relationships that the Agent believes to be true. DBpedia is one such Ontology whose lofty goal is to create a knowledge repository of general knowledge about the world such that Agents can have a grounding of the popular concepts and entities and their relationships [4]. DBpedia contains structured information extracted out of Wikipedia (e.g. page links, categories, infoboxes, etc.). [5][6]

The semantic data in DBpedia can be represented as a graph of nodes and edges. In this case, the things are the nodes and the links/relationships between the things are its edges. For determining the types and categories of a Thing, we start with the node for that thing in the center and look at its neighborhood (e.g. things 1 or 2 hop away) subgraph. We model this as a classification problem.

The input is a list of [different subgraphs centered around things] and the class values are a vector of [types of those things]. To predict the categories of the things, the input is a list of [different subgraphs centered around things] and the class values are a vector of [categories of those things]. DBPedia contains this ground truth information. We can split the data into training, validation & testing set using cross validation, etc. strategies.

### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------
References : 
[1] Knowledge Representation and Reasoning - https://en.wikipedia.org/wiki/Knowledge_representation_and_reasoning
[2] Machine Learning - https://en.wikipedia.org/wiki/Machine_learning
[3] Ontology - https://en.wikipedia.org/wiki/Ontology_(information_science)
[4] DBpedia - https://en.wikipedia.org/wiki/DBpedia
[5] DBpedia – A Large-scale, Multilingual
Knowledge Base Extracted from Wikipedia - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.685.7258&rep=rep1&type=pdf
[6] DBpedia Datasets - http://wiki.dbpedia.org/Datasets
[7] Example categories - http://downloads.dbpedia.org/preview.php?file=2016-04_sl_core-i18n_sl_en_sl_article_categories_en.tql.bz2

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
