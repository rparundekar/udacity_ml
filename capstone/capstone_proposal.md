# Machine Learning Engineer Nanodegree
## Capstone Proposal: Understanding 'Things' using Semantic Graph Classification
Rahul Parundekar  
January 27th, 2017

## Proposal

### Domain Background
The world around us contains different types of things (e.g. people, places, objects, ideas, etc.). Predominantly, these things are defined by their attributes like shape, color, etc. These things are also defined by the "roles" that they play in their relationships with other things. For example, Washington D.C. is a place and U.S.A is a country. But they have a relationship of Washington D.C. being the capital of USA, which adds extra meaning to Washington D.C. This same role is played by Paris for France. 

The field of [Knowledge Representation and Reasoning](https://en.wikipedia.org/wiki/Knowledge_representation_and_reasoning) within Artificial Intelligence deals with representing these things, types, attributes and relationships using symbols and enabling the agent to reason about them. Another convergence that has come about in this field of knowledge representation and data science is Graph databases since both fields can use graphs with nodes and edges to represent data.

Many domains use grahps to represent their information because nodes, properties and edges of graphs are very well suited to describe the attributes and relationships of things in the domain. 

For example:
1. Spoken systems, the output of Natural Language Processing is a [parse tree](https://en.wikipedia.org/wiki/Parse_tree).
2. Social networks are graphs.
3. High level semantic information in [images](https://arxiv.org/pdf/1405.0312.pdf) are graphs of arrangements of things. 
4. The arrangement of objects on the road for autonomous driving is a graph.
5. A user's browsing pattern of products, usage graphs, etc. is a graph (e.g. browing products, plan of actions, etc.).

In the classic sense, [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning) focuses on specific kinds of understanding - classification, clustering, regression, etc. The algorightims in these deal with feature vectors (e.g. the features used for classification, etc.) and are aimed at essentially discriminating between different types of input to produce some output. To make decisions based on the state of the world an A.I. Agent can read from the world using sensors etc., can easily perform a classification task once it learns the relation between the data to its output decision. For the most part, the feature vectors used in such cases as input encode the attributes of the things, BUT not necessarily the relationships between things. And while the designer of the inputs and outputs of the algorithms may manually craft features to represent some of these relationships, the Agent has no automatic way of comprehending and using these relationships.

Can we use machine learning to make Agents better understand Things, including their attributes AND their relationships?. If we are able to inspect the attributes and relationships of the things together and infer their roles, find its types, etc. our agent can act on those. If an Agent is able to classify things by understanding its semantic relationships, we could in the future generalize it to an Agent that can act on the meaning of the things. 

Existing research in this domain:
..- [Ontology alignment](https://en.wikipedia.org/wiki/Ontology_alignment) is a field of study that researches on understanding the classes of things by aligning types from one data source to the types defined in another source to increase interoperability. In my previous work on aligning ontologies, we employed a brute force method to discover new classes to describe things using 'Restriction Classes' defined by restricting the properties to their values and creating a set of instances to match that restriction. This and other [instance based methods](https://hal.archives-ouvertes.fr/file/index/docid/917910/filename/shvaiko2013a.pdf) can be used to understand things by either creating new class definitions or aligning the definition of things to other classes.
..- [Graph based classification](https://pdfs.semanticscholar.org/b430/6178fb343b4c6e66e64d101606b04f4b5a22.pdf) methods have been used in toxicology detection. 
..- [Text categorization using graph classification](http://frncsrss.github.io/papers/rousseau-acl2015.pdf) is also investigated.

We pick one of these problems as a candidate for exploring the use of Machine Learning to understand semantic data. 

### Problem Statement
Given the semantic data about things (i.e. their attributes and relationships), can we identify their types (e.g. hierarchy of classes) or categories (e.g. roles it plays) interest?  

For example, if you look at [examples of categories in DBpedia](http://downloads.dbpedia.org/preview.php?file=2016-04_sl_core-i18n_sl_en_sl_article_categories_en.tql.bz2), Achilies has been put into the categories - demigods, people of trojan war, characters in Illead, etc. What makes him part of those categories? Can we learn the definitions of these based on the attributes and relationships of Achilies?

### Datasets and Inputs
An [Ontology](https://en.wikipedia.org/wiki/Ontology_(information_science)) is one formalism of to represent semantic data. It specifies the facts about the things' attributes and their relationships that the Agent believes to be true. [DBpedia](https://en.wikipedia.org/wiki/DBpedia) is one such Ontology whose lofty goal is to create a knowledge repository of general knowledge about the world such that Agents can have a grounding of the popular concepts and entities and their relationships. DBpedia [datasets](http://wiki.dbpedia.org/Datasets) contains structured information extracted out of Wikipedia (e.g. page links, categories, infoboxes, etc.) (See [DBpedia – A Large-scale, Multilingual Knowledge Base Extracted from Wikipedia](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.685.7258&rep=rep1&type=pdf) for details of extraction). 

The semantic data in DBpedia can be represented as a graph of nodes and edges. In this case, the things are the nodes and the links/relationships between the things are its edges. For determining the types and categories of a Thing, we start with the node for that thing in the center and look at its neighborhood (e.g. things 1 or 2 hop away) subgraph. We model this as a classification problem.

The input is a list of [different subgraphs centered around things] and the class values are a vector of [types of those things]. To predict the categories of the things, the input is a list of [different subgraphs centered around things] and the class values are a vector of [categories of those things]. DBPedia contains this ground truth information. We can split the data into training, validation & testing set using cross validation, etc. strategies. This data is our ground truth with high fidelity as it has been mannually created by human wikipedia contributors.

### Solution Statement
As described in the [Graph classification](https://pdfs.semanticscholar.org/b430/6178fb343b4c6e66e64d101606b04f4b5a22.pdf) paper, we will use graph kernels. For example, we can use a random walk technique on our input of subgraphs to extract features that can be readily used with our machine learning algorithms. We can then use classification algorithms to identify the types or categories of the instances in our dataset.

By usign this approach we convert graph classification into the classic classification problem and can use differnt classification & algorithms like SVMs, Logistic Regression, Deep learning, etc. to correctly classify the instances

### Benchmark Model
Since there is not much research in this field, we will set our benchmark as the classes that can be identified using the brute force approach of ontology alignment in creating restriction classes (i.e. sets of things formed by restricting properties to values) and matching them to the Types & categories already annotated in dbpedia. 

### Evaluation Metrics
We will compare both the accuracy of finding class definitions as well as as the quality of the definitions by investigating the sets of instances in defined in the class. 

For the first part, the ontology alignment approach will align the classes (types and roles) with the list of instances. Our approach will also classify the instances. We can then compute the precision, recall and f1 score of these approaches by looking at the predicted values and the ground truth. 

For the second part, we qualitatively compare the set of instances in the classes in the ground truth with the false negative and false positive instances that we predict. This gives us insight into how different algorithms work.

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.


**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
