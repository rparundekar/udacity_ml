# Machine Learning Engineer Nanodegree
## Capstone Proposal: Understanding 'Things' using Semantic Graph Classification
Rahul Parundekar  
January 27th, 2017

## Proposal

### Domain Background
The world around us contains different types of things (e.g. people, places, objects, ideas, etc.). Predominantly, these things are defined by their attributes like shape, color, etc. These things are also defined by the "roles" that they play in their relationships with other things. For example, Washington D.C. is a place and U.S.A is a country. But they have a relationship of Washington D.C. being the capital of USA, which adds extra meaning to Washington D.C. This same role is played by Paris for France. 

For an Artificiallly Intelligent (AI) Agent to understand the world around us, it needs to be able to understand & reason about these things, their attributes AND ALSO their relationships. Knowledge Representation and Reasoning [1] is a branch of A.I. that aims to make computers understand and reason about the world. To achieve this, typically a symbolic language (e.g. Prolog, etc.) is used, which contains formalisms to describe the facts about the world that the Agent has sensed (e.g. semantic nets, frames, etc.) / believes to be true. The symbolic language also contains logic fomulas (e.g. rules, inference engines, etc.) that the Agent can use to reason about the world to discover entailments of the facts and identify its goals. Using these set of facts the Agent can then choose to act (e.g. plan & perform sequence of actions, communicate with user, etc.).  

In the classic sense, Machine Learning focuses on specific kinds of understanding - classification, clustering, regression, etc. [2] The algorightims in these deal with feature vectors (e.g. the features used for classification, etc.) and are aimed at essentially discriminating between different types of input to produce some output. 

To make decisions based on the state of the world an A.I. Agent can read from the world using sensors etc., can easily perform a classification task once it learns the relation between the data to its output decision. For the most part, the feature vectors used in such cases as input encode the attributes of the things, BUT not necessarily the relationships between things. 

And while the designer of the inputs and outputs of the algorithms may encode some of these relationships, the Agent has no automatic way of comprehending and using these relationships.

So the domain of interest is this: Can we use machine learning to make Agents understand Things, including their attributes and their relationships, so that it can make a decision and perform its acts. If we are able to inspect the attributes and relationships of the things and infer their roles, classes that they belong to, etc. our agent can act on those.


### Problem Statement
Given the semantic data about things (i.e. their attributes and relationships), can we classify the thing into a class heirarchy of interest so that we can take actions based on those classes?

### Datasets and Inputs
An Ontology [2] is one formalism of to represent semantic data. It specifies the facts about the things' attributes and their relationships that the Agent believes to be true. DBpedia is one such Ontology whose lofty goal is to create a knowledge repository of general knowledge about the world such that Agents can have a grounding of the popular concepts and entities and their relationships [3]. DBpedia contains structured information extracted out of Wikipedia (e.g. page links, categories, infoboxes, etc.).

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

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
[2] Ontology - https://en.wikipedia.org/wiki/Ontology_(information_science)
[3] DBpedia - https://en.wikipedia.org/wiki/DBpedia

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
