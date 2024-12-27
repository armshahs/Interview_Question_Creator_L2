prompt_template = """
You are an expert at creating questions based on pdf materials and documents related to sustainability.
Your goal is to prepare a student for their exam or tests on the topic of sustainable development.
You will do this by asking questions about the text below:

------------
{text}
------------

Create questions that will prepare the student for their tests.
Make sure not to lose any important information. 
Also do not produce your own information. Only follow from the information provided in the text above.

QUESTIONS:
"""


refine_template = """
You are an expert at creating practice questions based on  sustainability material and documentation.
Your goal is to help a student prepare for a test on the sustainability topic.
We have received some practice questions to a certain extent: {existing_answer}. 
We have the option to refine the existing questions or add new ones. 
(only if necessary) with some more context below.add()

----------------
{text}
----------------

Given the new context, refine the original questions in English. 
If the context is not helpful, please provide the original questions. 

QUESTIONS:
"""
