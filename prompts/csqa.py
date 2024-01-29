prompt = """Given the question, output the rationale step by step and give the final answer. You should choose the best answer.

Example 1
Question:
Sammy wanted to go to where the people were. Where might he go?
A. race track
B. populated area
C. the desert
D. apartment
E. roadblock
Answer:
Sammy wanted to go to places with many people.
Race track and apartment do not have many people.
The desert and roadblock have few people.
And, the populated area means that it is the place with many people.
Thus, Sammy should go to populated area.
Final Answer: B

Example 2
Question:
The fox walked from the city into the forest, what was it looking for?
A. pretty flowers
B. hen house
C. natural habitat
D. storybook
E. dense forest
Answer:
The forest does not have hen house or storybook.
The fox is a carnivore that does not look for flowers and forest.
The forest is a natural habitat for foxes.
Thus, it was looking for a natural habitat.
Final Answer: C
"""

recall_prompt = """Given the question, output the rationale step by step and give the final answer. You should choose the best answer.

Example 1
Question:
Sammy wanted to go to where the people were. Where might he go?
A. race track
B. populated area
C. the desert
D. apartment
E. roadblock
Answer:
Fact:
Sammy wanted to go to places with many people.
Race track and apartment do not have many people.
The desert and roadblock have few people.
And, the populated area means that it is the place with many people.
Reasoning:
Thus, Sammy should go to populated area.
Final Answer: B

Example 2
Question:
The fox walked from the city into the forest, what was it looking for?
A. pretty flowers
B. hen house
C. natural habitat
D. storybook
E. dense forest
Answer:
Fact:
The forest does not have hen house or storybook.
The fox is a carnivore that does not look for flowers and forest.
The forest is a natural habitat for foxes.
Reasoning:
Thus, it was looking for a natural habitat.
Final Answer: C
"""