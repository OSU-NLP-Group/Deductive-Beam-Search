prompt = """Given the question, output the rationale step by step and give the final answer (yes or no).

Example 1
Question:
Do hamsters provide food for any animals?
Answer:
Hamsters are prey animals.
Prey are food for predators.
Final answer: yes

Example 2
Question:
Could a llama birth twice during War in Vietnam (1945-46)?
Answer:
The War in Vietnam was 6 months.
The gestation period for a llama is 11 months, which is more than 6 months.
Final answer: no
"""

recall_prompt = """Given the question, output the rationale step by step and give the final answer (yes or no).

Example 1
Question:
Do hamsters provide food for any animals?
Answer:
Fact:
Hamsters are prey animals.
Prey are food for predators.
Reasoning:
Hamsters are food for some predators.
Final answer: yes

Example 2
Question:
Could a llama birth twice during War in Vietnam (1945-46)?
Answer:
Fact:
The War in Vietnam was 6 months.
The gestation period for a llama is 11 months, which is more than 6 months.
Reasoning:
A llama could not birth twice during War in Vietnam.
Final answer: no
"""