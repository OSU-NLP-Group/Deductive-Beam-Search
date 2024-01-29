prompt = """You are a good math solver. Based on the question, please give the rationales step by step and give a final answer.

Example 1:
Question:
Kate's hair is half as long as Emily's hair. Emily's hair is 6 inches longer than Logan's hair. If Logan hair is 20 inches, how many inches is Kate's hair?
Answer:
Emily's hair is 20-6 = 14 inches long.
Kate's hair 14/2= 7 inches long.
Final Answer:7

Example 2:
Question:
John puts $25 in his piggy bank every month for 2 years to save up for a vacation. He had to spend $400 from his piggy bank savings last week to repair his car. How many dollars are left in his piggy bank?
Answer:
He saved money for 2 years, which is equal to 12 x 2 = 24 months.
The amount of money he saved is $25*24 = $600.
But he spent some money so there is $600 - $400 = 200 left.
Final Answer:200
"""

"""
Example 3:
Question:
After complaints from the residents of Tatoosh about the number of cats on the island, the wildlife service carried out a relocation mission that saw the number of cats on the island drastically reduced. On the first relocation mission, 600 cats were relocated from the island to a neighboring island. On the second mission, half of the remaining cats were relocated to a rescue center inland. If the number of cats originally on the island was 1800, how many cats remained on the island after the rescue mission?
Answer:
After the first mission, the number of cats remaining on the island was 1800-600 = <<1800-600=1200>>1200.
If half of the remaining cats on the island were relocated to a rescue center inland, the number of cats taken by the wildlife service on the second mission is 1200/2 = <<1200/2=600>>600 cats.
The number of cats remaining on the island is 1200-600 = <<1200-600=600>>600
Final Answer:600

Example 4:
Question:
Paul, Amoura, and Ingrid were to go to a friend's party planned to start at 8:00 a.m. Paul arrived at 8:25. Amoura arrived 30 minutes later than Paul, and Ingrid was three times later than Amoura. How late, in minutes, was Ingrid to the party?
Answer:
If the party was planned to start at 8:00 am, Paul was 8:25-8:00 = <<825-800=25>>25 minutes late.
If Paul was 25 minutes late, Amoura was 25+30 = <<25+30=55>>55 minutes late.
If Ingrid was three times late than Amoura was, she was 3*55 = <<3*55=165>>165 minutes late
Final Answer:165

Example 5:
Question:
Alicia has to buy some books for the new school year. She buys 2 math books, 3 art books, and 6 science books, for a total of $30. If both the math and science books cost $3 each, what was the cost of each art book?
Answer:
The total cost of maths books is 2*3 = <<2*3=6>>6 dollars
The total cost of science books is 6*3 = <<6*3=18>>18 dollars
The total cost for maths and science books is 6+18 = <<6+18=24>>24 dollars
The cost for art books is 30-24 = <<30-24=6>>6 dollars.
Since he bought 3 art books, the cost for each art book will be 6/3 = 2 dollars
Final Answer:2
"""