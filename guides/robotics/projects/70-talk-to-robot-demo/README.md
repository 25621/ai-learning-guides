# Talk-to-Robot Demo

## Key Insight

Integrating [large language models (LLMs)](/shared/glossary/#llm) with robotic control enables natural language [task planning](/shared/glossary/#saycan) by decomposing high-level user commands into a sequence of executable [primitives](/shared/glossary/#saycan). In frameworks like [SayCan](/shared/glossary/#saycan), the LLM proposes actions ("say") while a learned [policy](/shared/glossary/#policy) evaluates the feasibility of each action in the current environment ("can"). This combined approach ensures the robot plans actions that are both semantically correct and physically executable in simulation or real-world kitchens.
