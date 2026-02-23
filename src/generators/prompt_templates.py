from textwrap import dedent

class PromptTemplates:

    @staticmethod
    def get_system_prompt():
        return dedent("""
        You are an expert synthetic data generator for financial relation extraction for the purpose of knowledge base generation. Your task is to generate financial news sentences with corresponding triplets. 
        
        Important instructions:
        1) You will be given a list of allowed relations. You must use a relation from the list, do NOT generate your own.
        2) Do NOT include any explanation or additional text, only the generated sentence and triplets.
        3) Do NOT use literal newlines or line breaks inside string values. Keep all text on a single line.
        4) You MUST return a valid JSON array of the following format:
        [
          {
            "text": "The generated financial news sentence.",
            "triplets": [
              {
                "subject": "Subject Entity",
                "relation": "ExactRelationName",
                "object": "Object Entity"
              }
            ]
          }
        ]
        """).strip()


    @staticmethod
    def generate_n_sentences(n):
        return f"Please generate {n} sentences with different relations according to the instructions above."


    @staticmethod
    def zero_shot(relation_types):
        # Join the list of relations into a string like
        rels_formatted = ", ".join(relation_types)

        return f"Here are the allowed relations: [{rels_formatted}]"

    @staticmethod
    def few_shot(relation_types, examples):
        rels_formatted = ", ".join(relation_types)

        examples_formatted = ""
        for i, example in enumerate(examples):
            examples_formatted += f"\nExample {i}:\nText: {example.sentence}\nTriples: {example.triplets}\n"

        return f"Here are the allowed relations: [{rels_formatted}]\n\nHere are a few examples of sentences with corresponding triplets:{examples_formatted}"