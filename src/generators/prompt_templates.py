class PromptTemplates:

    @staticmethod
    def get_system_prompt():
        return "You are an expert synthetic data generator for financial relation extraction for the purpose of knowledge base generation. Your task is to generate financial news sentences with corresponding triplets. You will be given a list of allowed relations. Return ONLY the sentence followed by the triplet in JSON format according to the following template:\n Sentence: 'This is the generated sentence'\n Triplet: {{'object': A, 'relation': Relation, 'subject': B}}"

    @staticmethod
    def five_unique_sentences():
        return "Please generate 5 sentences with different relations according to the instructions above."


    @staticmethod
    def zero_shot(relation_types):
        # Join the list of relations into a string like
        rels_formatted = ", ".join(relation_types)

        return f"Here are the allowed relations: [{rels_formatted}]"