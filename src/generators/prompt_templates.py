from textwrap import dedent

class PromptTemplates:

    """
    SYSTEM PROMPTS
    """
    @staticmethod
    def get_system_prompt_template():
        return dedent("""
            You are an expert synthetic data generator for financial Knowledge Graphs.
            You will be given [SubjectType] --(Relation)--> [ObjectType] templates. Your task is to generate highly realistic, complex financial news sentences paired with their relational triplet templates.
            
            JSON FORMAT:
            Output ONLY a valid JSON array. Do not include markdown blocks, explanations, or conversational text.
            [
              {
                "text": "The generated natural English sentence.",
                "triplets": [
                  {
                    "subject": "The subject entity from the sentence",
                    "relation": "The Exact Relation Name from the template",
                    "object": "The object entity from the sentence",
                    "subject_type": "The exact Subject Type from the template",
                    "object_type": "The exact Object Type from the template"
                  }
                ]
              }
            ]
            
            IMPORTANT RULES:
            1. FLUENT ENGLISH ONLY: NEVER insert the raw relation tokens (e.g., 'Actionto', 'actionin') or the Subject/Object types into the English sentence. You MUST express the relationship using natural, fluent English phrasing.
            2. EXACT MATCHING: The 'subject' and 'object' values in your JSON MUST be exact substrings extracted from your generated sentence. The relation and subject/object types MUST be from the triplet.
            3. STRICT ONTOLOGY: You may ONLY use the exact [SubjectType] --(Relation)--> [ObjectType] combinations you will be given. Do not invent relations or mix types.
        
        """).strip()

    @staticmethod
    def get_system_prompt_entity(allowed_ontology):
        schemas_formatted = "\n".join([f"- {schema}" for schema in allowed_ontology])

        prompt_entity = dedent("""
            You are an expert synthetic data generator for financial Knowledge Graphs.
            You will be given an Allowed Ontology, which is a collection of [SubjectType] --(Relation)--> [ObjectType] triplets, and a batch-specific Restricted Entity Pool. Your task is to generate highly realistic, complex financial news sentences paired with their relational triplets using only the provided entities.

            JSON FORMAT:
            Output ONLY a valid JSON array. Do not include markdown blocks, explanations, or conversational text.
            [
              {{
                "text": "The generated natural English sentence.",
                "triplets": [
                  {{
                    "subject": "The subject entity from the sentence",
                    "relation": "The Exact Relation Name from the triplet",
                    "object": "The object entity from the sentence",
                    "subject_type": "The exact Subject Type from the triplet",
                    "object_type": "The exact Object Type from the triplet"
                  }}
                ]
              }}
            ]

            IMPORTANT RULES:
            1. FLUENT ENGLISH ONLY: NEVER insert the raw relation tokens (e.g., 'Actionto', 'actionin') or the Subject/Object types into the English sentence. You MUST express the relationship using natural, fluent English phrasing.
            2. EXACT MATCHING: The 'subject' and 'object' values in your JSON MUST be exact substrings extracted from your generated sentence. The relation and subject/object types MUST be from the Allowed Ontology.
            3. RESTRICTED VOCABULARY: You MUST ONLY use the specific entities provided in the batch list. Do not invent new company names, people, dates, or financial figures.
            4. STRICT ONTOLOGY: The relationships you establish between the entities MUST strictly comply with the [SubjectType] --(Relation)--> [ObjectType] rules listed in the Allowed Ontology below. Do not invent relations or mix types.

            ALLOWED ONTOLOGY:
            {schemas_formatted}
        """).strip()

        return prompt_entity.format(schemas_formatted=schemas_formatted)

    @staticmethod
    def get_system_prompt_in_context(allowed_schemas):
        schemas_formatted = "\n".join([f"- {schema}" for schema in allowed_schemas])

        prompt_template = dedent("""
            You are an expert synthetic data generator for financial Knowledge Graphs.
            You will be given an Allowed Ontology, which is a collection of [SubjectType] --(Relation)--> [ObjectType] triplets. Your task is to generate highly realistic, complex financial news sentences paired with their relational triplets.

            JSON FORMAT:
            Output ONLY a valid JSON array. Do not include markdown blocks, explanations, or conversational text.
            [
              {{
                "text": "The generated natural English sentence.",
                "triplets": [
                  {{
                    "subject": "The subject entity from the sentence",
                    "relation": "The Exact Relation Name from the triplet",
                    "object": "The object entity from the sentence",
                    "subject_type": "The exact Subject Type from the triplet",
                    "object_type": "The exact Object Type from the triplet"
                  }}
                ]
              }}
            ]

            IMPORTANT RULES:
            1. FLUENT ENGLISH ONLY: NEVER insert the raw relation tokens (e.g., 'Actionto', 'actionin') or the Subject/Object types into the English sentence. You MUST express the relationship using natural, fluent English phrasing.
            2. EXACT MATCHING: The 'subject' and 'object' values in your JSON MUST be exact substrings extracted from your generated sentence. The relation and subject/object types MUST be from the Allowed Ontology.
            3. STRICT ONTOLOGY: You may ONLY use the exact [SubjectType] --(Relation)--> [ObjectType] combinations listed in the Allowed Ontology. Do not invent relations or mix types.
            
            ALLOWED ONTOLOGY:
            {schemas_formatted}

        """).strip()

        return prompt_template.format(schemas_formatted=schemas_formatted)

    """
    USER PROMPTS
    """
    @staticmethod
    def get_user_prompt_template(schemas):
        n = len(schemas)

        # Format the tuples into a comma-separated string for the prompt
        schema_lines = []
        for i, schema_tuple in enumerate(schemas):
            joined_triplets = ", ".join(schema_tuple)
            schema_lines.append(f"{i + 1}. Template Set: {joined_triplets}")

        schema_list = "\n".join(schema_lines)

        prompt_template = dedent("""
                TASK: Generate a batch of {n} unique, highly realistic financial sentences.

                REQUIRED BATCH MAPPING:
                You MUST follow the following mapping. Sentence 1 MUST be built around Template Set 1, Sentence 2 around Template Set 2, and so on. 

                {schema_list}

                EXECUTION RULES:
                1. TEMPLATE INTEGRATION: The requested Template Set contains one or more relation triplets. You MUST use ALL triplets in the assigned set as the core factual foundation of the sentence. 
                   - You do NOT need to present the facts in the exact order they are listed. 
                   - Integrate them so the semantic meaning is naturally woven together into a single, cohesive sentence.
                2. DIVERSITY: You are generating batch data. You MUST use variety in your writing. 
                   - Do NOT start sentences with the same grammatical structures (e.g., avoid starting multiple sentences with "The", "In [Year]", or "Company X").
                   - Vary the financial contexts (e.g., Annual reports, quarterly earnings, regulatory fines...).
                   - Vary the sentence lengths and complexity (mix shorter and longer sentences).
            """).strip()

        return prompt_template.format(n=n, schema_list=schema_list)

    @staticmethod
    def get_user_prompt_entity(pool_string, density_blueprint, n=10):
        prompt_entity_user = dedent("""
            TASK: Generate a batch of {n} unique, highly realistic financial sentences.

            RESTRICTED ENTITY POOL FOR THIS BATCH:
            You may ONLY use the following entities to construct your sentences and JSON triplets:
            {pool_string}

            EXECUTION RULES:
            1. FREE STRUCTURE: You may freely choose how to connect these entities using any valid triplets from the Allowed Ontology.
            2. REUSE AND COMBINE: You can reuse and recombine these specific entities across the {n} sentences.
            3. TRIPLET BLUEPRINT: For each sentence, use the following blueprint for the number of triplets:
            {density_blueprint}
            4. DIVERSITY: You are generating batch data. You MUST use variety in your writing. 
               - Do NOT start sentences with the same grammatical structures (e.g., avoid starting multiple sentences with "The", "In [Year]", or "Company X").
               - Vary the financial contexts (e.g., Annual reports, quarterly earnings, regulatory fines...).
               - Vary the sentence lengths and complexity (mix shorter and longer sentences).
        """).strip()

        return prompt_entity_user.format(n=n, pool_string=pool_string, density_blueprint=density_blueprint)

    @staticmethod
    def get_user_prompt_few_shot(n, examples, density_blueprint):
        examples_formatted = "\n".join([f"Example {i + 1}: {ex}" for i, ex in enumerate(examples)])

        prompt_template = dedent("""
            TASK: Generate a batch of {n} unique, highly realistic financial sentences.

            REFERENCE EXAMPLES (STYLE GUIDE):
            Below are {m} real examples demonstrating the exact linguistic style, complexity, and JSON formatting. 

            {examples_formatted}

            EXECUTION RULES:
            1. ORIGINALITY: Do NOT copy the exact facts or entities from the examples. Generate entirely new facts using the Allowed Ontology from your system rules.
            2. TRIPLET BLUEPRINT: For each sentence, use the following blueprint for the number of triplets:
            {density_blueprint}
            3. DIVERSITY: You are generating batch data. You MUST use variety in your writing. 
               - Do NOT start sentences with the same grammatical structures (e.g., avoid starting multiple sentences with "The", "In [Year]", or "Company X").
               - Vary the financial contexts (e.g., Annual reports, quarterly earnings, regulatory fines...).
               - Vary the sentence lengths and complexity (mix shorter and longer sentences).
        """).strip()

        return prompt_template.format(n=n, m=len(examples), examples_formatted=examples_formatted, density_blueprint=density_blueprint)

    @staticmethod
    def get_user_prompt_zero_shot(n, density_blueprint):
        prompt_template = dedent("""
            TASK: Generate a batch of {n} unique, highly realistic financial sentences.

            EXECUTION RULES:
            1. FREE GENERATION: You may freely choose any valid combinations from the Allowed Ontology in your system rules to build your sentences.
            2. TRIPLET BLUEPRINT: For each sentence, use the following blueprint for the number of triplets:
            {density_blueprint}
            3. DIVERSITY: You are generating batch data. You MUST use variety in your writing. 
               - Do NOT start sentences with the same grammatical structures (e.g., avoid starting multiple sentences with "The", "In [Year]", or "Company X").
               - Vary the financial contexts (e.g., Annual reports, quarterly earnings, regulatory fines...).
               - Vary the sentence lengths and complexity (mix shorter and longer sentences).
        """).strip()

        return prompt_template.format(n=n, density_blueprint=density_blueprint)


    """
    EVALUATION PROMPTS
    """
    @staticmethod
    def adequacy_eval_system_prompt():
        return dedent("""
        You are an expert linguistic evaluator for a financial dataset.
        Your task is to evaluate how well a generated English sentence expresses a specific knowledge graph triplet.
        
        IMPORTANT CONTEXT:
        The triplet represents underlying factual concepts and relationships. The sentence does NOT need to use the exact literal strings or vocabulary of the triplet, as long as the semantic meaning, factual accuracy, and relationship are perfectly conveyed in natural English.

        Evaluation Criteria (Relational Fidelity & Fluency):
        5 - Perfect: Subject, object, and relation are expressed perfectly in natural English.
        4 - Good: Facts are present but phrasing is slightly awkward.
        3 - Forced: Facts are present, but the raw relation token is unnaturally injected into the text.
        2 - Partial/Wrong: Sentence misses the relation, or hallucinates wrong relationships.
        1 - Failure: Sentence completely fails to include the triplet facts.

        Provide ONLY a single integer between 1 and 5. Do not provide any other text.
        """).strip()

    @staticmethod
    def adequacy_eval_user_prompt(triplet, sentence):
        return dedent(f"""
        Triplet to express:
        Subject: {triplet.get('subject')} (Type: [{triplet.get('subject_type')}])
        Relation: {triplet.get('relation')}
        Object: {triplet.get('object')} (Type: [{triplet.get('object_type')}])
        
        Generated Sentence: "{sentence}"
        """).strip()