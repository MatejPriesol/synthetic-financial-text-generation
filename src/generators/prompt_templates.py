from textwrap import dedent

class PromptTemplates:

    """
    SYSTEM PROMPTS
    """
    @staticmethod
    def get_system_prompt_with_template(allowed_schemas):
        schemas_formatted = ", ".join(allowed_schemas)

        return dedent(f"""
            You are an expert synthetic data generator for financial relation extraction for the purpose of knowledge base generation. Your task is to generate financial news sentences with corresponding triplets. 
    
            Important instructions:
            1) Do NOT include any explanation or additional text, only the generated sentence and triplets.
            2) Do NOT use literal newlines or line breaks inside string values. Keep all text on a single line.
            3) You MUST return a valid JSON array of the following format:
            [
              {{
                "text": "The generated financial news sentence.",
                "triplets": [
                  {{
                    "subject": "Subject Entity",
                    "relation": "ExactRelationName",
                    "object": "Object Entity",
                    "subject_type": "Type of the subject",
                    "object_type": "Type of the object"
                  }}
                ]
              }}
            ]
            4) The sentence can have multiple triplets and you are allowed to generate compound sentences. 
            5) For the triplets:
               - "subject" and "object" MUST be the exact words or phrases used in the sentence.
               - "subject_type" and "object_type" MUST exactly match the bracketed types from the allowed structures.
               - "relation" MUST exactly match the relation in parentheses from the allowed structures.
            6) You are ONLY allowed to use the following [SubjectType] --(Relation)--> [ObjectType] structures. Do not invent new relations or mix entity types that are not explicitly defined below.
            7) CRITICAL GRAMMAR RULE: The 'relation' in your JSON must exactly match the allowed list, but in the generated 'text' sentence, you MUST express this relation using natural, fluent English.\n
            ALLOWED STRUCTURES:
            {schemas_formatted}
        """).strip()

    @staticmethod
    def get_system_prompt_with_fixed_templates():
        return dedent("""
            You are an expert synthetic data generator for financial relation extraction for the purpose of knowledge base generation. Your task is to generate financial news sentences with corresponding triplets. 

            Important instructions:
            1) Do NOT include any explanation or additional text, only the generated sentence and triplets.
            2) Do NOT use literal newlines or line breaks inside string values. Keep all text on a single line.
            3) You MUST return a valid JSON array of the following format:
            [
              {
                "text": "The generated financial news sentence.",
                "triplets": [
                  {
                    "subject": "Subject Entity",
                    "relation": "ExactRelationName",
                    "object": "Object Entity",
                    "subject_type": "Type of the subject",
                    "object_type": "Type of the object"
                  }
                ]
              }
            ]
            4) The sentence can have multiple triplets and you are allowed to generate compound sentences. 
            5) For the triplets:
               - "subject" and "object" MUST be the exact words or phrases used in the sentence.
               - "subject_type", "object_type", and "relation" MUST exactly match the bracketed types and parentheses from the specific schema requested by the user.
            6) You will be given [SubjectType] --(Relation)--> [ObjectType] structures. Do not invent new relations or mix entity types that are not explicitly defined in the user's request.
            7) CRITICAL GRAMMAR RULE: The 'relation' in your JSON must exactly match the schema list, but in the generated 'text' sentence, you MUST express this relation using natural, fluent English.
        """).strip()

    @staticmethod
    def get_system_prompt_universal(allowed_schemas):
        schemas_formatted = "\n".join([f"- {schema}" for schema in allowed_schemas])

        prompt_template = dedent("""
            You are an expert synthetic data generator for financial Knowledge Graphs.
            Your task is to generate highly realistic, complex financial news sentences paired with their exact relation triplets.
        
            CRITICAL RULES:
            1. TRIPLET DENSITY: Vary the complexity of your generated sentences to mimic real-world text. Generate a diverse mix: some should be simple single-triplet sentences, most should contain 2 to 3 triplets, and occasionally you should write highly complex sentences containing up to 7 triplets.
            2. FLUENT ENGLISH ONLY: NEVER insert the raw relation tokens (e.g., 'Actionto', 'actionin') into the English sentence. You MUST express the relationship using natural, fluent English phrasing.
            3. EXACT MATCHING: The 'subject' and 'object' values in your JSON MUST be exact substrings extracted from your generated sentence.
            4. STRICT ONTOLOGY: You may ONLY use the exact [SubjectType] --(Relation)--> [ObjectType] combinations listed below. Do not invent relations or mix types.
        
            JSON FORMAT:
            Output ONLY a valid JSON array. Do not include markdown blocks, explanations, or conversational text.
            [
              {{
                "text": "The natural English sentence.",
                "triplets": [
                  {{
                    "subject": "The subject entity from sentence",
                    "relation": "ExactRelationName",
                    "object": "The object entity from sentence",
                    "subject_type": "SubjectType",
                    "object_type": "ObjectType"
                  }}
                ]
              }}
            ]
        
            ALLOWED ONTOLOGY SCHEMAS:
            {schemas_formatted}
        """).strip()

        return prompt_template.format(schemas_formatted=schemas_formatted)


    """
    USER PROMPTS
    """
    @staticmethod
    def get_user_prompt_template_based(schemas):
        n = len(schemas)

        # Format the forced schemas into a clean, numbered list
        schema_list = "\n".join([f"Sentence {i + 1}: {schema}" for i, schema in enumerate(schemas)])

        return (
            f"Please generate {n} unique financial sentences.\n\n"
            f"You MUST follow this exact 1-to-1 mapping for the primary triplet in each sentence:\n"
            f"{schema_list}\n\n"
            f"CRITICAL RULES:\n"
            f"- Sentence 1 MUST contain a triplet matching the schema for Sentence 1.\n"
            f"- Sentence 2 MUST contain a triplet matching the schema for Sentence 2, and so on.\n"
            f"- You may include additional valid financial triplets in the sentences if it sounds natural, but the requested schema MUST be the core of the sentence.\n"
            f"- Ensure high diversity in sentence structures, length, vocabulary, and financial contexts to avoid repetition."
        )

    @staticmethod
    def get_user_prompt_entity(schemas):
        n = len(schemas)
        schema_list = "\n".join([f"{i + 1}. Template: {schema}" for i, schema in enumerate(schemas)])

        prompt_template = dedent("""
                TASK: Generate a batch of {n} unique, highly realistic financial sentences.

                REQUIRED BATCH MAPPING:
                You MUST follow this exact 1-to-1 mapping. Sentence 1 must be built around Template 1, Sentence 2 around Template 2, and so on. 

                {schema_list}

                EXECUTION RULES:
                1. LITERAL FACT INTEGRATION: The requested Template dictates the core factual foundation of your sentence. 
                   - The text inside the brackets `[...]` are the EXACT literal entity names. You MUST include these exact strings in your English sentence.
                   - The `(Types: ...)` at the end of the line is purely context to help you understand what those entities are. Do NOT write these type names in the English sentence.
                2. COMPLEXITY: Naturally weave in additional valid facts using the Allowed Ontology from your system rules to meet the density requirements.
                3. EXTREME DIVERSITY: You are generating batch data. You MUST use variety in your writing. 
               - Do NOT start sentences with the same grammatical structures (e.g., avoid starting multiple sentences with "The", "In [Year]", or "Company X").
               - Vary the financial contexts (e.g., Annual reports, quarterly earnings, regulatory fines...).
               - Vary the sentence lengths (mix shorter and longer sentences with multiple valid triplets).
            """).strip()

        return prompt_template.format(n=n, schema_list=schema_list)

    @staticmethod
    def get_user_prompt_template(schemas):
        n = len(schemas)
        schema_list = "\n".join([f"{i + 1}. Template: {schema}" for i, schema in enumerate(schemas)])

        prompt_template = dedent("""
            TASK: Generate a batch of {n} unique, highly realistic financial sentences.
    
            REQUIRED BATCH MAPPING:
            You MUST follow this exact 1-to-1 mapping. Sentence 1 must be built around Template 1, Sentence 2 around Template 2, and so on. 
    
            {schema_list}
    
            EXECUTION RULES:
            1. TEMPLATE INTEGRATION: The requested Template must act as the core factual foundation of the sentence. 
            2. COMPLEXITY: As instructed in the system rules, naturally weave in additional valid triplets from the Allowed Ontology to meet the density requirements.
            3. EXTREME DIVERSITY: You are generating batch data. You MUST use variety in your writing. 
               - Do NOT start sentences with the same grammatical structures (e.g., avoid starting multiple sentences with "The", "In [Year]", or "Company X").
               - Vary the financial contexts (e.g., Annual reports, quarterly earnings, regulatory fines...).
               - Vary the sentence lengths (mix shorter and longer sentences with multiple valid triplets).
        """).strip()

        return prompt_template.format(n=n, schema_list=schema_list)

    @staticmethod
    def get_user_prompt_few_shot(n, examples):
        examples_formatted = "\n".join([f"Example {i + 1}: \"{ex}\"" for i, ex in enumerate(examples)])

        prompt_template = dedent("""
            TASK: Generate a batch of {n} unique, highly realistic financial sentences.

            REFERENCE EXAMPLES (STYLE GUIDE):
            Below are real examples from a financial dataset. Analyze their style, tone, and complexity. 

            {examples_formatted}

            EXECUTION RULES:
            1. ORIGINALITY: Do NOT copy the exact facts or entities from the examples. Generate entirely new facts using the Allowed Ontology Schemas from your system rules.
            2. COMPLEXITY: Mimic the linguistic complexity of the examples, weaving in multiple valid triplets per sentence.
            3. EXTREME DIVERSITY: You are generating batch data. You MUST use variety in your writing. 
               - Do NOT start sentences with the same grammatical structures (e.g., avoid starting multiple sentences with "The", "In [Year]", or "Company X").
               - Vary the financial contexts (e.g., Annual reports, quarterly earnings, regulatory fines...).
               - Vary the sentence lengths (mix shorter and longer sentences with multiple valid triplets).
        """).strip()

        return prompt_template.format(n=n, examples_formatted=examples_formatted)

    @staticmethod
    def get_user_prompt_zero_shot(n):
        prompt_template = dedent("""
            TASK: Generate a batch of {n} unique, highly realistic financial sentences.

            EXECUTION RULES:
            1. FREE GENERATION: You may freely choose any valid combinations from the Allowed Ontology Schemas in your system rules to build your sentences.
            2. COMPLEXITY: Naturally weave in multiple valid triplets to meet the density requirements.
            3. EXTREME DIVERSITY: You are generating batch data. You MUST use variety in your writing. 
               - Do NOT start sentences with the same grammatical structures (e.g., avoid starting multiple sentences with "The", "In [Year]", or "Company X").
               - Vary the financial contexts (e.g., Annual reports, quarterly earnings, regulatory fines...).
               - Vary the sentence lengths (mix shorter and longer sentences with multiple valid triplets).
        """).strip()

        return prompt_template.format(n=n)


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