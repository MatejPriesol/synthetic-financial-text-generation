import torch
from transformers import pipeline


def check_single_sentence(sentence):
    # Set up device to use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model on {device}...")
    grammar_checker = pipeline(
        "text-classification",
        model="textattack/roberta-base-CoLA",
        device=device
    )

    # Run inference
    result = grammar_checker(sentence, truncation=True, max_length=512)[0]

    # Convert to show probability of being grammatically acceptable
    is_acceptable = True if result['label'] == 'LABEL_1' else False
    raw_score = result['score']

    final_score = raw_score if is_acceptable else (1.0 - raw_score)

    # Print results
    print("\n" + "=" * 50)
    print("GRAMMAR EVALUATION")
    print("=" * 50)
    print(f"Sentence:  {sentence}")
    print(f"Label:     {'Acceptable (LABEL_1)' if is_acceptable else 'Unacceptable (LABEL_0)'}")
    print(f"Raw Score: {raw_score:.4f} (Model Confidence)")
    print(f"-> FINAL CALCULATED SCORE: {final_score:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    # Change this to evaluate other sentences
    sentence = "Schering - Plough's cell therapy product was a result of an acquisition by Blue Dart Aviation Ltd, which Ernst Young LLP's employee Iyer oversaw; this company is a subsidiary of Prologis."

    check_single_sentence(sentence)