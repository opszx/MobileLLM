import torch

def evaluate_zeroshot_logic(model, tokenizer):
    """
    Evaluates basic reasoning and grammar by giving the model a prefix
    and comparing the probability of a logical completion vs an illogical one.
    
    This is necessary because standard benchmarks (MMLU, PIQA) require 
    general world knowledge that a TinyStories model does not have.
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Test cases: Prefix -> (Logical Completion, Illogical Completion)
    tests = [
        ("Tim was very hungry, so he ate an", " apple", " car"),
        ("Lily was tired, so she went to", " bed", " tree"),
        ("The dog barked loudly at the", " cat", " sky"),
        ("It was raining outside, so he took his", " umbrella", " chair")
    ]
    
    print(f"{'Prefix':<45} | {'Logical Score':<15} | {'Illogical Score':<15} | Result")
    print("-" * 95)
    
    correct = 0
    with torch.no_grad():
        for prefix, good_ending, bad_ending in tests:
            
            # Encode inputs
            prefix_ids = tokenizer.encode(prefix, return_tensors='pt').to(device)
            good_ids = tokenizer.encode(good_ending, return_tensors='pt').to(device)
            bad_ids = tokenizer.encode(bad_ending, return_tensors='pt').to(device)
            
            # Function to calculate probability of ending given prefix
            def score_completion(ending_ids):
                full_ids = torch.cat([prefix_ids, ending_ids], dim=1)
                logits = model(full_ids)
                
                # We only care about the logits predicting the ending tokens
                # Shift logits by 1 (predicting next token)
                shift_logits = logits[0, prefix_ids.shape[1]-1:-1, :]
                shift_labels = ending_ids[0]
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                nll = loss_fct(shift_logits, shift_labels)
                # Return log probability (higher is better)
                return -nll.item()
                
            score_good = score_completion(good_ids)
            score_bad = score_completion(bad_ids)
            
            is_correct = score_good > score_bad
            if is_correct:
                correct += 1
                
            print(f"{prefix:<45} | {score_good:>15.2f} | {score_bad:>15.2f} | {'PASS' if is_correct else 'FAIL'}")
            
    accuracy = (correct / len(tests)) * 100
    print("-" * 95)
    print(f"Zero-Shot Logic Accuracy: {accuracy:.1f}%")

# Example usage:
# evaluate_zeroshot_logic(model, tokenizer)
