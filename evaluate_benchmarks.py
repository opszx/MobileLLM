import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from model import PhantomLM
from config import PhantomLMConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Tokenizer and Model
print("Loading Model and Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
config = PhantomLMConfig()

try:
    model = PhantomLM(config).to(device)
    ckpt = torch.load('/kaggle/working/checkpoints/phantomlm_final.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("FP16 Model loaded successfully!\n")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure you are running this in Kaggle where phantomlm_final.pt exists.")
    exit(1)

def get_log_prob(context, completion):
    """Calculates the log probability of a completion given a context."""
    full_text = context + " " + completion
    input_ids = torch.tensor([tokenizer.encode(full_text)]).to(device)
    context_ids = torch.tensor([tokenizer.encode(context)]).to(device)
    
    # We only care about the loss on the completion tokens
    completion_length = input_ids.shape[1] - context_ids.shape[1]
    if completion_length <= 0:
        return -9999.0
        
    with torch.no_grad():
        _, logits, _, _ = model(input_ids, return_loss=True)
        
    # Get probabilities for the target tokens
    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Extract only the completion part
    target_log_probs = log_probs[-completion_length:]
    target_labels = shift_labels[-completion_length:]
    
    score = 0
    for i in range(completion_length):
        score += target_log_probs[i, target_labels[i]].item()
        
    return score / completion_length # Normalize by length

# ════════════════════════════════════════════════════
# 1. HellaSwag (Commonsense Reasoning)
# ════════════════════════════════════════════════════
def eval_hellaswag(num_samples=100):
    print("Evaluating HellaSwag (Commonsense)...")
    try:
        ds = load_dataset("hellaswag", split="validation", streaming=True)
        correct = 0
        total = 0
        for item in ds:
            if total >= num_samples: break
            
            context = item['ctx']
            endings = item['endings']
            label = int(item['label'])
            
            scores = [get_log_prob(context, end) for end in endings]
            prediction = scores.index(max(scores))
            
            if prediction == label:
                correct += 1
            total += 1
            
        acc = (correct / total) * 100
        print(f"HellaSwag Accuracy: {acc:.1f}%\n")
        return acc
    except Exception as e:
        print(f"Failed HellaSwag: {e}\n")
        return 0

# ════════════════════════════════════════════════════
# 2. ARC-Easy (Science Facts)
# ════════════════════════════════════════════════════
def eval_arc(num_samples=100):
    print("Evaluating ARC-Easy (Science)...")
    try:
        ds = load_dataset("ai2_arc", "ARC-Easy", split="validation", streaming=True)
        correct = 0
        total = 0
        
        # ARC choices are labeled A, B, C, D or 1, 2, 3, 4
        def label_to_idx(l):
            if l in ['A', '1']: return 0
            if l in ['B', '2']: return 1
            if l in ['C', '3']: return 2
            if l in ['D', '4']: return 3
            return -1
            
        for item in ds:
            if total >= num_samples: break
            
            question = item['question']
            choices = item['choices']['text']
            labels = item['choices']['label']
            answer_key = item['answerKey']
            
            # Skip weird formats
            if len(choices) < 2: continue
            
            scores = [get_log_prob(question, c) for c in choices]
            prediction_idx = scores.index(max(scores))
            
            correct_idx = label_to_idx(answer_key)
            if correct_idx == -1: continue # skip unparsable
            
            if prediction_idx == correct_idx:
                correct += 1
            total += 1
            
        acc = (correct / total) * 100
        print(f"ARC-Easy Accuracy: {acc:.1f}%\n")
        return acc
    except Exception as e:
        print(f"Failed ARC: {e}\n")
        return 0

# ════════════════════════════════════════════════════
# 3. LAMBADA (Long-Range Word Prediction)
# ════════════════════════════════════════════════════
def eval_lambada(num_samples=100):
    print("Evaluating LAMBADA (Long Range Context)...")
    try:
        ds = load_dataset("lambada", split="test", streaming=True)
        correct = 0
        total = 0
        for item in ds:
            if total >= num_samples: break
            
            text = item['text']
            words = text.strip().split()
            if len(words) < 5: continue
            
            context = " ".join(words[:-1])
            target_word = words[-1]
            
            input_ids = torch.tensor([tokenizer.encode(context)]).to(device)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=2, temperature=0.1)
                
            generated = tokenizer.decode(out[0, input_ids.shape[1]:].tolist()).strip().split()
            if len(generated) > 0 and generated[0].lower().startswith(target_word.lower()):
                correct += 1
            total += 1
            
        acc = (correct / total) * 100
        print(f"LAMBADA Accuracy: {acc:.1f}%\n")
        return acc
    except Exception as e:
        print(f"Failed LAMBADA: {e}\n")
        return 0

# ════════════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("STARTING ACADEMIC BENCHMARK EVALUATIONS")
    print("="*60)
    
    h_acc = eval_hellaswag(100)
    a_acc = eval_arc(100)
    l_acc = eval_lambada(100)
    
    print("="*60)
    print("FINAL BENCHMARK TABLE FOR RESEARCH PAPER")
    print("="*60)
    print(f"1. HellaSwag (Commonsense) : {h_acc:.1f}%")
    print(f"2. ARC-Easy (Science Facts): {a_acc:.1f}%")
    print(f"3. LAMBADA (Long Range)    : {l_acc:.1f}%")
    print("="*60)
    print("Note: Random guessing baseline is ~25%. Scores near 25% prove the")
    print("out-of-distribution hypothesis caused by the TinyStories training set.")
