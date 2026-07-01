import torch
import torch.nn.functional as F
import random
from transformers import AutoTokenizer
from model import PhantomLM
from config import PhantomLMConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Tokenizer and Model
import bitlinear
bitlinear.QUANTIZE_ENABLED = False

print("Loading Model and Tokenizer for n=100 Evaluation...")
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
config = PhantomLMConfig.phantom_medium()
config.vocab_size = 50257

try:
    model = PhantomLM(config).to(device)
    ckpt = torch.load('/kaggle/working/checkpoints/phantomlm_final.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("FP16 Model loaded successfully!\n")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


def get_log_prob(context, completion):
    full_text = context + completion
    input_ids = torch.tensor([tokenizer.encode(full_text)]).to(device)
    context_ids = torch.tensor([tokenizer.encode(context)]).to(device)
    
    completion_length = input_ids.shape[1] - context_ids.shape[1]
    if completion_length <= 0:
        return -9999.0
        
    with torch.no_grad():
        logits = model(input_ids, return_loss=False)
        
    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    target_log_probs = log_probs[-completion_length:]
    target_labels = shift_labels[-completion_length:]
    
    score = 0
    for i in range(completion_length):
        score += target_log_probs[i, target_labels[i]].item()
        
    return score / completion_length

# ════════════════════════════════════════════════════
# 1. Grammar Generation (100 pairs)
# ════════════════════════════════════════════════════
def generate_grammar_pairs(n=100):
    pairs = []
    
    plural_nouns = ['cats', 'dogs', 'birds', 'boys', 'girls', 'trees', 'cars', 'bears', 'frogs', 'horses']
    verbs_ing = ['sleeping', 'running', 'jumping', 'playing', 'eating', 'walking', 'singing', 'flying', 'swimming', 'hiding']
    
    vowel_nouns = ['apple', 'elephant', 'orange', 'owl', 'egg', 'igloo', 'ice', 'ant', 'uncle', 'island']
    
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'pink', 'purple', 'orange']
    nouns = ['ball', 'house', 'car', 'flower', 'bird', 'dog', 'cat', 'tree', 'book', 'toy']

    while len(pairs) < n:
        t = random.choice([1, 2, 3])
        if t == 1:
            # Subject-verb agreement
            pn = random.choice(plural_nouns)
            v = random.choice(verbs_ing)
            pairs.append((f"The {pn} are {v}.", f"The {pn} is {v}."))
        elif t == 2:
            # Article agreement
            vn = random.choice(vowel_nouns)
            pairs.append((f"He saw an {vn}.", f"He saw a {vn}."))
        elif t == 3:
            # Adjective order
            c = random.choice(colors)
            n_ = random.choice(nouns)
            pairs.append((f"He threw the {c} {n_}.", f"He threw the {n_} {c}."))
            
    # Ensure exactly n unique or randomly sampled
    return pairs[:n]

def eval_grammar(pairs):
    print(f"Evaluating Custom Grammar (n={len(pairs)})...")
    passed = 0
    for correct, incorrect in pairs:
        score_c = get_log_prob("", correct)
        score_i = get_log_prob("", incorrect)
        if score_c > score_i:
            passed += 1
    acc = (passed / len(pairs)) * 100
    print(f"Grammar Accuracy: {acc:.1f}%\n")
    return acc

# ════════════════════════════════════════════════════
# 2. Logic Generation (100 questions)
# ════════════════════════════════════════════════════
def generate_logic_questions(n=100):
    tests = []
    
    animals = ['dog', 'cat', 'lion', 'tiger', 'bear', 'wolf', 'fox', 'deer', 'rabbit', 'mouse']
    objects = ['rock', 'stone', 'brick', 'table', 'chair', 'wall', 'stick', 'coin', 'book', 'cup']
    
    states = [
        ('hungry', ' kitchen', ' bathroom'),
        ('tired', ' bed', ' tree'),
        ('thirsty', ' river', ' desert'),
        ('cold', ' fire', ' freezer'),
        ('scared', ' house', ' monster')
    ]
    
    while len(tests) < n:
        t = random.choice([1, 2, 3])
        if t == 1:
            a = random.choice(animals)
            tests.append((f"Premise: A {a} is a mammal. All mammals are animals.\nQuestion: Is a {a} an animal?\nAnswer:", " Yes", " No"))
        elif t == 2:
            o = random.choice(objects)
            tests.append((f"Premise: A {o} is a solid. No solids are alive.\nQuestion: Is a {o} alive?\nAnswer:", " No", " Yes"))
        elif t == 3:
            st, g_end, b_end = random.choice(states)
            n_ = random.choice(['Tim', 'Lily', 'Tom', 'Anna', 'Sam', 'Mia'])
            tests.append((f"{n_} was very {st}, so they went to the", g_end, b_end))
            
    return tests[:n]

def eval_logic(tests):
    print(f"Evaluating Deductive Logic (n={len(tests)})...")
    passed = 0
    for ctx, good, bad in tests:
        score_g = get_log_prob(ctx, good)
        score_b = get_log_prob(ctx, bad)
        if score_g > score_b:
            passed += 1
    acc = (passed / len(tests)) * 100
    print(f"Logic Accuracy: {acc:.1f}%\n")
    return acc

# ════════════════════════════════════════════════════
# 3. Diversity (20 prompts)
# ════════════════════════════════════════════════════
def eval_diversity(n_prompts=20):
    print(f"Evaluating Distinct-1 Diversity (n={n_prompts} prompts)...")
    prompts = [
        "Once upon a time", "In a dark forest", "The little boy", 
        "A small dog", "One sunny day", "Deep in the cave",
        "The magical tree", "A brave girl", "The old man",
        "A tiny mouse", "The big bear", "On top of the hill",
        "The flying bird", "A red apple", "The green frog",
        "Under the bridge", "The lost cat", "A happy family",
        "The mysterious box", "A shiny coin"
    ]
    
    all_words = []
    
    for p in prompts:
        ids = torch.tensor([tokenizer.encode(p)]).to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=50, temperature=0.8)
        gen = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        words = gen.lower().split()
        all_words.extend(words)
        
    unique = set(all_words)
    diversity = (len(unique) / len(all_words)) * 100
    print(f"Distinct-1 Score: {diversity:.1f}% ({len(unique)} unique / {len(all_words)} total words)\n")
    return diversity

if __name__ == "__main__":
    grammar_pairs = generate_grammar_pairs(100)
    logic_tests = generate_logic_questions(100)
    
    g_acc = eval_grammar(grammar_pairs)
    l_acc = eval_logic(logic_tests)
    div_score = eval_diversity(20)
    
    print("="*70)
    print("UPDATED n=100 EVALUATION METRICS")
    print("="*70)
    print(f"Custom Grammar (Log-Prob)        : {g_acc:.1f}%   (n=100 pairs)")
    print(f"Custom Deductive Logic (Log-Prob): {l_acc:.1f}%   (n=100 questions)")
    print(f"Distinct-1 (Unigram Diversity)   : {div_score:.1f}%   (n=20 prompts, ~1000 tokens)")
    print("="*70)
