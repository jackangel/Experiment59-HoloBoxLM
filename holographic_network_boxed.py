import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import requests
import os
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# --- Hyperparameters ---
VOCAB_SIZE = 4096   
EMBED_DIM = 512    
MATRIX_DIM = 64    
NUM_LAYERS = 2
SEQ_LEN = 256
BATCH_SIZE = 64    
LEARNING_RATE = 5e-4 
STEPS = 5000        
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'holo_box_stable.pth'
TOKENIZER_FILE = 'holo_tokenizer.json'

# --- 1. Stable Box Logic ---
@torch.jit.script
def stable_box_score(q_min, q_max, v_min, v_max):
    inter_min = torch.max(q_min, v_min)
    inter_max = torch.min(q_max, v_max)
    width = F.softplus(inter_max - inter_min)
    log_vol = torch.mean(torch.log(width + 1e-6), dim=-1) 
    return log_vol

# --- 2. JIT Compiled Associative Core ---
@torch.jit.script
def holo_scan(x, memory, 
              w_k, b_k, w_q, b_q, w_v, b_v, w_out, b_out, 
              w_gw, b_gw, w_gf, b_gf):
    outputs: list[torch.Tensor] = []
    k_all = F.normalize(F.linear(x, w_k, b_k), p=2.0, dim=-1)
    q_all = F.normalize(F.linear(x, w_q, b_q), p=2.0, dim=-1)
    v_all = torch.tanh(F.linear(x, w_v, b_v))
    gw_all = torch.sigmoid(F.linear(x, w_gw, b_gw)).unsqueeze(-1)
    gf_all = torch.sigmoid(F.linear(x, w_gf, b_gf)).unsqueeze(-1)
    
    for t in range(x.size(1)):
        k, q, v = k_all[:, t], q_all[:, t], v_all[:, t]
        beta, decay = gw_all[:, t], gf_all[:, t]
        readout = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        association = torch.bmm(v.unsqueeze(-1), k.unsqueeze(1))
        memory = (decay * memory) + (beta * association)
        outputs.append(readout)
        
    return F.linear(torch.stack(outputs, dim=1), w_out, b_out), memory

# --- 3. Hybrid Modules ---
class BoxEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center = nn.Embedding(vocab_size, embed_dim)
        self.offset = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.center.weight)
        nn.init.constant_(self.offset.weight, -2.0)

    def get_boxes(self, idx=None):
        if idx is None:
            c, o = self.center.weight, F.softplus(self.offset.weight)
        else:
            c, o = self.center(idx), F.softplus(self.offset(idx))
        return c - o, c + o

class HoloGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, matrix_dim, layers):
        super().__init__()
        self.box_emb = BoxEmbedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            FastHoloBoxBlock(embed_dim, matrix_dim) for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.q_offset_net = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0) 
        
    def forward(self, idx, states=None):
        b_min, b_max = self.box_emb.get_boxes(idx)
        x = (b_min + b_max) / 2.0 
        
        new_states = []
        if states is None: states = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, s = layer(x, states[i])
            new_states.append(s)
            
        x = self.ln_f(x)
        center_logits = F.linear(x, self.box_emb.center.weight) 
        logits = center_logits * (self.logit_scale / EMBED_DIM**0.5)
        
        return logits, new_states

class FastHoloBoxBlock(nn.Module):
    def __init__(self, embed_dim, matrix_dim):
        super().__init__()
        self.matrix_dim = matrix_dim
        self.proj_k = nn.Linear(embed_dim, matrix_dim)
        self.proj_q = nn.Linear(embed_dim, matrix_dim)
        self.proj_v = nn.Linear(embed_dim, matrix_dim)
        self.proj_out = nn.Linear(matrix_dim, embed_dim)
        self.gate_write = nn.Linear(embed_dim, 1)
        self.gate_forget = nn.Linear(embed_dim, 1)
        self.ln1, self.ln2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim), nn.GELU(), nn.Linear(4*embed_dim, embed_dim))

    def forward(self, x, memory=None):
        if memory is None: memory = torch.zeros(x.shape[0], self.matrix_dim, self.matrix_dim, device=x.device)
        x_n = self.ln1(x)
        m_out, new_mem = holo_scan(x_n, memory, self.proj_k.weight, self.proj_k.bias, self.proj_q.weight, self.proj_q.bias,
                                   self.proj_v.weight, self.proj_v.bias, self.proj_out.weight, self.proj_out.bias,
                                   self.gate_write.weight, self.gate_write.bias, self.gate_forget.weight, self.gate_forget.bias)
        x = x + m_out
        x = x + self.ffn(self.ln2(x))
        return x, new_mem

# --- 4. Sampling Helper ---
def sample_top_p_top_k(logits, temperature=1.0, top_k=50, top_p=0.9):
    logits = logits / temperature
    
    # Top-K
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    
    # Top-P (Nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        for b in range(logits.size(0)):
            indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
            logits[b, indices_to_remove] = -float('Inf')
            
    return logits

# --- 5. Chat Interface ---
def run_chat_mode(model, tokenizer):
    print("\n--- Entering Chat Mode ---")
    print("Settings: Temp=0.8, Top-K=50, Top-P=0.9")
    print("(Type 'exit' to quit)")
    
    model.eval()
    
    TEMPERATURE = 0.8
    TOP_K = 50
    TOP_P = 0.9

    while True:
        try:
            prompt = input("\nYou: ")
        except EOFError:
            break

        if not prompt or prompt.strip() == "":
            continue
            
        if prompt.lower() == "exit": break
        
        # 1. Tokenize Input
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
        
        # Warmup memory / Process Prompt
        _, states = model(idx)
        
        print(f"Holo: ", end="", flush=True)
        curr = idx[:, -1:]
        
        # Generate
        with torch.no_grad():
            for _ in range(200): # Generation length
                logits, states = model(curr, states)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :]
                
                # Apply Top-K / Top-P / Temperature
                filtered_logits = sample_top_p_top_k(
                    next_token_logits, 
                    temperature=TEMPERATURE, 
                    top_k=TOP_K, 
                    top_p=TOP_P
                )
                
                # Convert to probabilities and Sample
                probs = F.softmax(filtered_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1)
                
                # Decode Token
                decoded_char = tokenizer.decode([next_token_id.item()])
                
                print(decoded_char, end="", flush=True)
                curr = next_token_id
        print()

# --- 6. Utilities & Main ---
def get_tokenizer_and_data():
    if not os.path.exists('input.txt'):
        r = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        with open('input.txt', 'w') as f: f.write(r.text)
    if os.path.exists(TOKENIZER_FILE): 
        tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.train(["input.txt"], BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]"]))
        tokenizer.save(TOKENIZER_FILE)
    
    with open('input.txt', 'r') as f: text = f.read()
    return torch.tensor(tokenizer.encode(text).ids, dtype=torch.long), tokenizer

def main():
    data, tokenizer = get_tokenizer_and_data()
    model = HoloGPT(tokenizer.get_vocab_size(), EMBED_DIM, MATRIX_DIM, NUM_LAYERS).to(DEVICE)
    
    # CHECKPOINT CHECK
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint found at {CHECKPOINT_PATH}. Loading model...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        run_chat_mode(model, tokenizer)
        return # Exit after chat mode finishes

    # TRAINING MODE
    print("No checkpoint found. Starting training...")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(range(STEPS), desc="Training Stable Holo-Box")
    for step in pbar:
        ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
        xb = torch.stack([data[i:i+SEQ_LEN] for i in ix]).to(DEVICE)
        yb = torch.stack([data[i+1:i+SEQ_LEN+1] for i in ix]).to(DEVICE)
        
        logits, _ = model(xb)
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 20 == 0:
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Model saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()