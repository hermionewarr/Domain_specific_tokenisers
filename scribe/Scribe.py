#https://github.com/karpathy/nanoGPT/blob/master/model.py
"""
Adapted from Kaparthy's nanoGPT model.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
torch.autograd.set_detect_anomaly(True)
# NOTE: Don't initialize device at module level - it forces CUDA initialization on import
# and defaults to cuda:0, which prevents proper GPU selection via --gpu argument
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_LAUNCH_BLOCKING=1
@dataclass
class ScribeConfig:
    block_size: int = 1024 #256 <- see if this is enough for the reports. 512 was used until the 6th dec
    vocab_size: int = 50304 #5000 # 
    n_layer: int = 8 #16 #8 #12
    n_head: int =  8 #16. 8
    n_embd: int = 512 #256  #1024 #512 #512 #768 #800
    #im_embd: int = 512
    dropout: float = 0.0 #0.25 #0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    #concat: bool = True #True # whether to concatenate the image features to the token embeddings
    
class Scribe(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        print("config.vocab_size", config.vocab_size)
        #self.bos_token_id = 50256
        #self.eos_token_id = 50256

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            #wpe = nn.Embedding(config.block_size, config.n_embd),
            rope = RotaryPositionalEmbedding(config.n_embd, config.block_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        #self.rope = RotaryPositionalEmbedding(config.n_embd, config.block_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        #self.im_embed_tok = nn.Linear(config.n_embd, 10*config.n_embd, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        #print("number params minus pos embeds: ",n_params-self.transformer.rope.weight.numel()) #doesnt work
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    """ # Initialize cache if needed
        if use_cache:
            if past_key_values is None:
                past_key_values = [None] * self.num_layers 

        # Process each layer
        for layer_idx, layer in enumerate(self.layers):
            # Pass past key-values to each layer if provided
            layer_past = past_key_values[layer_idx] if past_key_values else None
            layer_output = layer(idx, attn_mask, layer_past)

            # Update the cache if caching is enabled
            if use_cache:
                past_key_values[layer_idx] = layer_output.new_key_value """
        

    def forward(self, 
                idx, 
                attn_mask = None,
                targets = None, 
                past_key_values = None,
                use_cache = False,
                image_features = None
                ):
        
        b, t = idx.size()
        device = idx.device

        if attn_mask is not None:
            #MASK 
            # get a boolean copy of the attention_mask and invert it
            mask_to_ignore_padding_tokens_for_loss_computation = ~(attn_mask.to(torch.bool))
            attn_mask = attn_mask.view(b, -1)
            attn_mask = attn_mask[:, None, None, :]
            # Reshape from [B, 1, 1, T] to [B, 1, T, T] for flash attention
            B, _, _, T = attn_mask.shape
            attn_mask = attn_mask.expand(B, 1, T, T)
            attn_mask = attn_mask.to(dtype=torch.bool)
        
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        #assert max(idx[0]) < self.config.vocab_size, "Token ID exceeds vocab size!"

        
        tok_emb = self.transformer.wte(idx.clone()) # token embeddings idky i need to clone it rip
        
        # Add image features if provided
        if image_features is not None:
            # Project image features to match token embedding dimension
            image_features = image_features #.unsqueeze(1)  # Add sequence dimension
            
            # Get number of image tokens dynamically from feature shape
            num_image_tokens = image_features.size(1)
            
            tok_emb = torch.cat([image_features, tok_emb], dim=1) # Add image features to token embeddings
            
            new_attn_mask = torch.ones(size=(b, num_image_tokens), dtype=torch.int64, device=device)
            
            attn_mask = torch.cat([new_attn_mask[:,None,None,:], attn_mask], dim=-1)
            

        if image_features is not None:
            num_image_tokens = image_features.size(1)
            t = t + num_image_tokens
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        pos_emb = self.transformer.rope(pos) #rotary positional embeddings
        x = self.transformer.drop(tok_emb + pos_emb)#

        # Initialize new past_key_values if caching
        new_past_key_values = [] if use_cache else None

        for i, block in enumerate(self.transformer.h):
            # Pass the current layer's past_key_values
            layer_past = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                x, new_kv = block(x, attn_mask, layer_past=layer_past, use_cache=use_cache)
                new_past_key_values.append(new_kv)
            else:
                x = block(x, attn_mask) 

        x = self.transformer.ln_f(x)
        
        if targets is not None: #potnetially put the loss in the training loop
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) 
            if image_features is not None:
                logits = logits[:,10:,:]
            shift_logits = logits[:,:-1,:].contiguous()
            labels = targets 
            # set padding tokens to -100, such that they are ignored and don't count towards the loss
            labels[mask_to_ignore_padding_tokens_for_loss_computation] = -100
            shift_labels = labels[:,1:].contiguous()
            # flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

            return x, shift_logits, loss #adding x 12/6/25
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # This dramatically reduces memory usage during validation/inference
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None #, new_past_key_values

            return x, logits, loss, past_key_values if use_cache else None #adding x 12/6/25
    
    @torch.no_grad()
    def generate(self, 
                 idx,
                 attn_mask, 
                 max_new_tokens,  
                 tokenizer,
                 top_k=None,#5 
                 stop_token_id=None,
                 image_features=None,
                 temperature=1.0,
                 top_p=None
                 ):
       
        batch_size, current_length = idx.size()
        device = idx.device  # Get device from input tensor
       
        past_key_values = None  # Start with no past cache 
        use_cache = False #True s'not working with flash attention.
        i=0
        gen_idx = None
        max_new_tokens = max_new_tokens - current_length 
        stop_token_id = stop_token_id or tokenizer.bos_token_id #encode('[BOS]') #tokenizer.eos_token_id  # Default to EOS token if not provided
        
        #print("stop token id: ", stop_token_id) 
        #print("max new tokens: ", max_new_tokens)
        if idx is None:
            idx = torch.full(size=(batch_size, 1), fill_value=tokenizer.bos_token_id, dtype=torch.int64, device=device)

        if attn_mask == None:
            attention_mask = torch.ones(size=(batch_size, idx.shape[-1]), dtype=torch.int64, device=device)
        else:
            attention_mask = attn_mask
     
        #generation loop
        for _ in range(max_new_tokens):
            _,logits, _,  past_key_values = self(idx, attention_mask.to(bool), use_cache=use_cache, past_key_values=past_key_values, image_features=image_features)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            #probs = F.softmax(logits, dim=-1)
            idx_next = torch.argmax(logits, dim=-1).unsqueeze(-1)
            #idx_next = torch.multinomial(probs, num_samples=1) #if you want to sample from the distribution
            
            idx = torch.cat((idx, idx_next), dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)
  
            torch.cuda.empty_cache()
            i+=1
            # Stop generation if all sequences predict a stop token
            if (batch_size == 1) and (idx_next==stop_token_id):
                print("stop token predicted")
                break
            elif batch_size >1 and torch.all(idx[:,-1]==stop_token_id):
                break

        for i in range(idx.size(0)): 
                row = idx[i]
                indices = (row == torch.tensor(stop_token_id).to(device)).nonzero() #.encode('[BOS]')
                if len(indices) > 1:
                    first_occurrence_index = indices[1].item() #1 rather than 0 as start of report has bos token
                    idx[i, first_occurrence_index+1:] = torch.tensor(stop_token_id) #2 
        #print("genereted rep:")
        #print(tokenizer.batch_decode(idx))
        return idx
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        #self.concat = config.concat
        self.config = config
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        """ if self.concat:
            #self.flash = False #to have more control over the attention mask
            #(to make sure the mask does not get updated during backprop)
            tril = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size) 
            self.register_buffer("bias", tril) """
        
    def forward(self, x, attention_mask, layer_past=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        if layer_past is not None:
            past_k, past_v = layer_past
            #print(k.size())
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2) #this currently doesnt work with flash attentions
            #print(k.size())
            
        T_q = q.size(1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_q, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T_q, hs) x (B, nh, hs, T) -> (B, nh, T_q, T)
        if self.flash:
            #print("flashing")
            #y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            
        """ else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attention_mask = (1.0 - attention_mask) * -100000.0
            att = att.masked_fill(self.bias[:,:,:T_q,:T] == 0, float('-inf')) #+ attention_mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T_q, T) x (B, nh, T, hs) -> (B, nh, T, hs) """
        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, (k , v)
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask, layer_past=None, use_cache=False):
        attn_out, new_past_kv =self.attn(self.ln_1(x), attn_mask, layer_past)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return (x, new_past_kv) if use_cache else x

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_positions):
        super(RotaryPositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.inv_freq = 1. / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))

    def forward(self, positions):
        """
        Args:
            positions (torch.Tensor): 1-D tensor containing the sequence positions.

        Returns:
            torch.Tensor: Rotary positional embeddings for the given positions.
        """
        device = positions.device  # Get device from input tensor
        sinusoids = torch.einsum("i,d->id", positions.float(), self.inv_freq.to(device))
        embeddings = torch.cat([sinusoids.sin(), sinusoids.cos()], dim=-1)
        return embeddings[:, :self.embed_dim]