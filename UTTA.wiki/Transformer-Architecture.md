# Chapter 3: Transformer Architecture

[![Status](https://img.shields.io/badge/status-complete-green.svg)]() 
[![Last Updated](https://img.shields.io/badge/last%20updated-April%202024-blue.svg)]()

## Learning Objectives

By the end of this chapter, you'll be able to:
- Understand the fundamental concepts of attention mechanisms
- Implement self-attention and multi-head attention components
- Explain the encoder-decoder architecture of transformers
- Apply positional encoding to sequence data
- Analyze transformer performance on different types of text

## 3.1 Introduction to Transformer Architecture

The Transformer architecture, introduced in the landmark 2017 paper "Attention Is All You Need" by Vaswani et al., revolutionized natural language processing and forms the foundation of modern large language models.

### 3.1.1 Evolution of Neural Networks for NLP

Before transformers, sequence processing relied on recurrent architectures:

| Architecture | Key Features | Limitations |
|--------------|--------------|-------------|
| **RNNs** | Sequential processing<br>Hidden state memory | Vanishing gradients<br>Limited context window |
| **LSTMs/GRUs** | Improved memory cells<br>Better gradient flow | Still sequential<br>Computational inefficiency |
| **Seq2Seq** | Encoder-decoder structure<br>Attention mechanisms | Bottleneck at fixed-size context vector |
| **Transformers** | Parallel processing<br>Self-attention | Quadratic complexity<br>Position information loss |

The key innovation of transformers is replacing recurrence with attention mechanisms, allowing parallel processing of sequences and capturing long-range dependencies more effectively.

### 3.1.2 The Transformer Architecture

The complete transformer architecture consists of encoder and decoder stacks:

```
┌─────────────────────┐                  ┌─────────────────────┐
│     Encoder Stack   │                  │    Decoder Stack    │
├─────────────────────┤                  ├─────────────────────┤
│  ┌───────────────┐  │                  │  ┌───────────────┐  │
│  │ Self-Attention │  │                  │  │ Self-Attention │  │
│  └───────────────┘  │                  │  └───────────────┘  │
│         ↓           │                  │         ↓           │
│  ┌───────────────┐  │                  │  ┌───────────────┐  │
│  │ Feed Forward  │  │                  │  │Cross-Attention │  │
│  └───────────────┘  │                  │  └───────────────┘  │
│                     │                  │         ↓           │
└─────────────────────┘                  │  ┌───────────────┐  │
          ↓                              │  │ Feed Forward  │  │
          └─────────────────────────────→│  └───────────────┘  │
                                         └─────────────────────┘
```

This architecture enables:
- Parallel processing of all sequence positions
- Direct connections between any positions in the sequence
- Effective modeling of long-range dependencies
- Scalable model capacity through stacking layers

## 3.2 Attention Mechanisms

Attention is the core innovation that enables transformers to process sequences effectively.

### 3.2.1 The Intuition Behind Attention

Attention mechanisms allow a model to focus on relevant parts of the input when producing each element of the output:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def simple_attention(query, keys, values):
    """
    Simple attention mechanism
    
    Args:
        query (torch.Tensor): Query vector [query_dim]
        keys (torch.Tensor): Key vectors [seq_len, key_dim]
        values (torch.Tensor): Value vectors [seq_len, value_dim]
        
    Returns:
        torch.Tensor: Context vector [value_dim]
    """
    # Calculate attention scores
    scores = torch.matmul(query, keys.transpose(0, 1))  # [seq_len]
    
    # Apply softmax to get attention weights
    weights = F.softmax(scores, dim=0)  # [seq_len]
    
    # Apply weights to values
    context = torch.matmul(weights, values)  # [value_dim]
    
    return context
```

### 3.2.2 Self-Attention

Self-attention allows each position in a sequence to attend to all positions, capturing relationships regardless of distance:

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        Self-attention forward pass
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        q = self.query_proj(x)  # [batch_size, seq_len, embed_dim]
        k = self.key_proj(x)    # [batch_size, seq_len, embed_dim]
        v = self.value_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        scores = scores / (self.embed_dim ** 0.5)  # Scale by sqrt(d_k)
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # Apply attention weights to values
        context = torch.matmul(weights, v)  # [batch_size, seq_len, embed_dim]
        
        # Output projection
        output = self.out_proj(context)  # [batch_size, seq_len, embed_dim]
        
        return output
```

### 3.2.3 Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        Multi-head attention forward pass
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        q = self.query_proj(x)  # [batch_size, seq_len, embed_dim]
        k = self.key_proj(x)    # [batch_size, seq_len, embed_dim]
        v = self.value_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / (self.head_dim ** 0.5)  # Scale by sqrt(d_k)
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply attention weights to values
        context = torch.matmul(weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape back to original dimensions
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(context)  # [batch_size, seq_len, embed_dim]
        
        return output
```

### 3.2.4 Attention Visualization

Attention weights can be visualized to understand what the model is focusing on:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(tokens, attention_weights):
    """
    Visualize attention weights
    
    Args:
        tokens (list): List of tokens
        attention_weights (torch.Tensor): Attention weights [seq_len, seq_len]
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights.detach().cpu().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        annot=True,
        fmt=".2f"
    )
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.title("Attention Weights")
    plt.tight_layout()
    plt.show()
```

## 3.3 Positional Encoding

Since transformers process all positions simultaneously, they need explicit position information.

### 3.3.1 The Need for Position Information

Unlike RNNs, transformers have no inherent notion of sequence order. Positional encoding adds this information to the input embeddings.

### 3.3.2 Sinusoidal Positional Encoding

The original transformer paper used sinusoidal functions to encode positions:

```python
def positional_encoding(seq_len, embed_dim):
    """
    Generate sinusoidal positional encoding
    
    Args:
        seq_len (int): Sequence length
        embed_dim (int): Embedding dimension
        
    Returns:
        torch.Tensor: Positional encoding [seq_len, embed_dim]
    """
    # Create position indices
    positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
    
    # Create dimension indices
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / embed_dim)
    )
    
    # Create positional encoding
    pe = torch.zeros(seq_len, embed_dim)
    pe[:, 0::2] = torch.sin(positions * div_term)  # Even dimensions
    pe[:, 1::2] = torch.cos(positions * div_term)  # Odd dimensions
    
    return pe
```

### 3.3.3 Learned Positional Embeddings

Modern transformers often use learned positional embeddings:

```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super(LearnedPositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_seq_len, embed_dim)
        
    def forward(self, x):
        """
        Add learned positional embeddings to input
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: Output with positional information [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.embedding(positions)
        
        return x + pos_embeddings
```

## 3.4 Encoder-Decoder Architecture

The full transformer consists of encoder and decoder components working together.

### 3.4.1 Encoder Block

Each encoder block contains self-attention and feed-forward layers:

```python
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(EncoderBlock, self).__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Encoder block forward pass
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, embed_dim]
        """
        # Self-attention with residual connection and layer normalization
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 3.4.2 Decoder Block

Each decoder block adds cross-attention to the encoder outputs:

```python
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Multi-head cross-attention
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        """
        Decoder block forward pass
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]
            encoder_output (torch.Tensor): Encoder output [batch_size, enc_seq_len, embed_dim]
            self_attn_mask (torch.Tensor): Self-attention mask [batch_size, seq_len, seq_len]
            cross_attn_mask (torch.Tensor): Cross-attention mask [batch_size, seq_len, enc_seq_len]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, embed_dim]
        """
        # Self-attention with residual connection and layer normalization
        self_attn_output = self.self_attention(x)  # In practice, would use mask for autoregressive decoding
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with residual connection and layer normalization
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

### 3.4.3 Complete Transformer

Putting it all together for a complete transformer model:

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, num_heads, ff_dim, 
                 num_encoder_layers, num_decoder_layers, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Token embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(positional_encoding(max_seq_len, embed_dim))
        
        # Encoder and decoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt):
        """
        Transformer forward pass
        
        Args:
            src (torch.Tensor): Source tokens [batch_size, src_seq_len]
            tgt (torch.Tensor): Target tokens [batch_size, tgt_seq_len]
            
        Returns:
            torch.Tensor: Output logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Get sequence lengths
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        
        # Embed tokens and add positional encoding
        src_embedded = self.src_embedding(src) + self.positional_encoding[:src_seq_len, :]
        tgt_embedded = self.tgt_embedding(tgt) + self.positional_encoding[:tgt_seq_len, :]
        
        # Apply dropout
        src_embedded = self.dropout(src_embedded)
        tgt_embedded = self.dropout(tgt_embedded)
        
        # Encoder forward pass
        encoder_output = src_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output)
        
        # Decoder forward pass
        decoder_output = tgt_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output)
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output
```

## 3.5 Hands-On Exercise: Implementing Attention

Let's implement a simplified version of self-attention and apply it to educational text:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SimpleSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        
        # Linear projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        Self-attention forward pass
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            tuple: Output tensor and attention weights
        """
        # Project inputs to queries, keys, and values
        q = self.query(x)  # [batch_size, seq_len, embed_dim]
        k = self.key(x)    # [batch_size, seq_len, embed_dim]
        v = self.value(x)  # [batch_size, seq_len, embed_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        
        # Scale scores
        scores = scores / (self.embed_dim ** 0.5)
        
        # Compute attention weights
        weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # Apply attention weights to values
        output = torch.matmul(weights, v)  # [batch_size, seq_len, embed_dim]
        
        return output, weights

# Example usage
def analyze_educational_text():
    # Sample educational text
    text = "Effective classroom management requires clear expectations, consistent routines, and positive relationships with students."
    
    # Tokenize text
    tokens = text.split()
    
    # Create simple word embeddings (for demonstration)
    embed_dim = 16
    embeddings = {}
    
    for token in tokens:
        if token not in embeddings:
            # Random embedding for each unique token
            embeddings[token] = torch.randn(embed_dim)
    
    # Create input tensor
    seq_len = len(tokens)
    x = torch.stack([embeddings[token] for token in tokens]).unsqueeze(0)  # [1, seq_len, embed_dim]
    
    # Initialize self-attention module
    self_attention = SimpleSelfAttention(embed_dim)
    
    # Apply self-attention
    output, weights = self_attention(x)
    
    # Visualize attention weights
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        weights[0].detach().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        annot=True,
        fmt=".2f"
    )
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.title("Self-Attention Weights")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Analyze relationships
    print("Top relationships identified by self-attention:")
    for i, token_i in enumerate(tokens):
        for j, token_j in enumerate(tokens):
            if i != j and weights[0, i, j].item() > 0.1:
                print(f"'{token_i}' attends to '{token_j}' with weight {weights[0, i, j].item():.2f}")
    
    return output, weights, tokens

# Run the analysis
output, weights, tokens = analyze_educational_text()
```

## 3.6 UTTA Case Study: Analyzing Transformer Performance on Educational Text

The UTTA project provides an excellent opportunity to analyze how transformers process educational content.

### 3.6.1 Educational Text Characteristics

Educational content in the UTTA project has several characteristics that affect transformer performance:

1. **Domain-specific terminology**: Educational terms like "differentiated instruction" or "formative assessment" require specialized understanding
2. **Structured content**: Educational materials often follow specific formats with headings, lists, and sections
3. **Abstract concepts**: Many educational concepts are abstract and require deeper semantic understanding
4. **Varied text lengths**: From short instructions to lengthy explanations and case studies

### 3.6.2 Attention Pattern Analysis

We analyzed attention patterns in transformers processing educational content:

```python
def analyze_attention_patterns(model, tokenizer, text):
    """
    Analyze attention patterns in transformer model
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer
        text (str): Input text
        
    Returns:
        dict: Analysis results
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get model outputs with attention weights
    outputs = model(**inputs, output_attentions=True)
    
    # Get attention weights from all layers and heads
    attention_weights = outputs.attentions  # Tuple of tensors [batch, heads, seq_len, seq_len]
    
    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Analyze patterns
    results = {
        "tokens": tokens,
        "attention_weights": attention_weights,
        "layer_patterns": [],
        "key_relationships": []
    }
    
    # Analyze each layer
    for layer_idx, layer_attention in enumerate(attention_weights):
        # Average across attention heads
        avg_attention = layer_attention[0].mean(dim=0)
        
        # Find strongest connections
        values, indices = avg_attention.max(dim=1)
        
        # Record key relationships
        for i, (value, idx) in enumerate(zip(values, indices)):
            if value > 0.2:  # Only significant connections
                results["key_relationships"].append({
                    "layer": layer_idx,
                    "from_token": tokens[i],
                    "to_token": tokens[idx],
                    "strength": value.item()
                })
        
        # Record layer pattern
        results["layer_patterns"].append({
            "layer": layer_idx,
            "avg_attention": avg_attention.detach().numpy(),
            "entropy": -(avg_attention * torch.log(avg_attention + 1e-10)).sum(dim=1).mean().item()
        })
    
    return results
```

### 3.6.3 Key Findings

Our analysis of transformer performance on UTTA educational content revealed:

1. **Terminology recognition**: Later transformer layers develop specialized attention patterns for educational terminology
2. **Contextual understanding**: Attention heads specialize in different aspects of educational content
3. **Hierarchical processing**: Early layers focus on syntax, while deeper layers capture semantic relationships
4. **Cross-reference handling**: Transformers effectively handle references across educational materials

### 3.6.4 Performance Optimization

Based on our findings, we optimized transformer performance for educational content:

```python
def optimize_transformer_for_education(model_name, tokenizer_name):
    """
    Optimize transformer for educational content
    
    Args:
        model_name (str): Base model name
        tokenizer_name (str): Tokenizer name
        
    Returns:
        tuple: Optimized model and tokenizer
    """
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Add education-specific tokens
    education_tokens = [
        "[LEARNING_OBJECTIVE]",
        "[ASSESSMENT]",
        "[STRATEGY]",
        "[EXAMPLE]",
        "[CITATION]"
    ]
    tokenizer.add_tokens(education_tokens)
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Optimize attention patterns
    # In practice, this would involve fine-tuning on educational data
    
    return model, tokenizer
```

## 3.7 Key Takeaways

- Attention mechanisms allow models to focus on relevant parts of the input
- Self-attention enables direct connections between any positions in a sequence
- Multi-head attention lets the model attend to different representation subspaces
- Positional encoding provides sequence order information to the transformer
- The encoder-decoder architecture enables flexible sequence-to-sequence processing
- Transformers excel at capturing long-range dependencies in text

## 3.8 Chapter Project: Educational Content Analyzer

For this chapter's project, you'll build a transformer-based educational content analyzer:

1. Implement a simplified transformer encoder
2. Process educational texts and visualize attention patterns
3. Identify key concepts and relationships in educational materials
4. Compare attention patterns across different types of educational content
5. Optimize the model for educational domain-specific tasks

## References

- Vaswani, A., et al. (2017). *Attention Is All You Need*
- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- Alammar, J. (2018). *The Illustrated Transformer*
- Rush, A. (2018). *The Annotated Transformer*

## Further Reading

- [Chapter 2: Natural Language Processing Fundamentals](NLP-Fundamentals)
- [Chapter 4: Embeddings and Vector Representations](Knowledge-Base-Structure)
- [Chapter 5: Large Language Models](Knowledge-LLM-Integration)
- [Chapter 8: Autonomous Agents](Autonomous-Agents) 