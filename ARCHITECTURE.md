# CSM Architecture Guide

This guide explains the Conversational Speech Model (CSM) architecture for developers new to audio models and neural networks. By the end, you'll understand how CSM generates natural-sounding conversational speech from text.

## Table of Contents
- [Key Concepts](#key-concepts)
- [Architecture Overview](#architecture-overview)
- [Data Flow](#data-flow)
- [What Makes CSM Special](#what-makes-csm-special)
- [File Structure Guide](#file-structure-guide)
- [Glossary](#glossary)
- [Further Reading](#further-reading)

---

## Key Concepts

### Transformers: The Foundation

**What are they?**  
Transformers are neural network architectures that excel at understanding sequences (like sentences or audio). CSM is built on Llama 3.2, a powerful transformer model originally designed for language.

**How do they work?**  
Instead of processing words one-by-one like older models, transformers use an **attention mechanism** that looks at all words (or audio frames) simultaneously. Think of it like reading a sentence and being able to instantly consider how every word relates to every other word.

```
Example: "The cat sat on the mat"
Traditional: cat → sat → on (processes sequentially)
Transformer: Sees all words at once and understands relationships
              (cat relates to sat, mat relates to on, etc.)
```

**In CSM:**  
CSM uses two transformer models:
- **Backbone (Llama 3.2 1B)**: Understands the conversation context and decides "what to say"
- **Decoder (Llama 3.2 100M)**: Generates the audio details, deciding "how it sounds"

### RVQ (Residual Vector Quantization): Representing Audio as Codes

**The Challenge:**  
Raw audio waveforms are continuous signals with thousands of samples per second. Neural networks work better with discrete tokens (like words in text).

**The Solution: RVQ**  
RVQ converts audio into a sequence of discrete "codes" or tokens. Think of it like converting a smooth curve into a series of points that can be stored and transmitted efficiently.

```
Raw Audio Wave:     ∿∿∿∿∿∿∿∿∿∿∿∿∿∿
                        ↓ [RVQ Encoding]
Discrete Codes:     [42, 17, 91, 53, ...]
```

**How it works:**  
RVQ learns a "vocabulary" of audio patterns during training. When encoding audio, it finds the closest matching patterns and represents them as integer codes. During decoding, these codes are converted back to audio.

**Why "Residual"?**  
RVQ works in layers. The first layer captures the main audio structure. Each subsequent layer captures the "residual" (what's left over) in finer detail. This hierarchical approach allows precise audio reconstruction.

### Codebooks: The Audio Vocabulary

**What are they?**  
A codebook is like a dictionary of audio patterns. Each entry (called a "code vector") represents a small piece of audio information. CSM uses **32 codebooks**, each capturing different aspects of the audio.

```
Codebook 0:  Captures high-level prosody (pitch contours, rhythm)
Codebook 1:  Refines pronunciation details
Codebook 2:  Adds vocal texture
  ...
Codebook 31: Captures finest details (breathiness, subtle harmonics)
```

**In CSM:**  
- Each codebook has 2,048 possible codes (vocabulary size)
- Audio is encoded as 32 parallel sequences, one per codebook
- The model generates these 32 sequences hierarchically (codebook 0 first, then 1-31)

### Tokenization: Converting to Model Input

**Text Tokenization:**  
Text is split into tokens (subword units) using the Llama tokenizer. For example:
```
"Hello from Sesame" → [15339, 505, 64027, 373]
```

**Audio Tokenization:**  
Audio is encoded using the Mimi codec into 32 codebook sequences:
```
Audio (24kHz) → Mimi Encoder → 32 sequences of codes
```

**Combined Representation:**  
CSM processes both text and audio tokens together. Each timestep has 33 dimensions:
- 32 dimensions for audio codebooks
- 1 dimension for text tokens

```python
# From generator.py lines 65-68 (text) and 88-91 (audio)
# Text tokenization:
text_frame = torch.zeros(len(text_tokens), 33).long()
text_frame[:, -1] = torch.tensor(text_tokens)  # Text in last column

# Audio tokenization:
audio_frame = torch.zeros(audio_tokens.size(1), 33).long()
audio_frame[:, :-1] = audio_tokens.transpose(0, 1)  # Audio in first 32 columns
```

### KV Caching: Efficient Generation

**The Problem:**  
Transformers need to "remember" all previous tokens when generating new ones. Without optimization, this means recomputing everything from scratch at each step—very slow!

**The Solution: KV Caching**  
The model caches **Key** and **Value** matrices from the attention mechanism. These represent the "memory" of what's been processed so far.

```
Step 1: Process "Hello" → Save K,V
Step 2: Process "from" → Use cached K,V, only compute new token
Step 3: Process "Sesame" → Use cached K,V, only compute new token
```

**In CSM:**
```python
# From models.py lines 120-127
def setup_caches(self, max_batch_size: int):
    """Setup KV caches for efficient inference."""
    self.backbone.setup_caches(max_batch_size, dtype)
    self.decoder.setup_caches(max_batch_size, dtype, 
                              decoder_max_seq_len=self.config.audio_num_codebooks)
```

- `setup_caches()`: Allocates memory for caching before generation
- `reset_caches()`: Clears the cache between different generations
- The decoder cache is reset **every frame** because it generates all 32 codebooks fresh

### Embeddings: From Tokens to Vectors

**What are embeddings?**  
Embeddings convert discrete tokens (integers) into dense vectors (lists of real numbers) that capture meaning and relationships.

```
Token 42 → [0.23, -0.45, 0.67, ..., 0.12]  (2048 dimensions for backbone)
```

**Why?**  
Neural networks work with continuous numbers, not discrete symbols. Embeddings allow the model to learn that similar sounds/words have similar vector representations.

**In CSM:**
```python
# From models.py line 113-114
self.text_embeddings = nn.Embedding(text_vocab_size, backbone_dim)
self.audio_embeddings = nn.Embedding(audio_vocab_size * audio_num_codebooks, backbone_dim)
```

- Text tokens are embedded into 2048-dimensional vectors
- Audio codes are embedded separately for each codebook position
- These embeddings are learned during training

---

## Architecture Overview

CSM uses a **two-stage transformer architecture** to generate speech. Here's how the pieces fit together:

```
                    ┌─────────────────────────────────────┐
                    │   INPUT: Text + Context Audio      │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │  TEXT & AUDIO EMBEDDINGS            │
                    │  (Convert tokens to vectors)        │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │  BACKBONE TRANSFORMER               │
                    │  (Llama 3.2 1B)                     │
                    │  • 16 layers                        │
                    │  • 32 attention heads               │
                    │  • 2048 embedding dim               │
                    │  → Processes full context           │
                    │  → Decides "what to say"            │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │  CODEBOOK 0 HEAD                    │
                    │  (Linear layer: 2048 → 2048)        │
                    │  → Generates first codebook         │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │  PROJECTION LAYER                   │
                    │  (Linear layer: 2048 → 1024)        │
                    │  → Bridges backbone to decoder      │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │  DECODER TRANSFORMER                │
                    │  (Llama 3.2 100M)                   │
                    │  • 4 layers                         │
                    │  • 8 attention heads                │
                    │  • 1024 embedding dim               │
                    │  → Autoregressively generates       │
                    │    codebooks 1-31                   │
                    │  → Decides "how it sounds"          │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │  AUDIO HEAD (31 output heads)       │
                    │  (31x matrices: 1024 → 2048 each)   │
                    │  → Generates codebooks 1-31         │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │  OUTPUT: 32 Codebook Sequences      │
                    │  → Decoded to audio by Mimi         │
                    └─────────────────────────────────────┘
```

### Component Details

#### 1. Backbone Transformer (Llama 3.2 1B)

**Purpose:** Process the entire conversation context (text + audio history) and generate high-level representations.

**Configuration:**
```python
# From models.py line 10-23
vocab_size=128_256      # Llama's text vocabulary
num_layers=16           # Depth of the network
num_heads=32            # Parallel attention computations
num_kv_heads=8          # Grouped-query attention for efficiency
embed_dim=2048          # Size of hidden representations
max_seq_len=2048        # Maximum context length
```

**Input:** Combined text and audio tokens (shape: `[batch, seq_len, 33]`)  
**Output:** Hidden states (shape: `[batch, seq_len, 2048]`)

**Why 1B parameters?**  
The backbone needs strong language understanding to capture conversational context, speaker identity, and prosody. A larger model (1 billion parameters) provides this capacity.

#### 2. Decoder Transformer (Llama 3.2 100M)

**Purpose:** Generate detailed audio codes autoregressively, one codebook at a time.

**Configuration:**
```python
# From models.py line 26-39
num_layers=4            # Smaller/faster than backbone
num_heads=8
embed_dim=1024          # Half the backbone dimension
max_seq_len=2048
```

**Why smaller?**  
The decoder's job is more focused: given the backbone's decision about "what to say," fill in the audio details. This doesn't require as much capacity, so a smaller, faster model works well.

**Autoregressive Generation:**  
The decoder generates codebooks sequentially: c1 depends on c0, c2 depends on c0 and c1, etc. This allows fine-grained control over audio quality.

```python
# From models.py lines 171-182
for i in range(1, self.config.audio_num_codebooks):  # Generate codebooks 1-31
    curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
    decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, 
                             mask=curr_decoder_mask).to(dtype=dtype)
    ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
    ci_sample = sample_topk(ci_logits, topk, temperature)
    ci_embed = self._embed_audio(i, ci_sample)
    curr_h = ci_embed  # Feed into next codebook
    curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
    curr_pos = curr_pos[:, -1:] + 1
```

#### 3. Projection Layer

**Purpose:** Bridge the backbone's 2048-dimensional space to the decoder's 1024-dimensional space.

```python
# From models.py line 116
self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
```

This is a simple linear transformation that reduces dimensionality while preserving important information for the decoder.

#### 4. Output Heads

**Codebook 0 Head:**
```python
# From models.py line 117
self.codebook0_head = nn.Linear(backbone_dim, audio_vocab_size, bias=False)
```
Directly generates the first codebook from the backbone's output. This captures high-level prosody and structure.

**Audio Head (Codebooks 1-31):**
```python
# From models.py line 118
self.audio_head = nn.Parameter(torch.empty(31, decoder_dim, audio_vocab_size))
```
31 separate transformation matrices, one for each remaining codebook. Each matrix converts decoder hidden states to vocabulary logits.

---

## Data Flow

Let's walk through what happens when you call `generator.generate()`:

### Step 1: Text Tokenization with Speaker ID

```python
# From generator.py line 64
text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
```

**What happens:**
- The speaker ID is prepended: `"[0]Hello from Sesame"`
- Llama's tokenizer converts to tokens: `[128000, 58, 15339, 505, 64027, 373, 128001]`
- These are placed in the text column (dimension 33) of the input tensor

**Why speaker IDs?**  
They allow the model to learn different voice characteristics for different speakers in a conversation.

### Step 2: Context Encoding

```python
# From generator.py line 122-125
for segment in context:
    segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
    tokens.append(segment_tokens)
    tokens_mask.append(segment_tokens_mask)
```

**What happens:**
- Previous utterances are encoded as text+audio pairs
- Each `Segment` contains: `speaker`, `text`, and `audio`
- Text is tokenized, audio is encoded through Mimi into 32 codebook sequences
- These form the "conversational memory" for the model

**Example context structure:**
```
[Speaker 0 Text] [Speaker 0 Audio] [Speaker 1 Text] [Speaker 1 Audio] [Speaker 0 Text (current)]
```

### Step 3: Audio Encoding via Mimi

```python
# From generator.py line 83
audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
```

**What happens:**
- Raw audio waveform (24kHz sample rate) → Mimi encoder
- Mimi uses RVQ to produce 32 codebook sequences
- Shape: `[32, num_frames]` where each frame ≈ 80ms of audio
- These tokens are placed in the first 32 columns of the input tensor

**Why Mimi?**  
Mimi is a high-quality neural codec that produces efficient, high-fidelity audio representations. It's trained to preserve perceptual quality while compressing audio into discrete codes.

### Step 4: Backbone Processing

```python
# From models.py line 155-158
embeds = self._embed_tokens(tokens)
masked_embeds = embeds * tokens_mask.unsqueeze(-1)
h = masked_embeds.sum(dim=2)
h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
```

**What happens:**
1. Tokens are converted to embeddings (text and audio separately)
2. The mask ensures only valid positions contribute (text OR audio, never both)
3. Embeddings are summed across the 33 dimensions to create single vectors
4. The backbone transformer processes the full sequence

**Output:** Hidden states representing the model's understanding of the conversation

### Step 5: Autoregressive Audio Generation

The model generates audio frame-by-frame (each frame ≈ 80ms). For each frame:

**5a. Generate Codebook 0 (from backbone):**
```python
# From models.py lines 160-162
last_h = h[:, -1, :]  # Last position's hidden state
c0_logits = self.codebook0_head(last_h)
c0_sample = sample_topk(c0_logits, topk, temperature)
```

**5b. Generate Codebooks 1-31 (from decoder):**
```python
# From models.py lines 170-182
self.decoder.reset_caches()  # Fresh start for each frame
for i in range(1, self.config.audio_num_codebooks):  # 1 to 31
    curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
    decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, 
                             mask=curr_decoder_mask).to(dtype=dtype)
    ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
    ci_sample = sample_topk(ci_logits, topk, temperature)
    ci_embed = self._embed_audio(i, ci_sample)
    curr_h = ci_embed  # Feed into next codebook
    curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
    curr_pos = curr_pos[:, -1:] + 1
```

**Key points:**
- Decoder caches are reset every frame (line 170)
- Each codebook is generated conditioned on all previous codebooks
- This hierarchical generation allows fine control over audio details

**5c. Check for end-of-sequence:**
```python
# From generator.py line 148-149
if torch.all(sample == 0):
    break  # All zeros = EOS token
```

### Step 6: Decode to Waveform

```python
# From generator.py line 159
audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0))
```

**What happens:**
- Stack all generated frames: shape `[num_frames, batch, 32]`
- Permute to Mimi's expected format: `[batch, 32, num_frames]`
- Mimi decoder converts codebook sequences back to raw audio waveform
- Output: 24kHz audio tensor

### Step 7: Apply Watermark

```python
# From generator.py line 165
audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
```

**What happens:**
- Audio is upsampled to 44.1kHz for watermarking
- An imperceptible watermark is embedded using silentcipher
- Audio is resampled back to 24kHz
- The watermark identifies the audio as AI-generated

**Why watermarking?**  
Ensures transparency and helps prevent misuse. The watermark is nearly inaudible but can be detected to verify if audio was generated by CSM.

---

## What Makes CSM Special

### 1. Conversational Context via Segments

Unlike models that generate speech in isolation, CSM maintains **multi-turn dialogue context** using the `Segment` dataclass:

```python
# From generator.py line 15-19
@dataclass
class Segment:
    speaker: int
    text: str
    audio: torch.Tensor  # (num_samples,), sample_rate = 24_000
```

**Benefits:**
- Natural turn-taking in conversations
- Consistent speaker identities across utterances
- Context-aware prosody (e.g., responding to questions with appropriate intonation)

**Example usage:**
```python
# From run_csm.py line 97-104
context = [prompt_a, prompt_b]  # Initial voice samples
for utterance in conversation:
    audio = generator.generate(
        text=utterance['text'],
        speaker=utterance['speaker_id'],
        context=context + generated_segments,  # Full conversation history
    )
    generated_segments.append(Segment(...))
```

### 2. Multi-Speaker Support

CSM can model different voices using speaker IDs:

```python
# Text is prefixed with speaker: "[0]Hello" or "[1]Hey there"
text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
```

**How it works:**
- Each speaker ID allows the model to learn distinct voice characteristics
- In training, the model learns to associate IDs with prosody, pitch, and speaking style
- At inference, you can switch speakers mid-conversation

**Practical application:**
- Generate dialogues between multiple characters
- Create consistent character voices across long conversations
- Voice cloning (with appropriate fine-tuning and consent)

### 3. Hierarchical Generation

The two-stage architecture provides a clean separation of concerns:

**Backbone (1B params):** "What to say"
- Understands conversation context
- Captures semantic meaning
- Determines high-level prosody patterns
- Generates the first codebook (main structure)

**Decoder (100M params):** "How it sounds"
- Fills in acoustic details
- Generates 31 additional codebooks
- Controls fine-grained audio quality
- Fast generation due to smaller size

**Why this matters:**
- **Efficiency:** Decoder is small, so generating 31 codebooks is fast
- **Quality:** Backbone's language understanding informs high-level decisions
- **Modularity:** Could potentially swap decoders for different audio qualities

### 4. Based on Proven LLM Architecture

CSM leverages **Llama 3.2**, a state-of-the-art language model:

**Advantages:**
- **Strong language understanding:** Llama's pre-training on text helps with conversational coherence
- **Efficient attention:** Grouped-query attention (GQA) reduces memory usage
- **Well-studied architecture:** Benefits from extensive research on Llama models

**From Llama to Speech:**
```python
# From models.py line 48-52
def _prepare_transformer(model):
    model.tok_embeddings = nn.Identity()  # Replace with custom embeddings
    model.output = nn.Identity()          # Remove language modeling head
    return model, embed_dim
```

The Llama transformer core is repurposed: token embeddings and output heads are replaced with CSM's text+audio embeddings and codebook generation heads.

---

## File Structure Guide

### `models.py` - Model Architecture Definition

**What it contains:**
- `llama3_2_1B()`, `llama3_2_100M()`: Functions to create transformer instances
- `Model` class: The main CSM model combining backbone + decoder
- `ModelArgs` dataclass: Configuration (vocab sizes, codebook count)
- `generate_frame()`: Core generation logic for one audio frame
- Sampling utilities: `sample_topk()`, causal mask helpers

**Key sections:**
- **Lines 10-45:** Transformer configurations
- **Lines 106-119:** Model initialization (embeddings, projection, heads)
- **Lines 120-130:** KV cache setup
- **Lines 132-184:** Frame generation logic

**When to modify:**
- Changing model dimensions or layer counts
- Adding new output heads or embeddings
- Adjusting generation algorithm (sampling, temperature)

### `generator.py` - Inference & Tokenization

**What it contains:**
- `Segment` dataclass: Represents a conversation turn
- `Generator` class: High-level interface for speech generation
- `load_llama3_tokenizer()`: Sets up text tokenization
- `load_csm_1b()`: Convenience function to load the model

**Key sections:**
- **Lines 60-73:** Text tokenization with speaker IDs
- **Lines 75-96:** Audio tokenization with Mimi
- **Lines 109-168:** Main generation loop (context → frame-by-frame generation → watermarking)

**When to modify:**
- Changing how context is processed
- Adjusting generation parameters (temperature, max length)
- Adding pre/post-processing steps

### `run_csm.py` - Example Usage Script

**What it contains:**
- Example conversation generation
- Speaker prompt loading
- Audio saving utilities

**Key sections:**
- **Lines 21-44:** Speaker prompt definitions
- **Lines 46-57:** Prompt loading utilities
- **Lines 86-114:** Conversation generation loop

**When to use:**
- As a starting point for your own applications
- To test the model on example conversations
- To understand end-to-end usage

### `watermarking.py` - AI Audio Identification

**What it contains:**
- `watermark()`: Embeds imperceptible watermark in audio
- `verify()`: Checks if audio is watermarked
- `CSM_1B_GH_WATERMARK`: Public key for this release

**Key sections:**
- **Lines 28-40:** Watermark embedding (uses silentcipher)
- **Lines 43-59:** Watermark verification
- **Lines 62-70:** CLI utility for checking files

**When to use:**
- To verify if audio was generated by CSM
- To understand watermarking integration
- **Keep watermarking enabled** to ensure responsible AI usage

---

## Glossary

**Attention:** Mechanism allowing models to focus on relevant parts of input when processing each token.

**Autoregressive:** Generating one element at a time, where each element depends on previous ones.

**Backbone:** The main transformer (Llama 3.2 1B) that processes conversation context.

**Causal Mask:** Ensures the model only attends to previous tokens, not future ones (prevents "cheating").

**Codebook:** A learned dictionary of audio patterns used in quantization.

**Decoder:** The smaller transformer (Llama 3.2 100M) that generates detailed audio codebooks.

**Embedding:** Continuous vector representation of discrete tokens (text or audio codes).

**EOS (End of Sequence):** Special token indicating generation is complete (all zeros in CSM).

**Frame:** A unit of audio in CSM, approximately 80ms long (1920 samples at 24kHz).

**GQA (Grouped-Query Attention):** Efficient attention mechanism using fewer key/value heads than query heads.

**Hidden State:** Internal representation computed by transformer layers.

**KV Cache:** Stores Key and Value matrices from attention to avoid recomputation.

**Logits:** Raw, unnormalized scores output by the model before sampling/softmax.

**Mimi:** Neural audio codec used to encode/decode audio as RVQ codes.

**Projection Layer:** Linear transformation bridging backbone (2048D) to decoder (1024D).

**Prosody:** Rhythm, stress, and intonation patterns in speech.

**RVQ (Residual Vector Quantization):** Hierarchical method for converting audio to discrete codes.

**Sample Rate:** Number of audio samples per second (CSM uses 24,000 Hz).

**Sampling:** Selecting tokens from probability distributions (CSM uses top-k sampling).

**Segment:** A conversation turn containing speaker ID, text, and audio.

**Token:** A discrete unit of input/output (text subword or audio code).

**Top-k Sampling:** Selecting from the k most likely next tokens (increases diversity).

**Transformer:** Neural architecture using attention to process sequences in parallel.

**Vocab Size:** Number of unique tokens (128,256 for text, 2,048 per audio codebook).

**Watermark:** Imperceptible signal embedded in audio to identify it as AI-generated.

---

## Further Reading

### Papers & Research

**Transformers:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Llama 3.2: Open Foundation Models](https://ai.meta.com/blog/llama-3-2/) - Llama 3.2 announcement

**Audio Codecs & RVQ:**
- [Moshi: A Speech-Text Foundation Model](https://kyutai.org/Moshi.pdf) - Mimi codec details
- [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312) - RVQ in audio
- [Neural Audio Codecs: A Guide](https://arxiv.org/abs/2310.01251) - Comprehensive overview

**Speech Generation:**
- [AudioLM: A Language Modeling Approach to Audio Generation](https://arxiv.org/abs/2209.03143) - Hierarchical audio generation
- [VALL-E: Neural Codec Language Models](https://arxiv.org/abs/2301.02111) - Text-to-speech with codebooks

**Efficient Transformers:**
- [GQA: Training Generalized Multi-Query Attention](https://arxiv.org/abs/2305.13245) - Efficient attention
- [Fast Transformer Decoding](https://arxiv.org/abs/2211.05102) - KV caching details

### Code & Documentation

- [CSM GitHub Repository](https://github.com/SesameAILabs/csm) - Official codebase
- [CSM Hugging Face Model](https://huggingface.co/sesame/csm-1b) - Pre-trained weights
- [TorchTune Documentation](https://pytorch.org/torchtune/) - Llama implementation library
- [Mimi Documentation](https://github.com/kyutai-labs/moshi) - Audio codec details

### Sesame AI Research

- [Crossing the Uncanny Valley of Voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) - CSM blog post
- [Interactive Voice Demo](https://www.sesame.com/voicedemo) - Try CSM in action
- [Hugging Face Space](https://huggingface.co/spaces/sesame/csm-1b) - Online demo

---

## Contributing

Found an error or have suggestions for improving this guide? Please open an issue or pull request on the [CSM GitHub repository](https://github.com/SesameAILabs/csm).

---

**Last updated:** 2025-01-12  
**CSM Version:** 1B  
**License:** Apache 2.0
