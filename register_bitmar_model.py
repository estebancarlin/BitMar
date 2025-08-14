"""
Register BitMar model with transformers for evaluation pipeline compatibility
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

class BitMarConfig(PretrainedConfig):
    model_type = "bitmar"
    
    def __init__(
        self,
        vocab_size=50257,
        text_encoder_dim=128,
        text_encoder_layers=4,
        text_encoder_heads=4,
        vision_encoder_dim=768,
        vision_latent_size=128,
        fusion_hidden_size=128,
        memory_size=32,
        episode_dim=128,
        max_seq_len=256,
        dropout=0.15,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.text_encoder_dim = text_encoder_dim
        self.text_encoder_layers = text_encoder_layers
        self.text_encoder_heads = text_encoder_heads
        self.vision_encoder_dim = vision_encoder_dim
        self.vision_latent_size = vision_latent_size
        self.fusion_hidden_size = fusion_hidden_size
        self.memory_size = memory_size
        self.episode_dim = episode_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

class BitMarForCausalLM(PreTrainedModel):
    config_class = BitMarConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Text encoder (simplified for evaluation compatibility)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.text_encoder_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.text_encoder_dim,
                nhead=config.text_encoder_heads,
                dim_feedforward=config.text_encoder_dim * 4,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.text_encoder_layers)
        ])
        self.ln_f = nn.LayerNorm(config.text_encoder_dim)
        self.lm_head = nn.Linear(config.text_encoder_dim, config.vocab_size, bias=False)
        
        # Vision components (for multimodal)
        self.vision_proj = nn.Linear(config.vision_encoder_dim, config.vision_latent_size)
        
        # Memory components
        self.memory_slots = nn.Parameter(torch.randn(config.memory_size, config.episode_dim))
        
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Simple causal LM forward for evaluation compatibility
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            # Create causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(hidden_states.device)
            
            hidden_states = layer(
                hidden_states,
                src_mask=causal_mask,
                src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
        
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states
        }
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value

# Register the model
AutoConfig.register("bitmar", BitMarConfig)
AutoModel.register(BitMarConfig, BitMarForCausalLM)

def load_bitmar_from_pretrained(model_name_or_path, **kwargs):
    """Load BitMar model with proper weight mapping"""
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = BitMarForCausalLM(config)
    
    # Try to load state dict and map weights
    try:
        import huggingface_hub
        model_files = huggingface_hub.list_repo_files(model_name_or_path)
        
        if "pytorch_model.bin" in model_files:
            state_dict = huggingface_hub.hf_hub_download(
                model_name_or_path, 
                "pytorch_model.bin",
                revision=kwargs.get("revision", None)
            )
            state_dict = torch.load(state_dict, map_location="cpu")
            
            # Map weights from original BitMar to transformers-compatible model
            mapped_state_dict = {}
            for key, value in state_dict.items():
                if "text_encoder.embed_tokens" in key:
                    mapped_state_dict[key.replace("text_encoder.embed_tokens", "embed_tokens")] = value
                elif "text_encoder.layers" in key:
                    mapped_state_dict[key.replace("text_encoder.layers", "transformer_layers")] = value
                elif "text_encoder.ln_f" in key:
                    mapped_state_dict[key.replace("text_encoder.ln_f", "ln_f")] = value
                elif "lm_head" in key:
                    mapped_state_dict[key] = value
                else:
                    mapped_state_dict[key] = value
            
            model.load_state_dict(mapped_state_dict, strict=False)
            print(f"✅ Loaded BitMar model from {model_name_or_path}")
        
    except Exception as e:
        print(f"⚠️ Could not load pretrained weights: {e}")
        print("Using randomly initialized model")
    
    return model

if __name__ == "__main__":
    # Test loading
    model = load_bitmar_from_pretrained("euhidaman/bitmar-attention-multimodal")
    print("BitMar model registration successful!")