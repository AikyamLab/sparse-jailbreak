class GemmaScopeInterventionModel(torch.nn.Module):
    """
    Model wrapper that applies Gemma Scope SAE reconstruction during all forward passes.
    """
    def __init__(
        self,
        model_name: str,
        sae_release: str,
        sae_id: str,
        device: str = "cuda",
        dtype = torch.float32,
        detach_reconstruction: bool = False,  # False = grads flow through SAE
        record_stats: bool = True,
    ):
        super().__init__()
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=dtype,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load Gemma Scope SAE
        self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device
        )
        
        # Extract layer information from sae_id
        self.sae_layer_idx = int(sae_id.split('/')[0].split('_')[1])
        
        # Set device and dtype
        self.dtype = dtype
        self.device = torch.device(device)
        
        # SAE options
        self.detach_reconstruction = detach_reconstruction
        self.record_stats = record_stats
        
        # Tracking
        self.sae_hook_calls = 0
        self.last_recon_norm = None
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def _get_sae_module(self):
        if 'gemma' in self.model.config.model_type.lower():
            return self.model.model.layers[self.sae_layer_idx]
        elif hasattr(self.model.model, 'layers'):
            return self.model.model.layers[self.sae_layer_idx]
        else:
            raise ValueError("Unsupported model architecture")
    
    def _make_sae_hook(self):
        def hook(module, input, output):
            # Extract activations
            if hasattr(output, 'last_hidden_state'):
                activations = output.last_hidden_state
            elif isinstance(output, tuple) and len(output) > 0:
                activations = output[0]
            else:
                activations = output
            
            # Apply SAE (no no_grad for gradients during opt)
            batch_size, seq_len, hidden_size = activations.shape
            acts_reshaped = activations.reshape(-1, hidden_size)
            sae_output = self.sae(acts_reshaped)
            reconstructed = sae_output.sae_out.reshape(batch_size, seq_len, hidden_size)
            
            if self.detach_reconstruction:
                reconstructed = reconstructed.detach()
            
            if self.record_stats:
                self.sae_hook_calls += 1
                self.last_recon_norm = torch.norm(activations - reconstructed).detach().item()
            
            # Return reconstructed
            if hasattr(output, 'last_hidden_state'):
                output.last_hidden_state = reconstructed
                return output
            elif isinstance(output, tuple):
                return (reconstructed,) + output[1:]
            else:
                return reconstructed
        
        return hook
    
    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        module = self._get_sae_module()
        handle = module.register_forward_hook(self._make_sae_hook())
        try:
            return self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        finally:
            handle.remove()
    
    def generate(self, input_ids=None, **kwargs):
        module = self._get_sae_module()
        handle = module.register_forward_hook(self._make_sae_hook())
        try:
            return self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            handle.remove()
    
    def test_sae_reconstruction(self, prompt="Hello, how are you?"):
        """Test if SAE reconstruction is working properly"""
        print("Testing Gemma Scope SAE reconstruction...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        # Reset tracking
        self.sae_hook_calls = 0
        self.last_recon_norm = None
        
        # Run without SAE (direct inner model)
        with torch.no_grad():
            output_without_sae = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Run with SAE (via wrapper forward)
        with torch.no_grad():
            output_with_sae = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compare logits
        logits_without_sae = output_without_sae.logits[:, -1, :]
        logits_with_sae = output_with_sae.logits[:, -1, :]
        
        # Get top predictions
        top_tokens_without_sae = torch.topk(logits_without_sae, 5, dim=-1).indices[0]
        top_tokens_with_sae = torch.topk(logits_with_sae, 5, dim=-1).indices[0]
        
        # Calculate similarity
        probs_without_sae = torch.softmax(logits_without_sae, dim=-1)[0]
        probs_with_sae = torch.softmax(logits_with_sae, dim=-1)[0]
        
        similarity = torch.nn.functional.cosine_similarity(
            probs_with_sae.unsqueeze(0), 
            probs_without_sae.unsqueeze(0)
        ).item()
        
        # Store metrics
        self.reconstruction_metrics = {
            'cosine_similarity': similarity,
            'sae_hook_calls': self.sae_hook_calls,
            'last_recon_norm': self.last_recon_norm,
            'tokens_match': top_tokens_with_sae[0].item() == top_tokens_without_sae[0].item(),
            'sae_layer': self.sae_layer_idx,
            'sae_width': self.sae.cfg.d_sae,
            'sae_sparsity': self.sparsity
        }
        
        self.reconstruction_verified = (
            similarity < 0.9999 and 
            self.sae_hook_calls > 0 and 
            self.last_recon_norm > 0
        )
        
        print(f"Gemma Scope SAE Reconstruction Test Results:")
        print(f"- SAE: {self.sae.cfg.metadata.neuronpedia_id if hasattr(self.sae.cfg, 'metadata') else 'Unknown'}")
        print(f"- Layer: {self.sae_layer_idx}, Width: {self.sae.cfg.d_sae}")
        print(f"- Cosine similarity: {similarity:.6f}")
        print(f"- Same top token: {self.reconstruction_metrics['tokens_match']}")
        print(f"- SAE hook calls: {self.sae_hook_calls}")
        print(f"- Last recon norm: {self.last_recon_norm}")
        print(f"- Reconstruction verified: {self.reconstruction_verified}")
        
        return self.reconstruction_verified