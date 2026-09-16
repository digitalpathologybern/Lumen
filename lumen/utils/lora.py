from peft import LoraConfig, get_peft_model

txt_attention_modules = {
        'biomedBERT': ['query', 'key', 'value', 'attention.output.dense'],
        'bioGPT': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        "MedBERT" : ['query', 'key', 'value', 'attention.output.dense'],
        "ClinicalBERT" : ['q_lin', 'k_lin', 'v_lin', 'out_lin'],
        "clip_vit_b16": ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        "clip_vit_b32": ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        "clip_vit_l14": ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        "plip": ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        "quilt_b32": ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        "quilt_b16": ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
    }

vit_attention_modules = {
        'Virchow2' : ['attn.qkv', 'attn.proj'],
        "clip_vit_b16": ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        "clip_vit_b32": ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        "clip_vit_l14": ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        "plip": ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        "quilt_b32": ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        "quilt_b16": ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        "pathgen_l14": ['q_proj', 'k_proj', 'v_proj', 'out_proj']
}

def wrap_with_lora(encoder, r, dropout, backbone, enc_type):
    if enc_type == 'text':
        txt_attn_modules = txt_attention_modules[backbone]
        
        txt_config = LoraConfig(
            r=r,
            lora_alpha=r*2,
            target_modules=txt_attn_modules,
            lora_dropout=dropout,
            bias='none',
            task_type = 'FEATURE_EXTRACTION'
        )
        
        enc = get_peft_model(encoder, txt_config)
    
    elif enc_type == 'image':
        vit_attn_modules = vit_attention_modules[backbone]  
        vit_config = LoraConfig(
            r=r,
            lora_alpha=r*2,
            target_modules=vit_attn_modules,
            lora_dropout=dropout,
            bias='none',
            task_type = 'FEATURE_EXTRACTION'
        )
        
        enc = get_peft_model(encoder, vit_config)
    
    else:
        raise ValueError("Encoder type must either be 'text' or 'image'")
    
    return enc