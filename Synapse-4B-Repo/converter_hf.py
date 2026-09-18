import torch
import json
import os

print("Carregando o checkpoint bruto do Kaggle (Passo 4100)...")
ckpt = torch.load(r"C:\Users\Fernando\Downloads\synapse_4b_latest.pt", map_location="cpu")
state_dict = ckpt["model_state_dict"]
config = ckpt["config"]

out_dir = r"C:\Users\Fernando\Downloads\Synapse-4B-HF"
print(f"Criando pasta oficial {out_dir}...")
os.makedirs(out_dir, exist_ok=True)

hf_dict = {}

print("Mapeando a estrutura neural para o formato Universal (Llama)...")
hf_dict["model.embed_tokens.weight"] = state_dict["token_embedding.weight"]
hf_dict["lm_head.weight"] = state_dict["lm_head.weight"]
hf_dict["model.norm.weight"] = state_dict["norm.weight"]

for i in range(config["n_layer"]):
    hf_dict[f"model.layers.{i}.input_layernorm.weight"] = state_dict[f"blocks.{i}.attn_norm.weight"]
    hf_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = state_dict[f"blocks.{i}.mlp_norm.weight"]
    
    hf_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = state_dict[f"blocks.{i}.attention.q_proj.weight"]
    hf_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = state_dict[f"blocks.{i}.attention.k_proj.weight"]
    hf_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = state_dict[f"blocks.{i}.attention.v_proj.weight"]
    hf_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = state_dict[f"blocks.{i}.attention.o_proj.weight"]
    
    hf_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = state_dict[f"blocks.{i}.mlp.gate_proj.weight"]
    hf_dict[f"model.layers.{i}.mlp.up_proj.weight"] = state_dict[f"blocks.{i}.mlp.up_proj.weight"]
    hf_dict[f"model.layers.{i}.mlp.down_proj.weight"] = state_dict[f"blocks.{i}.mlp.down_proj.weight"]

print("Salvando os Pesos oficiais (pytorch_model.bin)...")
torch.save(hf_dict, os.path.join(out_dir, "pytorch_model.bin"))

hf_config = {
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": config["n_embd"],
  "intermediate_size": config["intermediate_size"],
  "max_position_embeddings": config["max_seq_len"],
  "model_type": "llama",
  "num_attention_heads": config["n_head"],
  "num_hidden_layers": config["n_layer"],
  "num_key_value_heads": config["n_kv_head"],
  "rms_norm_eps": config.get("norm_eps", 1e-5),
  "rope_theta": config.get("rope_base", config.get("rope_theta", 10000.0)),
  "vocab_size": config["vocab_size"],
  "hidden_act": "silu",
  "tie_word_embeddings": True
}

with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(hf_config, f, indent=2)

print("CONVERSAO CONCLUIDA COM SUCESSO!")
