import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from colorama import init, Fore, Style

init(autoreset=True)

model_dir = r"C:\Users\Fernando\Downloads\Synapse-4B-HF"
print(Fore.CYAN + "Carregando a mente da Synapse-4B para a RAM (Aguarde)...")

# Tenta carregar o modelo em bfloat16, se o PC aguentar. Caso não suporte, usa float32.
try:
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, device_map="cpu")
except:
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, device_map="cpu")

print(Fore.CYAN + "Carregando o Dicionario (Tokenizer)...")
# O LlamaTokenizer usa o mesmo padrao de arquivo do nosso
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
except Exception as e:
    print(Fore.RED + "AVISO: O arquivo 'tokenizer.json' ou 'tokenizer.model' nao foi encontrado dentro da pasta Synapse-4B-HF.")
    print(Fore.RED + "Por favor, copie o tokenizer.json do Kaggle para dentro da pasta C:\\Users\\Fernando\\Downloads\\Synapse-4B-HF e rode de novo!")
    exit(1)

print(Fore.GREEN + "\n=======================================================")
print(Fore.GREEN + "  SYNAPSE-4B LIGADA COM SUCESSO! (Passo 4100)")
print(Fore.GREEN + "=======================================================\n")

while True:
    try:
        user_input = input(Fore.YELLOW + "Você: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            break
            
        # Formata a entrada do mesmo jeito que foi treinado
        prompt = f"<sistema>\nVocê é a Synapse, uma inteligência artificial avançada e prestativa.\nPense em <pensamento> antes de responder.\n</sistema>\n<usuario> {user_input} <synapse> "
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        print(Fore.MAGENTA + "Synapse: ", end="", flush=True)
        
        # Gera a resposta (Isso vai rodar na CPU, entao vai ser devagar)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2 # Default EOS
        )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        # Limpa as tags de pensamento se quiser, ou deixa pra ele ver
        print(Style.RESET_ALL + response)
        print("-" * 50)
        
    except KeyboardInterrupt:
        break
