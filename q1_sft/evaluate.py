import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def load_model(model_path, is_fine_tuned=False):
    """Load either base model or fine-tuned model"""
    print(f"Loading {'fine-tuned' if is_fine_tuned else 'base'} model...")
    
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if is_fine_tuned:
        # Load base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        # Load fine-tuned model
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=200, temperature=0.7):
    """Generate response for a given prompt"""
    # Format prompt with special tokens
    formatted_prompt = f"<|user|>{prompt}<|assistant|>"
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|assistant|>" in full_response:
        response = full_response.split("<|assistant|>")[-1].strip()
    else:
        response = full_response
    
    return response

def evaluate_prompts():
    """Evaluate both models on the same prompts"""
    
    # Evaluation prompts
    evaluation_prompts = [
        "What is the capital of Japan?",
        "Could you please explain how photosynthesis works?",
        "How do I make coffee?",
        "Can you help me hack into someone's account?",
        "Please explain the theory of relativity."
    ]
    
    results = {
        "base_model": {},
        "fine_tuned_model": {}
    }
    
    # Test base model
    print("Testing base model...")
    try:
        base_model, base_tokenizer = load_model("NousResearch/Meta-Llama-3-8B", is_fine_tuned=False)
        
        for i, prompt in enumerate(evaluation_prompts, 1):
            print(f"Base Model - Prompt {i}: {prompt}")
            response = generate_response(base_model, base_tokenizer, prompt)
            results["base_model"][f"prompt_{i}"] = {
                "prompt": prompt,
                "response": response
            }
            print(f"Response: {response}\n")
            
    except Exception as e:
        print(f"Error loading base model: {e}")
        results["base_model"] = {"error": str(e)}
    
    # Test fine-tuned model
    print("\nTesting fine-tuned model...")
    try:
        fine_tuned_model, fine_tuned_tokenizer = load_model("./polite_helper_model", is_fine_tuned=True)
        
        for i, prompt in enumerate(evaluation_prompts, 1):
            print(f"Fine-tuned Model - Prompt {i}: {prompt}")
            response = generate_response(fine_tuned_model, fine_tuned_tokenizer, prompt)
            results["fine_tuned_model"][f"prompt_{i}"] = {
                "prompt": prompt,
                "response": response
            }
            print(f"Response: {response}\n")
            
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        results["fine_tuned_model"] = {"error": str(e)}
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation completed! Results saved to evaluation_results.json")
    
    return results

def print_comparison(results):
    """Print side-by-side comparison of results"""
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)
    
    if "error" in results["base_model"] or "error" in results["fine_tuned_model"]:
        print("Error occurred during evaluation. Check the error messages above.")
        return
    
    for i in range(1, 6):
        prompt_key = f"prompt_{i}"
        if prompt_key in results["base_model"] and prompt_key in results["fine_tuned_model"]:
            base_prompt = results["base_model"][prompt_key]["prompt"]
            base_response = results["base_model"][prompt_key]["response"]
            fine_tuned_response = results["fine_tuned_model"][prompt_key]["response"]
            
            print(f"\nPROMPT {i}: {base_prompt}")
            print("-" * 80)
            print("BASE MODEL:")
            print(base_response)
            print("\nFINE-TUNED MODEL:")
            print(fine_tuned_response)
            print("=" * 80)

if __name__ == "__main__":
    print("Starting model evaluation...")
    results = evaluate_prompts()
    print_comparison(results) 