import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def load_model(model_type="base"):
    """Load the specified model type"""
    print(f"Loading {model_type} model...")
    
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if model_type == "fine_tuned":
        # Load base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        # Load fine-tuned model
        model = PeftModel.from_pretrained(base_model, "./polite_helper_model")
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

def interactive_mode(model, tokenizer):
    """Run interactive chat mode"""
    print("\nInteractive mode started. Type 'quit' to exit.")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("Assistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test model responses")
    parser.add_argument("--model", choices=["base", "fine_tuned"], default="base",
                       help="Model type to use (default: base)")
    parser.add_argument("--prompt", type=str, help="Single prompt to test")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    
    try:
        model, tokenizer = load_model(args.model)
        
        if args.interactive:
            interactive_mode(model, tokenizer)
        elif args.prompt:
            print(f"Prompt: {args.prompt}")
            print("Response:", end=" ")
            response = generate_response(model, tokenizer, args.prompt, 
                                      args.max_length, args.temperature)
            print(response)
        else:
            # Default test prompt
            test_prompt = "Could you please explain how photosynthesis works?"
            print(f"Testing with default prompt: {test_prompt}")
            print("Response:", end=" ")
            response = generate_response(model, tokenizer, test_prompt, 
                                      args.max_length, args.temperature)
            print(response)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 