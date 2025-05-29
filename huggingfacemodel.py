import time
import psutil
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import os

class SimpleSQLGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_short_name = model_name.split('/')[-1]
        
        print(f"Loading model: {self.model_name}")
        print(f"Optimizing for CPU inference...")
        start_time = time.time()
        
        # CPU optimization - use all available cores
        torch.set_num_threads(os.cpu_count())
        
        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Model architecture detection - this is crucial for proper loading
            self.is_seq2seq = any(model_type in model_name.lower() 
                                for model_type in ['t5', 'flan', 'bart', 'pegasus', 'codet5'])
            self.is_code_model = any(keyword in model_name.lower() 
                                   for keyword in ['code', 'sql', 'codegen', 'starcoder'])
            self.is_instruct_model = any(keyword in model_name.lower() 
                                       for keyword in ['instruct', 'chat', 'flan'])
            
            print(f"✓ Model architecture: {'Seq2Seq' if self.is_seq2seq else 'Causal LM'}")
            print(f"✓ Code model: {self.is_code_model}, Instruct model: {self.is_instruct_model}")
            
            # Load appropriate model type with optimal settings
            if self.is_seq2seq:
                print("✓ Loading as Seq2Seq model (T5/FLAN/BART/CodeT5)")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # CPU works best with float32
                    device_map="cpu",
                    low_cpu_mem_usage=True,     # Important for large models
                    trust_remote_code=True      # Some models need this
                )
            else:
                print("✓ Loading as Causal LM model (GPT/CodeGen/Llama/SQLCoder)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            # Handle pad token - essential for proper generation
            if self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    print("✓ Using EOS token as pad token")
                else:
                    self.tokenizer.pad_token = self.tokenizer.unk_token or '[PAD]'
                    print("✓ Using UNK token as pad token")
            
            load_time = time.time() - start_time
            print(f"✓ {self.model_short_name} loaded in {load_time:.2f} seconds")
            print(f"✓ Using {torch.get_num_threads()} CPU threads")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def create_focused_prompt(self, natural_query):
            """
            Create a universal prompt that provides clear context for any model.
            Inspired by production Flask server patterns.
            """
            
            prompt = f"""You are an expert SQL Developer.

        Your task:
        - Read the USER REQUEST carefully
        - Generate a valid SQL query that answers their question
        - Output ONLY the SQL query (no explanation)

        Rules:
        - Use only these tables: customers, orders, products
        - Use proper SQL syntax
        - For time-based queries, use appropriate date functions
        - Match user requests to the most appropriate table

        USER REQUEST:
        {natural_query}

        Generate SQL query:
        SELECT"""
            
            return prompt
    
    def generate_sql(self, natural_query):
        """
        Generate SQL with proper model-specific handling and minimal post-processing.
        
        The goal is to get clean SQL directly from the model without needing
        complex cleaning functions.
        """
        
        prompt = self.create_focused_prompt(natural_query)
        
        # Tokenize with reasonable limits - too long prompts can cause issues
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512  # Reasonable limit to prevent memory issues
        )
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        # Performance tracking
        start_time = time.time()
        memory_before = psutil.virtual_memory().used / 1024**3
        
        with torch.no_grad():
            if self.is_seq2seq:
                # Seq2Seq models (T5, FLAN) - these typically generate cleaner output
                print("    Using Seq2Seq generation...")
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=60,          # Reasonable limit for SQL queries
                    temperature=0.2,            # Low temperature for more deterministic output
                    do_sample=True,
                    num_beams=1,               # Faster than beam search
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1     # Slight penalty to avoid repetition
                )
                # Seq2seq models only return the generated tokens
                sql_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            else:
                # Causal LM models (GPT, CodeGen, SQLCoder)
                print("    Using Causal LM generation...")
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=60,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9
                )
                # Causal LM includes the prompt, so we need to remove it
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                sql_response = full_response.replace(prompt, "").strip()
        
        # Calculate performance metrics
        generation_time = time.time() - start_time
        memory_after = psutil.virtual_memory().used / 1024**3
        memory_used = memory_after - memory_before
        
        # Minimal cleaning - just essential formatting
        sql_query = self._minimal_clean(sql_response)
        
        print(f"    Raw output: '{sql_response[:100]}{'...' if len(sql_response) > 100 else ''}'")
        print(f"    Final SQL: '{sql_query}'")
        
        return {
            "sql_query": sql_query,
            "generation_time": generation_time,
            "memory_used": memory_used,
            "input_tokens": len(inputs['input_ids'][0]),
            "output_tokens": len(outputs[0]) if not self.is_seq2seq else len(outputs[0]),
            "model_type": "Seq2Seq" if self.is_seq2seq else "Causal LM",
            "prompt_strategy": self._get_prompt_strategy(),
            "raw_response": sql_response
        }
    
    def _minimal_clean(self, sql_response):
        """
        Minimal cleaning - only remove obvious artifacts, let the model do the work.
        
        Philosophy: If we need extensive cleaning, the prompt is probably wrong.
        """
        if not sql_response:
            return ""
        
        # Remove common prefixes that models sometimes add
        sql_response = sql_response.strip()
        
        # Remove markdown code blocks if present
        if sql_response.startswith("```"):
            lines = sql_response.split('\n')
            sql_response = '\n'.join(lines[1:-1]) if len(lines) > 2 else sql_response
        
        # Remove common prefixes
        prefixes_to_remove = ["SQL:", "Query:", "Answer:", "sql:", "query:"]
        for prefix in prefixes_to_remove:
            if sql_response.startswith(prefix):
                sql_response = sql_response[len(prefix):].strip()
                break
        
        # Take only the first line/statement to avoid model rambling
        # This is key to preventing the repetition issue you encountered
        lines = sql_response.split('\n')
        first_meaningful_line = ""
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('--'):
                first_meaningful_line = line
                break
        
        return first_meaningful_line.strip()
    
    def _get_prompt_strategy(self):
        """Return which prompting strategy was used for debugging purposes"""
        if 'sqlcoder' in self.model_name.lower():
            return "SQLCoder-specific"
        elif self.is_code_model:
            return "Code-focused"
        elif self.is_instruct_model:
            return "Instruction-following"
        else:
            return "General"

def run_focused_test():
    """
    Test the focused approach with various models and queries.
    
    Key insight: We want to see clean, direct SQL generation without 
    the need for complex post-processing.
    """
    
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
    generator = SimpleSQLGenerator(model_name)
    
    # Focused test queries - start simple, then get more complex
    test_queries = [
        "show top 5 customers",    
        "count all orders",   
        "find expensive products",       
        "customers who bought recently", 
        "average order value by month"    
    ]
    
    print(f"\n{'='*60}")
    print(f"FOCUSED SQL GENERATION TEST: {generator.model_short_name}")
    print(f"{'='*60}")
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: '{query}'")
        print("-" * 50)
        
        result = generator.generate_sql(query)
        
        print(f"Strategy: {result['prompt_strategy']}")
        print(f"Generated: {result['sql_query']}")
        print(f"Time: {result['generation_time']:.2f}s")
        print(f"Tokens/sec: {result['output_tokens']/result['generation_time']:.1f}")
        
        results.append({
            'query': query,
            'sql': result['sql_query'],
            'time': result['generation_time'],
            'success': bool(result['sql_query'] and len(result['sql_query']) > 5)
        })
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    avg_time = sum(r['time'] for r in results) / len(results)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {generator.model_name}")
    print(f"Architecture: {result['model_type']}")
    print(f"Successful generations: {successful}/{len(results)}")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Strategy used: {result['prompt_strategy']}")

if __name__ == "__main__":
    # System info - useful for debugging performance issues
    print(f"🖥️  System Information")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"Available RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    
    run_focused_test()