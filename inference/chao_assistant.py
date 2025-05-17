"""
Chao AI Assistant - Inference Script
Based on fine-tuned Qwen3 using Unsloth with extensive personalization
"""

import torch
import json
import os
import sys
import re
import datetime
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer
from unsloth import FastModel

# Load configuration
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

CONFIG = load_config()
USER_NAME = CONFIG["user_name"]
ASSISTANT_NAME = CONFIG["assistant_name"]
TONE = CONFIG["tone"]
AI_SETTINGS = CONFIG.get("ai_settings", {})
REASONING_MODE = CONFIG.get("reasoning_mode", {})
AUTOMATION_TRIGGERS = CONFIG.get("automation_triggers", {})
PERSONALIZATION = CONFIG.get("personalization", {})
ENVIRONMENT = CONFIG.get("environment", {})
TEMPLATES = CONFIG.get("templates", {})
SECURITY = CONFIG.get("security", {})

# Initialize conversation history
conversation_history = []
max_history_items = PERSONALIZATION.get("conversation_history", {}).get("max_history_items", 100)

# Load the fine-tuned model
def load_model(model_path="./models/chao-assistant"):
    # Get model settings from config
    model_config = AI_SETTINGS.get("model", {})
    context_length = model_config.get("context_length", 2048)
    use_4bit = model_config.get("use_4bit", True)
    
    print(f"Loading model with context length: {context_length}, 4-bit quantization: {use_4bit}")
    
    # Check if model path exists, if not use the base model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, falling back to base model")
        model_path = model_config.get("base_model", "unsloth/Qwen3-8B")
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=context_length,
        dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=use_4bit,
    )
    return model, tokenizer

# Check if prompt contains automation triggers
def detect_automation_trigger(prompt: str) -> Optional[str]:
    # Check if the assistant is being addressed
    greetings = PERSONALIZATION.get("greetings", [ASSISTANT_NAME])
    is_addressed = any(greeting.lower() in prompt.lower() for greeting in greetings)
    
    if not is_addressed:
        return None
        
    # Check for automation triggers
    for trigger_type, keywords in AUTOMATION_TRIGGERS.items():
        if any(keyword.lower() in prompt.lower() for keyword in keywords):
            return trigger_type
            
    return None

# Execute commands based on security settings
def execute_command(command: str) -> str:
    if not SECURITY.get("ask_before_execution", True):
        return _run_command(command)
        
    # Check against allowed executions
    allowed = SECURITY.get("allowed_execution", [])
    is_allowed = any(cmd in command.split()[0].lower() for cmd in allowed)
    
    if not is_allowed:
        return f"⚠️ Command execution not allowed for: {command}"
        
    print(f"\n⚠️ Do you want to execute: {command} [y/N]? ", end="")
    confirmation = input().lower()
    
    if confirmation == 'y':
        return _run_command(command)
    else:
        return "Command execution cancelled."

def _run_command(command: str) -> str:
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=60  # Add timeout for safety
        )
        if result.stderr and not result.stdout:
            return f"Error: {result.stderr}"
        return result.stdout or "Command executed successfully."
    except Exception as e:
        return f"Error executing command: {str(e)}"

# Generate a file from template
def generate_from_template(template_name: str, file_path: str, variables: Dict[str, str]) -> str:
    if template_name not in TEMPLATES:
        return f"Template '{template_name}' not found."
        
    template = TEMPLATES[template_name]
    content = ""
    
    # Generate content based on template type
    if template_name == "python_script":
        content = template.get("header", "").replace("{{description}}", variables.get("description", ""))
        for imp in template.get("imports", []):
            content += f"{imp}\n"
        content += "\n\n"
        content += variables.get("body", "# Your code here\n\n")
        content += template.get("main", "")
    elif template_name == "react_component":
        imports = ", ".join(template.get("imports", []))
        component_name = variables.get("name", "MyComponent")
        content = f"import React, {{ {imports} }} from 'react';\n\n"
        
        if template.get("style") == "functional":
            content += f"const {component_name} = () => {{\n"
            content += "  return (\n    <div>\n      {/* Component content */}\n    </div>\n  );\n};\n\n"
            content += f"export default {component_name};"
    
    # Write to file
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return f"File created successfully at {file_path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"

# Process automation request
def process_automation(trigger_type: str, prompt: str) -> Optional[str]:
    if trigger_type == "generate":
        # Extract information for generation
        python_match = re.search(r"python\s+(?:script|file)\s+(?:called|named)?\s*[\"']?([^\"']+)[\"']?", prompt, re.IGNORECASE)
        react_match = re.search(r"react\s+component\s+(?:called|named)?\s*[\"']?([^\"']+)[\"']?", prompt, re.IGNORECASE)
        
        if python_match:
            file_name = python_match.group(1).strip()
            if not file_name.endswith('.py'):
                file_name += '.py'
                
            file_path = os.path.join(ENVIRONMENT.get("project_root", "."), file_name)
            description = re.search(r"description:?\s*[\"']?([^\"']+)[\"']?", prompt, re.IGNORECASE)
            desc_text = description.group(1) if description else "Generated Python script"
            
            return generate_from_template("python_script", file_path, {"description": desc_text})
            
        elif react_match:
            component_name = react_match.group(1).strip()
            if not component_name.endswith('.jsx') and not component_name.endswith('.tsx'):
                component_name += '.jsx'
                
            file_path = os.path.join(ENVIRONMENT.get("project_root", "."), component_name)
            return generate_from_template("react_component", file_path, {"name": os.path.splitext(os.path.basename(component_name))[0]})
    
    elif trigger_type == "run":
        # Extract command to run
        command_match = re.search(r"(?:run|execute|start)\s+[\"']?(.+?)[\"']?(?:$|\s+in|\s+with)", prompt, re.IGNORECASE)
        if command_match:
            command = command_match.group(1).strip()
            return execute_command(command)
            
    # Other automation types can be added here
    return None

# Generate response with thinking mode
def generate_response(model, tokenizer, prompt: str, conversation_context: List[Dict[str, str]] = None, enable_thinking: bool = None) -> str:
    # Determine if thinking mode should be enabled
    if enable_thinking is None:
        enable_thinking = REASONING_MODE.get("default_enabled", True)
        # Check for thinking mode overrides in prompt
        if "/think" in prompt.lower():
            enable_thinking = True
            prompt = re.sub(r'/think', '', prompt, flags=re.IGNORECASE).strip()
        elif "/nothink" in prompt.lower():
            enable_thinking = False
            prompt = re.sub(r'/nothink', '', prompt, flags=re.IGNORECASE).strip()
    
    # Check for automation triggers
    trigger_type = detect_automation_trigger(prompt)
    if trigger_type:
        automation_result = process_automation(trigger_type, prompt)
        if automation_result:
            # Add the automation result to the prompt for context
            prompt = f"{prompt}\n\nI've completed this automation task: {automation_result}"
    
    # Prepare messages with context if available
    if conversation_context and PERSONALIZATION.get("remember_context", True):
        messages = conversation_context + [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]
    
    # Get model generation settings
    model_config = AI_SETTINGS.get("model", {})
    temperature = model_config.get("thinking_temperature" if enable_thinking else "temperature", 0.6 if enable_thinking else 0.7)
    top_p = model_config.get("thinking_top_p" if enable_thinking else "top_p", 0.95 if enable_thinking else 0.8)
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=temperature,
        top_p=top_p,
        top_k=20,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    assistant_response = response.split("<|im_start|>assistant\n")[-1]
    
    return assistant_response

# Format response based on style preferences
def format_response(response: str) -> str:
    response_style = CONFIG.get("response_style", {})
    
    # Apply emoji if enabled
    emoji_usage = response_style.get("emoji_usage", "none")
    if emoji_usage == "none":
        # Remove any emojis
        response = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]', '', response)
    
    # Adjust brevity if needed
    brevity = response_style.get("brevity", "balanced")
    if brevity == "very_concise" and len(response) > 200:
        sentences = re.split(r'(?<=[.!?])\s+', response)
        response = ' '.join(sentences[:3]) if len(sentences) > 3 else response
    
    return response

# Save conversation to history file
def save_conversation(user_input: str, assistant_response: str):
    if not PERSONALIZATION.get("conversation_history", {}).get("save_history", False):
        return
        
    history_dir = "conversation_history"
    os.makedirs(history_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    history_file = os.path.join(history_dir, f"{timestamp}_conversation.jsonl")
    
    with open(history_file, "a") as f:
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_input,
            "assistant": assistant_response
        }
        f.write(json.dumps(entry) + "\n")

# Interactive chat loop
def interactive_chat():
    print(f"Loading {ASSISTANT_NAME} Assistant...")
    model, tokenizer = load_model()
    print(f"{ASSISTANT_NAME} is ready to chat! Type 'exit' to end the conversation.")
    
    # Welcome message based on settings
    current_hour = datetime.datetime.now().hour
    greeting = "Good morning" if 5 <= current_hour < 12 else "Good afternoon" if 12 <= current_hour < 18 else "Good evening"
    welcome_msg = f"{greeting}, {USER_NAME}! I'm {ASSISTANT_NAME}, your personal AI assistant. How can I help you today?"
    print(f"\n{ASSISTANT_NAME}: {welcome_msg}")
    
    # Store conversation context if enabled
    context_window = PERSONALIZATION.get("context_window", 5)
    conversation_context = []
    
    while True:
        user_input = input(f"\n{USER_NAME}: ")
        if user_input.lower() == "exit":
            print(f"\n{ASSISTANT_NAME}: Goodbye! Have a great day!")
            break
        
        # Add user message to context
        if PERSONALIZATION.get("remember_context", True):
            conversation_context.append({"role": "user", "content": user_input})
            # Trim context to window size
            if len(conversation_context) > context_window * 2:  # *2 because we have pairs of messages
                conversation_context = conversation_context[-context_window * 2:]
        
        # Generate response
        response = generate_response(model, tokenizer, user_input, conversation_context)
        
        # Format response based on style preferences
        formatted_response = format_response(response)
        
        # Add response to context
        if PERSONALIZATION.get("remember_context", True):
            conversation_context.append({"role": "assistant", "content": response})
        
        # Save conversation if enabled
        save_conversation(user_input, response)
        
        # Display response
        print(f"\n{ASSISTANT_NAME}: {formatted_response}")

# Main entry point
if __name__ == "__main__":
    interactive_chat() 