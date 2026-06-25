from __future__ import annotations

import re
import requests
from typing import Any
from pydantic import BaseModel, Field

from butler.agent.provider import OllamaProvider, GeminiProvider, AnthropicProvider, NvidiaProvider
from butler.paths import butler_home_dir
from butler.tools.base import Tool, ToolContext

class InstallSkillArgs(BaseModel):
    url: str = Field(description="The URL of the remote Python script or instructions to download and install as a skill.")
    name: str = Field(description="The target name of the skill. Must be alphanumeric and underscores only.")

class InstallSkillTool(Tool[InstallSkillArgs]):
    name = "skills.install"
    description = "Downloads and installs a new skill (Python Tool) from a remote URL into the Butler environment. Use this when the user asks to download or install a skill from a URL."
    input_model = InstallSkillArgs
    side_effect = True

    def call(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        url = args.get("url")
        name = args.get("name")
        
        if not url or not name:
            return {"error": "url and name are required."}
            
        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            return {"error": "Invalid skill name. Must be alphanumeric and underscores only."}
            
        skills_dir = butler_home_dir() / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = skills_dir / f"{name}.py"
        
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            script_content = resp.text
        except Exception as e:
            return {"error": f"Failed to download skill from {url}: {e}"}
            
        system_prompt = """You are an expert Python developer. 
Your task is to write a Python script that implements a BUTLER Tool for the given skill instructions or script.
The BUTLER Tool must:
1. Import `Tool` and `ToolContext` from `butler.tools.base`.
2. Import `BaseModel` and `Field` from `pydantic`.
3. Define an arguments model inheriting from `BaseModel`.
4. Define a class inheriting from `Tool`.
5. Implement the `call(self, ctx: ToolContext, args: dict) -> dict` method.
6. Provide a module-level `build()` function that returns a list containing an instance of your Tool.
Return ONLY valid Python code. No markdown formatting, no explanations."""
        
        provider_name = ctx.config.skill_compiler_provider.lower()
        model_name = ctx.config.skill_compiler_model
        
        provider = None
        if provider_name == "gemini":
            provider = GeminiProvider(api_keys=ctx.config.gemini_api_keys, model=model_name)
        elif provider_name == "claude":
            provider = AnthropicProvider(api_keys=ctx.config.claude_api_keys, model=model_name)
        elif provider_name == "nvidia":
            provider = NvidiaProvider(api_keys=ctx.config.nvidia_api_keys, model=model_name)
        else:
            provider = OllamaProvider(base_url=ctx.config.ollama_url, model=model_name)
            
        user_prompt = f"Skill Name: {name}\n\nSkill Content/Instructions:\n{script_content}\n\nWrite the Python wrapper tool."
        
        try:
            python_code = provider.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.1)
            
            if "```python" in python_code:
                python_code = python_code.split("```python")[1].split("```")[0].strip()
            elif "```" in python_code:
                python_code = python_code.split("```")[1].strip()
                
        except Exception as e:
            return {"error": f"LLM compilation failed: {e}"}
            
        target_path.write_text(python_code, encoding="utf-8")
        
        if hasattr(ctx.registry, "reload"):
            ctx.registry.reload()
            
        return {"status": "success", "message": f"Skill '{name}' installed successfully from {url} and registry reloaded."}


def build() -> list[Tool]:
    return [
        InstallSkillTool(
            name=InstallSkillTool.name,
            description=InstallSkillTool.description,
            input_model=InstallSkillTool.input_model,
            handler=InstallSkillTool().call,
            side_effect=InstallSkillTool.side_effect,
        )
    ]
