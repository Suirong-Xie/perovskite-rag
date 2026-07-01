"""Agent: the orchestration loop that decides tool usage and generates answers."""

import json
import traceback
from typing import AsyncGenerator, List, Dict

from models import llm
from sessions import get_history_text, add_to_history
from tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS

# ── System prompt ──

SYSTEM_PROMPT = """You are PerovskiteGPT, a world-class materials science professor specialized in perovskite solar cells, optoelectronics, and device physics.

## Your capabilities
- You have access to a research paper database containing perovskite literature.
- You can use tools to retrieve information, search conversation history, or answer directly.
- You combine retrieved knowledge with your own expertise to give comprehensive answers.

## Available tools
1. `retrieve_papers(query, top_k=10)` — Search paper database for relevant literature
2. `read_paper_fulltext(source, section)` — Read the full abstract/conclusion of a specific paper from the original PDF. Use this when retrieved snippets lack sufficient detail.
3. `search_session_history(keywords)` — Search past conversation

## Rules
- If the question asks about specific data, papers, numbers, comparisons, or recent research:
  Use the `retrieve_papers` tool to find relevant literature.
- If the question is about general concepts, theory, analysis, coding, or opinion:
  Answer directly using your own knowledge — no need to retrieve.
- For ambiguous questions, use your judgment. When in doubt, retrieve.
- When using `retrieve_papers`, choose `top_k` wisely:
  - 0: no retrieval needed, answer from knowledge
  - 1-5: narrow/technical question about a specific concept
  - 5-10: standard question, moderate breadth
  - 10-15: broad topic needing comprehensive coverage
  - 15-20: survey/comparison question needing many sources
- ALWAYS cite sources when you use retrieved information.
- Clearly distinguish between:
  - Facts from retrieved papers → cite with [Source N]
  - Your own knowledge or analysis → mark as "based on my understanding"
- Be thorough and scientific. Don't pad answers, but don't truncate them either.

## Critical: When you retrieve papers, you MUST extract and report specific data
The retrieved text snippets are fragments from papers. Study them carefully and extract:
- **Efficiency numbers**: PCE, Jsc, Voc, FF values with units
- **Device architecture**: n-i-p, p-i-n, HTL/ETL materials, substrate type
- **Test conditions**: humidity, temperature, duration of stability tests
- **Comparison data**: treated vs control performance
- If a paper reports improvements in percentages, include both the absolute values and the relative improvement.

## Output format — CRITICAL
You MUST follow these formatting rules in ALL answers. Your formatting is as important as the science.

### Markdown formatting rules
- Use **bold** for ALL numbers and key terms: efficiency values, materials names, crucial concepts
- Use `code` for formulas, device architecture abbreviations, technical acronyms
- Use | tables | for comparisons: efficiency metrics, device parameters, experimental conditions
- Use bullet lists for enumerating factors, strategies, or sequential findings
- Use **###** section headers for organizing multi-part answers
- Use > blockquotes for important caveats or conclusions you want to highlight

### Structural rules for paper-related answers
Always structure with vivid formatting like this example:

```
### Overview

The literature on **passivation strategies for perovskite solar cells** reveals three dominant approaches...

### Key Findings

| Strategy | Material | PCE (%) | Voc (V) | FF (%) | Stability | Source |
|----------|----------|---------|---------|--------|-----------|--------|
| 2D/3D interface | PEAI | **25.7** | **1.18** | 81.2 | >1000h @85°C | [1] |
| Additive | MACl | **25.3** | 1.16 | **82.5** | >500h @RH40% | [2] |
| Surface passivation | OLAI | 24.8 | 1.15 | 80.1 | >800h @N₂ | [3] |

### Mechanism & Analysis

The **PEAI-based 2D/3D heterostructure** [Source 1] achieved the highest efficiency of **25.7%** by forming a low-dimensional perovskite layer at the interface, which effectively passivates **halide vacancy defects** and suppresses **non-radiative recombination**.

> This is particularly notable because the Voc of **1.18 V** approaches the Shockley-Queisser limit for a 1.6 eV bandgap material, indicating near-ideal interface quality.

### Critical Assessment

While the **25.7% PCE** is impressive, the stability test was conducted under N₂ atmosphere rather than real-world conditions (RH >50%, 85°C). This is a significant gap in the current literature...
```

- If a specific number/data point is missing from the retrieved text, say "the retrieved excerpt does not specify this value" — do NOT make up numbers
- For general/knowledge questions (no retrieval), still use bold + tables + structure where appropriate"""


def _call_llm(messages: List[Dict], include_tools: bool = True) -> dict:
    """Call Ollama. include_tools=False for the second-round synthesis (no tool definitions needed)."""
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"Human: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
        elif role == "tool":
            prompt_parts.append(f"Tool ({msg.get('name', 'tool')}): {content}")
    
    prompt = "\n\n".join(prompt_parts)
    
    if include_tools:
        prompt += "\n\nAVAILABLE TOOLS:\n"
        for t in TOOL_DEFINITIONS:
            fn = t["function"]
            prompt += f"- {fn['name']}: {fn['description']}\n"
            for param_name, param_desc in fn["parameters"]["properties"].items():
                prompt += f"  {param_name}: {param_desc['description']}\n"
        
        prompt += "\nTo use a tool, respond with EXACTLY:\nTOOL_CALL: <tool_name>\nARGS: <JSON arguments>\n\nOtherwise respond normally.\n"
    
    result = llm.invoke(prompt)
    return {"content": result, "tool_calls": _parse_tool_call(result)}


def _parse_tool_call(text: str):
    """Parse tool call from model output if present."""
    if "TOOL_CALL:" in text and "ARGS:" in text:
        try:
            lines = text.strip().split("\n")
            tool_name = ""
            args_str = ""
            for line in lines:
                if line.startswith("TOOL_CALL:"):
                    tool_name = line.replace("TOOL_CALL:", "").strip()
                elif line.startswith("ARGS:"):
                    args_str = line.replace("ARGS:", "").strip()
            
            if tool_name and args_str:
                args = json.loads(args_str)
                return [{"name": tool_name, "args": args}]
        except:
            return None
    return None


def _clean_answer(text: str) -> str:
    """Remove tool call syntax that leaked into the final answer."""
    import re
    # Remove TOOL_CALL lines and ARGS lines
    text = re.sub(r'\n*\s*TOOL_CALL:\s*\w+\s*\n*', '\n', text)
    text = re.sub(r'\n*\s*ARGS:\s*\{[^}]*\}\s*\n*', '\n', text)
    text = re.sub(r'^\s*\{[^}]*\}\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def _build_messages(question: str, session_id: str) -> List[Dict]:
    """Build message list with history."""
    history_text = get_history_text(session_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if history_text:
        messages.append({"role": "system", "content": f"Conversation history:\n{history_text}"})
    
    messages.append({"role": "user", "content": question})
    return messages


def run_agent(question: str, session_id: str) -> dict:
    """Run the agent loop: decide → tool → answer."""
    print(f"[AGENT] Session: {session_id} | Question: {question[:100]}...")
    
    try:
        messages = _build_messages(question, session_id)
        response = _call_llm(messages)
        
        tool_calls = response.get("tool_calls")
        if tool_calls:
            # Execute tool
            tc = tool_calls[0]
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"[AGENT] Using tool: {tool_name} | Args: {tool_args}")
            
            if tool_name == "retrieve_papers":
                result = TOOL_FUNCTIONS["retrieve_papers"](
                    query=tool_args.get("query", question),
                    top_k=tool_args.get("top_k", 10)
                )
            elif tool_name == "search_session_history":
                result = TOOL_FUNCTIONS["search_session_history"](
                    keywords=tool_args.get("keywords", ""),
                    session_id=session_id
                )
            else:
                result = {"result": f"Unknown tool: {tool_name}", "sources": []}
            
            # Auto-expand: try to read full abstract/conclusion from top papers
            # if the retrieved chunks seem insufficient
            expanded_context = result.get("result", "")
            sources_list = result.get("sources", [])
            
            # Extract source filenames from sources
            source_files = []
            for s in sources_list[:3]:  # top 3 papers
                src_name = s.split(":")[0].strip() if ":" in s else s[:50]
                if src_name.endswith(".pdf"):
                    source_files.append(src_name)
            
            if source_files:
                for src in source_files:
                    try:
                        fulltext = TOOL_FUNCTIONS["read_paper_fulltext"](src, "abstract")
                        if "未找到论文文件" not in fulltext.get("result", ""):
                            expanded_context += "\n\n" + fulltext.get("result", "")
                    except:
                        pass
            
            # Feed expanded context to LLM
            messages.append({"role": "tool", "name": tool_name, "content": expanded_context})
            
            tool_result_prompt = f"\n\nTool '{tool_name}' returned:\n{result['result']}\n\nNow provide your final answer based on this information."
            messages.append({"role": "user", "content": tool_result_prompt})
            
            final_response = _call_llm(messages, include_tools=False)
            answer = final_response["content"]
            
            # Clean up tool call syntax from answer if leaked
            answer = _clean_answer(answer)
            
            sources = result.get("sources", [])
        else:
            # No tool call — direct answer
            answer = _clean_answer(response["content"])
            sources = []
        
        # Clean up
        answer = answer.strip()
        add_to_history(session_id, question, answer, sources)
        
        print(f"[AGENT] Answer generated, length {len(answer)}")
        return {"result": answer, "sources": sources}
    
    except Exception as e:
        print("[AGENT ERROR]")
        traceback.print_exc()
        return {"result": f"I encountered an error: {str(e)}", "sources": []}


async def run_agent_stream(question: str, session_id: str) -> AsyncGenerator[str, None]:
    """Streaming version of the agent loop."""
    import asyncio
    
    print(f"[AGENT STREAM] Session: {session_id} | Question: {question[:100]}...")
    
    try:
        messages = _build_messages(question, session_id)
        response = _call_llm(messages)
        
        tool_calls = response.get("tool_calls")
        sources = []
        
        if tool_calls:
            tc = tool_calls[0]
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"[AGENT STREAM] Using tool: {tool_name}")
            
            if tool_name == "retrieve_papers":
                result = TOOL_FUNCTIONS["retrieve_papers"](
                    query=tool_args.get("query", question),
                    top_k=tool_args.get("top_k", 10)
                )
            elif tool_name == "search_session_history":
                result = TOOL_FUNCTIONS["search_session_history"](
                    keywords=tool_args.get("keywords", ""),
                    session_id=session_id
                )
            else:
                result = {"result": f"Unknown tool: {tool_name}", "sources": []}
            
            sources = result.get("sources", [])
            
            # Auto-expand: read full abstract from top papers
            expanded_context = result.get("result", "")
            source_files = [s.split(":")[0].strip() for s in sources[:3] if ":" in s and s.split(":")[0].strip().endswith(".pdf")]
            for src in source_files:
                try:
                    fulltext = TOOL_FUNCTIONS["read_paper_fulltext"](src, "abstract")
                    if "未找到论文文件" not in fulltext.get("result", ""):
                        expanded_context += "\n\n" + fulltext.get("result", "")
                except:
                    pass
            
            # Yield sources
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            # Feed result to LLM for streaming answer
            messages.append({"role": "tool", "name": tool_name, "content": expanded_context})
            tool_result_prompt = f"\n\nTool '{tool_name}' returned:\n{expanded_context}\n\nNow provide your final answer based on this information."
            messages.append({"role": "user", "content": tool_result_prompt})
            
            # Build prompt for streaming
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"Human: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
                elif role == "tool":
                    prompt_parts.append(f"Tool ({msg.get('name', 'tool')}): {content}")
            
            stream_prompt = "\n\n".join(prompt_parts)
            
            full_answer = ""
            for chunk in llm.stream(stream_prompt):
                if chunk:
                    full_answer += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                await asyncio.sleep(0)
        else:
            # No tool call — stream direct answer (no sources check-in needed)
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"Human: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            
            stream_prompt = "\n\n".join(prompt_parts)
            
            full_answer = ""
            for chunk in llm.stream(stream_prompt):
                if chunk:
                    full_answer += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                await asyncio.sleep(0)
        
        # Save and signal done
        full_answer_clean = _clean_answer(full_answer.strip())
        add_to_history(session_id, question, full_answer_clean, sources)
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
    
    except Exception as e:
        print("[AGENT STREAM ERROR]")
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
