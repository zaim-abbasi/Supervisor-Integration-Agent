"""
Plan execution: resolve inputs for each step, call the right agent, and collect
outputs for later synthesis or UI debugging.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Tuple

try:
    import httpx  # type: ignore
except ImportError:
    httpx = None

from .agent_caller import call_agent
from .models import AgentMetadata, AgentRequest, AgentResponse, ErrorModel, Plan, UsedAgentEntry
from .registry import find_agent_by_name


def resolve_input(input_source: str, user_query: str, step_outputs: Dict[int, AgentResponse]) -> str:
    """Resolve an input_source directive into text for the worker."""
    if input_source == "user_query":
        return user_query
    if input_source.startswith("step:"):
        parts = input_source.split(":")
        if len(parts) >= 2:
            step_id_str = parts[1].split(".")[0]
            try:
                step_id = int(step_id_str)
                prior = step_outputs.get(step_id)
                if prior and prior.output:
                    return str(prior.output.result)
            except ValueError:
                pass
    return user_query


async def call_agent_with_trigger(
    agent_meta: AgentMetadata,
    intent: str,
    context: Dict[str, Any],
) -> AgentResponse:
    """
    Call TDA with a database trigger signal.
    TDA will retrieve tasks from MongoDB automatically.
    """
    request_id = str(uuid.uuid4())
    
    # Build handshake with trigger signal
    handshake = AgentRequest(
        request_id=request_id,
        agent_name=agent_meta.name,
        intent=intent,
        input={"trigger": "database_update"},  # Signal to retrieve from DB
        context=context,
    )

    if agent_meta.type == "http" and agent_meta.endpoint and httpx is not None:
        try:
            async with httpx.AsyncClient(timeout=agent_meta.timeout_ms / 1000) as client:
                resp = await client.post(agent_meta.endpoint, json=handshake.dict())
                if resp.status_code != 200:
                    return AgentResponse(
                        request_id=request_id,
                        agent_name=agent_meta.name,
                        status="error",
                        error=ErrorModel(
                            type="http_error",
                            message=f"HTTP {resp.status_code} calling {agent_meta.endpoint}",
                        ),
                    )
                return AgentResponse(**resp.json())
        except Exception as exc:
            return AgentResponse(
                request_id=request_id,
                agent_name=agent_meta.name,
                status="error",
                error=ErrorModel(type="network_error", message=str(exc)),
            )
    else:
        return AgentResponse(
            request_id=request_id,
            agent_name=agent_meta.name,
            status="error",
            error=ErrorModel(type="config_error", message="Agent endpoint not configured"),
        )


async def execute_plan(
    query: str,
    plan: Plan,
    registry: List[AgentMetadata],
    context: Dict[str, Any],
) -> Tuple[Dict[int, AgentResponse], List[UsedAgentEntry]]:
    """Execute each planned step in order and capture responses."""
    step_outputs: Dict[int, AgentResponse] = {}
    used_agents: List[UsedAgentEntry] = []

    for step in plan.steps:
        agent_meta = find_agent_by_name(step.agent, registry)
        text = resolve_input(step.input_source, query, step_outputs)
        response = await call_agent(agent_meta, step.intent, text, context)
        step_outputs[step.step_id] = response
        used_agents.append(
            UsedAgentEntry(name=agent_meta.name, intent=step.intent, status=response.status)
        )
        
        # Auto-trigger TDA after KnowledgeBaseBuilderAgent successfully creates tasks
        if (step.agent == "KnowledgeBaseBuilderAgent" and 
            response.status == "success" and 
            step.intent == "create_task"):
            try:
                tda_meta = find_agent_by_name("task_dependency_agent", registry)
                # Call TDA with database trigger - it will retrieve tasks from MongoDB
                tda_response = await call_agent_with_trigger(
                    tda_meta,
                    "task.resolve_dependencies",
                    context
                )
                # Add TDA to outputs with next step_id
                next_step_id = max(step_outputs.keys()) + 1 if step_outputs else 0
                step_outputs[next_step_id] = tda_response
                used_agents.append(
                    UsedAgentEntry(
                        name=tda_meta.name,
                        intent="task.resolve_dependencies",
                        status=tda_response.status
                    )
                )
            except KeyError:
                # TDA not found in registry, skip auto-trigger
                pass
            except Exception:
                # TDA call failed, continue without blocking
                pass

    return step_outputs, used_agents
