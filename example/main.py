import asyncio
from agents import Agent, Runner, function_tool
from agents.models.openai_provider import OpenAIProvider
from agents.run import RunConfig


# 1) FUNCTION TOOL SKILL
@function_tool
def calculate_gst(amount: float) -> str:
    """Calculate 18% GST on a given amount."""
    gst = amount * 0.18
    total = amount + gst
    return f"GST: {gst:.2f}, Total: {total:.2f}"


# 2) SPECIALIST AGENTS
finance_agent = Agent(
    name="Finance Specialist",
    instructions="You answer finance and tax questions clearly.",
    model="gemma4",
    input_guardrails=[],
    output_guardrails=[],
    handoffs=[],
)

marketing_agent = Agent(
    name="Marketing Specialist",
    instructions="You answer branding and growth questions clearly.",
    model="gemma4",
)

# 3) MAIN ROUTER WITH HANDOFFS + TOOL
main_agent = Agent(
    name="Business Assistant",
    instructions=(
        "You are a startup assistant. "
        "Use calculate_gst for tax calculations. "
        "Hand off finance questions to Finance Specialist. "
        "Hand off marketing questions to Marketing Specialist."
    ),
    model="gemma4",
    tools=[calculate_gst],
    handoffs=[finance_agent, marketing_agent],
)


# 4) SIMPLE MEMORY LAYER
memory = {"user_name": "Shamkumar", "business_interest": "fintech MVP"}


def inject_memory(user_input: str) -> str:
    return (
        f"User name: {memory['user_name']}. "
        f"Business interest: {memory['business_interest']}. "
        f"User question: {user_input}"
    )


async def main():
    # Use Ollama's OpenAI-compatible endpoint (requires a dummy API key)
    ollama_provider = OpenAIProvider(
        base_url="http://localhost:11434/",
        api_key="gemma4",  # Ollama doesn't verify the API key
        use_responses=False,
    )
    run_config = RunConfig(model_provider=ollama_provider)

    user_query = "Can you calculate GST on 1000 and also advise my fintech MVP?"
    enriched_query = inject_memory(user_query)

    result = await Runner.run(main_agent, enriched_query, run_config=run_config)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
