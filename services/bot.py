"""Nutrition Bot service for handling customer queries with memory-augmented responses."""
import logging
import hashlib
from typing import Dict, List, Optional
from datetime import datetime

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mem0 import MemoryClient

from core.config import get_config
from core.cache import get_response_cache
from core.rate_limiter import get_rate_limiter
from core.metrics import get_metrics
from core.validation import get_validator, ValidationError
from agent_workflow.tools.rag import agentic_rag

logger = logging.getLogger(__name__)


class NutritionBot:
    """
    A conversational bot specialized in nutrition disorder guidance.
    
    Uses RAG for evidence-based responses and Mem0 for conversation memory.
    Features caching, rate limiting, metrics, and input validation.
    """
    
    def __init__(self):
        """Initialize the NutritionBot with all components."""
        self._config = get_config()
        self._cache = get_response_cache()
        self._rate_limiter = get_rate_limiter()
        self._metrics = get_metrics()
        self._validator = get_validator()
        
        self._setup_memory()
        self._setup_llm()
        self._setup_agent()
        logger.info("NutritionBot initialized successfully")

    def _setup_memory(self) -> None:
        """Initialize the memory client for storing conversation history."""
        if self._config.mem0_api_key:
            self.memory = MemoryClient(api_key=self._config.mem0_api_key)
            self._memory_enabled = True
            logger.info("Mem0 memory client initialized")
        else:
            self.memory = None
            self._memory_enabled = False
            logger.warning("Memory disabled - MEM0_API_KEY not configured")

    def _setup_llm(self) -> None:
        """Initialize the OpenAI chat model."""
        self.client = ChatOpenAI(
            model_name=self._config.chat_model,
            openai_api_key=self._config.openai_api_key,
            openai_api_base=self._config.openai_api_base,
            temperature=0,
            max_retries=3,
        )

    def _setup_agent(self) -> None:
        """Configure the tool-calling agent with RAG capabilities."""
        tools = [agentic_rag]

        system_prompt = """You are a caring and knowledgeable Medical Support Agent, specializing in nutrition disorder-related guidance. Your goal is to provide accurate, empathetic, and tailored nutritional recommendations while ensuring a seamless customer experience.

Guidelines for Interaction:
- Maintain a polite, professional, and reassuring tone.
- Show genuine empathy for customer concerns and health challenges.
- Reference past interactions to provide personalized and consistent advice.
- Engage with the customer by asking about their food preferences, dietary restrictions, and lifestyle before offering recommendations.
- Ensure consistent and accurate information across conversations.
- If any detail is unclear or missing, proactively ask for clarification.
- Always use the agentic_rag tool to retrieve up-to-date and evidence-based nutrition insights.
- Keep track of ongoing issues and follow-ups to ensure continuity in support.
- Your primary goal is to help customers make informed nutrition decisions that align with their health conditions and personal preferences.
- IMPORTANT: Do not provide medical diagnosis or prescribe treatments. Always recommend consulting healthcare professionals for serious concerns.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_tool_calling_agent(self.client, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=False,  # Set to True for debugging
            max_iterations=5,
            handle_parsing_errors=True,
        )

    def store_customer_interaction(
        self, 
        user_id: str, 
        message: str, 
        response: str, 
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Store customer interaction in memory for future reference.

        Args:
            user_id: Unique identifier for the customer.
            message: Customer's query or message.
            response: Bot's response.
            metadata: Additional metadata for the interaction.
        """
        if not self._memory_enabled:
            return
            
        if metadata is None:
            metadata = {}

        metadata["timestamp"] = datetime.now().isoformat()

        conversation = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]

        try:
            self.memory.add(
                conversation,
                user_id=user_id,
                output_format="v1.1",
                metadata=metadata
            )
            logger.debug(f"Stored interaction for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")

    def get_relevant_history(self, user_id: str, query: str) -> List[Dict]:
        """
        Retrieve past interactions relevant to the current query.

        Args:
            user_id: Unique identifier for the customer.
            query: The customer's current query.

        Returns:
            List of relevant past interactions.
        """
        if not self._memory_enabled:
            return []
            
        try:
            return self.memory.search(
                query=query,
                user_id=user_id,
                limit=5
            )
        except Exception as e:
            logger.error(f"Failed to retrieve history: {e}")
            return []

    def _format_context(self, relevant_history: List[Dict]) -> str:
        """Format relevant history into a context string."""
        if not relevant_history:
            return "No previous interactions found."
            
        context = "Previous relevant interactions:\n"
        for item in relevant_history:
            for turn in item.get("memory", []):
                role = turn.get("role", "unknown").capitalize()
                content = turn.get("content", "")
                context += f"{role}: {content}\n"
            context += "---\n"
        return context
    
    def _generate_cache_key(self, user_id: str, query: str) -> str:
        """Generate a cache key for the query."""
        combined = f"{user_id}:{query}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def handle_customer_query(self, user_id: str, query: str) -> str:
        """
        Process a customer's query and provide a response.

        Args:
            user_id: Unique identifier for the customer.
            query: Customer's query.

        Returns:
            Bot's response string.
        """
        start_time = datetime.now()
        
        # Input validation
        try:
            user_id = self._validator.validate_user_id(user_id)
            query = self._validator.validate_query(query)
        except ValidationError as e:
            logger.warning(f"Validation failed: {e}")
            self._metrics.increment("validation_failures")
            return str(e)
        
        logger.info(f"Handling query from user {user_id}: {query[:100]}...")
        
        # Rate limiting check
        if not self._rate_limiter.acquire(f"user:{user_id}"):
            logger.warning(f"Rate limit exceeded for user {user_id}")
            self._metrics.increment("rate_limit_exceeded")
            return "You're sending messages too quickly. Please wait a moment and try again."
        
        # Check cache for recent identical queries
        cache_key = self._generate_cache_key(user_id, query)
        cached_response = self._cache.get(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for user {user_id}")
            self._metrics.increment("cache_hits")
            return cached_response
        
        self._metrics.increment("cache_misses")
        
        # Retrieve relevant past interactions
        relevant_history = self.get_relevant_history(user_id, query)
        context = self._format_context(relevant_history)
        
        logger.debug(f"Retrieved {len(relevant_history)} relevant memories")

        # Prepare prompt with context
        prompt = f"""
Context from previous interactions:
{context}

Current customer query: {query}

Provide a helpful response that takes into account any relevant past interactions.
"""

        try:
            # Generate response
            self._metrics.increment("llm_requests")
            response = self.agent_executor.invoke({"input": prompt})
            output = response.get("output", "I apologize, but I couldn't generate a response. Please try again.")
            
            # Cache the response (5 minute TTL for personalized responses)
            self._cache.set(cache_key, output)
            
            # Store the interaction
            self.store_customer_interaction(
                user_id=user_id,
                message=query,
                response=output,
                metadata={"type": "support_query"}
            )
            
            # Record metrics
            latency = (datetime.now() - start_time).total_seconds()
            self._metrics.record_latency("query_response", latency)
            self._metrics.increment("successful_responses")
            
            logger.info(f"Response generated for user {user_id} in {latency:.2f}s")
            return output
            
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            self._metrics.increment("errors")
            return "I apologize, but I encountered an error processing your request. Please try again or rephrase your question."
    
    def get_health_status(self) -> Dict:
        """Get the health status and metrics of the bot."""
        cache_stats = self._cache.stats()
        return {
            "status": "healthy",
            "memory_enabled": self._memory_enabled,
            "cache_stats": cache_stats,
            "metrics": self._metrics.get_all_metrics(),
        }
