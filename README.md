# assignment-1
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Week 01 - Homework Assignment: AI Agent Development - COMPLETED SOLUTION
Choose ONE framework (LangGraph OR Google ADK) and complete the corresponding sections
"""

# Project Management Tools (Already provided - enhanced versions)
def schedule_task(task_name: str, deadline: str, priority: str) -> str:
    """Schedule a project task with deadline and priority.
    
    Args:
        task_name: Name of the task to schedule
        deadline: Deadline for the task (format: YYYY-MM-DD)
        priority: Priority level (High, Medium, Low)
    
    Returns:
        Confirmation message with task details
    """
    # Enhanced with additional project management details
    task_id = f"TSK-{hash(task_name) % 10000:04d}"
    
    # Add some basic validation and enhanced feedback
    priority_indicator = {"High": "[HIGH]", "Medium": "[MEDIUM]", "Low": "[LOW]"}.get(priority, "[UNKNOWN]")
    
    return (f"Task '{task_name}' successfully scheduled!\n"
            f"Deadline: {deadline}\n"
            f"Priority: {priority_indicator} {priority}\n"
            f"Task ID: {task_id}\n"
            f"Status: Ready for assignment")

def allocate_team_member(task_id: str, member_name: str, skills: str) -> str:
    """Allocate a team member to a specific task.
    
    Args:
        task_id: ID of the task to allocate
        member_name: Name of the team member
        skills: Relevant skills for the task
    
    Returns:
        Confirmation message with allocation details
    """
    # Enhanced with project management insights
    allocation_id = f"ALLOC-{hash(member_name) % 1000:03d}"
    skill_count = len([s.strip() for s in skills.split(',')])
    
    return (f"Team member '{member_name}' successfully allocated!\n"
            f"Assigned to: {task_id}\n"
            f"Relevant Skills: {skills}\n"
            f"Skill Match: {skill_count} skills identified\n"
            f"Allocation ID: {allocation_id}\n"
            f"Status: Assignment confirmed")

print("Enhanced Project Management tools defined successfully!")

# =============================================================================
# OPTION 1: LANGGRAPH IMPLEMENTATION - COMPLETE SOLUTION
# =============================================================================

def setup_langgraph_agent():
    """Complete LangGraph implementation"""
    
    # Install dependencies (uncomment if needed)
    # %pip install -U --quiet langgraph langchain pydantic langchain-openai
    
    from langgraph.prebuilt import create_react_agent
    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.memory import InMemorySaver
    
    print("LangGraph dependencies imported successfully!")
    
    # TODO COMPLETED: Set your OpenAI API Key
    OPENAI_API_KEY = "sk-proj-your-actual-openai-api-key-here"  # Replace with your real key
    
    # TODO COMPLETED: Modify temperature for optimal performance
    llm = init_chat_model(
        "openai:gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.1,  # Low temperature for consistent, professional responses
        max_tokens=1000
    )
    
    print("Language model initialized successfully!")
    
    # TODO COMPLETED: Complete the agent prompt
    agent_prompt = """
    You are a professional Project Management Assistant specializing in project planning and team coordination.

    CORE RESPONSIBILITIES:
    - Provide expert project management guidance and strategic advice
    - Schedule and organize project tasks with appropriate deadlines and priorities
    - Allocate team members to tasks based on skills, availability, and project requirements
    - Identify potential risks, bottlenecks, and optimization opportunities
    - Maintain professional, clear, and actionable communication

    TOOL USAGE GUIDELINES:
    - Use schedule_task() for any task scheduling requests - always include task name, deadline (YYYY-MM-DD format), and priority (High/Medium/Low)
    - Use allocate_team_member() for team assignments - always include task ID, member name, and relevant skills
    - When users request complex operations, break them down into appropriate tool calls
    - Always confirm actions taken and provide next steps or recommendations

    COMMUNICATION STYLE:
    - Be professional, concise, and action-oriented
    - Provide clear confirmations of completed actions
    - Offer strategic insights and project management best practices
    - Ask clarifying questions when information is missing or ambiguous
    - Structure responses logically with clear next steps

    RESPONSE FORMAT:
    1. Acknowledge the request
    2. Execute necessary tools
    3. Confirm actions taken
    4. Provide relevant insights or recommendations
    5. Suggest next steps when appropriate

    Remember: You are a strategic partner in project success, not just a task executor.
    """
    
    # TODO COMPLETED: Add tools to the agent
    project_agent = create_react_agent(
        model=llm,
        tools=[schedule_task, allocate_team_member],  # Both tools added
        prompt=agent_prompt,
        checkpointer=InMemorySaver()
    )
    
    print("Project Management Agent created successfully!")
    return project_agent

def test_langgraph_agent(project_agent):
    """Test the LangGraph agent with all scenarios"""
    config = {"configurable": {"thread_id": "project_session_001"}}
    
    test_scenarios = [
        {
            "name": "Task Scheduling",
            "query": "Schedule a task called 'Database Migration' for 2024-02-15 with High priority"
        },
        {
            "name": "Team Member Allocation", 
            "query": "Allocate team member 'Sarah Johnson' to task TSK-1234 with skills 'Python, SQL, Database Administration'"
        },
        {
            "name": "Complex Project Scenario",
            "query": "I need to schedule a 'Code Review' task for next Friday and allocate 'Mike Chen' who has 'Code Review, Python, Testing' skills"
        }
    ]
    
    print("\n" + "="*80)
    print("TESTING LANGGRAPH PROJECT MANAGEMENT AGENT")
    print("="*80)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\nTest {i}: {scenario['name']}")
        print("="*50)
        print(f"Query: {scenario['query']}")
        print("-"*50)
        
        try:
            response = project_agent.invoke(
                {"messages": [{"role": "user", "content": scenario['query']}]},
                config
            )
            print("Agent Response:")
            print(response["messages"][-1].content)
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "-"*60 + "\n")
    
    print("All LangGraph tests completed!")

# =============================================================================
# OPTION 2: GOOGLE ADK IMPLEMENTATION - COMPLETE SOLUTION  
# =============================================================================

def setup_google_adk_agent():
    """Complete Google ADK implementation"""
    
    # Install dependencies (uncomment if needed)
    # %pip install -U --quiet google-adk google-genai litellm
    
    from google.adk.tools import FunctionTool
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from google.adk.models.lite_llm import LiteLlm
    import os
    
    print("Google ADK dependencies imported successfully!")
    
    # TODO COMPLETED: Set your API Keys
    # Uncomment and set your actual API keys:
    # os.environ["GOOGLE_API_KEY"] = "your-actual-google-api-key"
    # os.environ['OPENAI_API_KEY'] = "sk-proj-your-actual-openai-api-key"
    
    # Configure the Language Model
    if os.environ.get('OPENAI_API_KEY') and os.environ.get('OPENAI_API_KEY') != "YOUR_OPENAI_API_KEY":
        MODEL_NAME = LiteLlm(model="openai/gpt-4o-mini")
    elif os.environ.get('GOOGLE_API_KEY') and os.environ.get('GOOGLE_API_KEY') != "YOUR_GOOGLE_API_KEY":
        MODEL_NAME = "gemini-2.0-flash"
    else:
        raise ValueError("No API key found. Please set either OPENAI_API_KEY or GOOGLE_API_KEY in the environment.")
    
    # Configuration constants
    APP_NAME = "project_management_app"
    USER_ID = "user_001"
    SESSION_ID = "session_001"
    
    print("Model configuration completed successfully!")
    print(f"Model: {MODEL_NAME}")
    
    # TODO COMPLETED: Create FunctionTool instances
    schedule_tool = FunctionTool(func=schedule_task)  # Schedule task tool
    allocate_tool = FunctionTool(func=allocate_team_member)  # Allocation tool
    
    print("Tools wrapped successfully!")
    
    # TODO COMPLETED: Complete the agent instruction
    agent_instruction = """
    You are an expert Project Management Assistant with deep expertise in project planning, team coordination, and strategic project execution.

    PRIMARY ROLE:
    As a professional project management consultant, you help users efficiently plan, organize, and execute projects through strategic task scheduling and optimal team resource allocation.

    CORE COMPETENCIES:
    - Strategic project planning and timeline optimization
    - Team resource allocation based on skills and availability  
    - Risk identification and mitigation planning
    - Process improvement and workflow optimization
    - Professional project communication and stakeholder management

    TOOL USAGE PROTOCOL:
    - Use schedule_task() for all task scheduling requests:
      * Always require: task_name, deadline (YYYY-MM-DD), priority (High/Medium/Low)
      * Provide strategic insights about timeline and dependencies
    - Use allocate_team_member() for team assignments:
      * Always require: task_id, member_name, skills
      * Consider skill-task alignment and resource optimization
    - Execute multiple tool calls for complex multi-step requests
    - Always validate inputs and provide clear confirmations

    COMMUNICATION EXCELLENCE:
    - Maintain professional, consultative tone throughout interactions
    - Provide structured, actionable responses with clear next steps
    - Offer strategic recommendations beyond basic tool execution
    - Ask clarifying questions to ensure optimal project outcomes
    - Anticipate potential issues and proactively suggest solutions

    RESPONSE STRUCTURE:
    1. Professional acknowledgment of the request
    2. Strategic analysis of requirements (if applicable)
    3. Execution of appropriate tools with clear reasoning
    4. Comprehensive confirmation of actions taken
    5. Strategic recommendations and optimization opportunities
    6. Clear next steps and follow-up actions

    QUALITY STANDARDS:
    - Ensure all responses demonstrate deep project management expertise
    - Provide value-added insights beyond simple tool execution
    - Maintain consistency in professional communication
    - Focus on actionable, results-oriented guidance
    - Consider both immediate needs and long-term project success

    Your goal is to be an indispensable strategic partner in achieving project excellence.
    """
    
    # TODO COMPLETED: Create the Project Management Agent
    project_agent = LlmAgent(
        model=MODEL_NAME,
        name="Strategic Project Management Assistant",  # Professional agent name
        description="An AI-powered project management consultant specializing in strategic planning, task scheduling, and team resource optimization for maximum project success",  # Comprehensive description
        instruction=agent_instruction,
        tools=[schedule_tool, allocate_tool]  # Both tools added
    )
    
    print("Project Management Agent created successfully!")
    print(f"Agent Name: {project_agent.name}")
    print(f"Agent Description: {project_agent.description}")
    print(f"Number of Tools: {len(project_agent.tools)}")
    
    return project_agent, APP_NAME, USER_ID, SESSION_ID

def setup_google_adk_runner(project_agent, APP_NAME, USER_ID, SESSION_ID):
    """Set up the Google ADK runner and session management"""
    import asyncio
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    
    session_service = InMemorySessionService()
    
    # TODO COMPLETED: Create runner with all required parameters
    runner = Runner(
        agent=project_agent,  # The agent we created
        app_name=APP_NAME,    # Application name
        session_service=session_service  # Session service
    )
    
    # Session setup with error handling
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, session_service.create_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=SESSION_ID
                ))
                session = future.result()
        else:
            session = asyncio.run(session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=SESSION_ID
            ))
        print("Session created successfully!")
    except Exception as e:
        print("Session will be created automatically when needed")
    
    print("Session management set up successfully!")
    print(f"Agent: {runner.agent.name}")
    print(f"App: {runner.app_name}")
    print(f"Session Service: {type(runner.session_service).__name__}")
    
    return runner, USER_ID, SESSION_ID

def create_google_adk_caller(runner, USER_ID, SESSION_ID):
    """Create the agent calling function for Google ADK"""
    from google.genai import types
    
    def call_agent(query):
        """Call the Google ADK agent with enhanced response handling"""
        content = types.Content(role='user', parts=[types.Part(text=query)])
        events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
        
        print(f"\nQuery: {query}")
        print("="*60)
        
        final_response = None
        for event in events:
            if event.is_final_response() and event.content:
                print("\nAGENT RESPONSE:")
                print("-" * 40)
                
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(f"Response: {part.text}")
                        final_response = part.text.strip()
                    elif hasattr(part, 'function_call'):
                        print(f"Function Call: {part.function_call}")
                    else:
                        print(f"Unknown Part Type: {type(part)}")
                
                break
        
        return final_response
    
    return call_agent

def test_google_adk_agent(call_agent):
    """Test the Google ADK agent with all scenarios"""
    test_scenarios = [
        "Schedule a task called 'Database Migration' for 2024-02-15 with High priority",
        "Allocate team member 'Sarah Johnson' to task TSK-1234 with skills 'Python, SQL, Database Administration'",
        "I need to schedule a 'Code Review' task for next Friday and allocate 'Mike Chen' who has 'Code Review, Python, Testing' skills"
    ]
    
    print("\n" + "="*80)
    print("TESTING GOOGLE ADK PROJECT MANAGEMENT AGENT")
    print("="*80)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\nTest {i}: {scenario}")
        print("-"*60)
        try:
            call_agent(scenario)
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "-"*60 + "\n")
    
    print("All Google ADK tests completed!")

# =============================================================================
# MAIN EXECUTION - CHOOSE YOUR FRAMEWORK
# =============================================================================

def main():
    """Main execution function - choose your framework"""
    print("WEEK 01 HOMEWORK: AI AGENT DEVELOPMENT")
    print("="*80)
    print("Choose your framework:")
    print("1. LangGraph Implementation")
    print("2. Google ADK Implementation")
    print("3. Run both for comparison (advanced)")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\nSetting up LangGraph Agent...")
        agent = setup_langgraph_agent()
        test_langgraph_agent(agent)
        
    elif choice == "2":
        print("\nSetting up Google ADK Agent...")
        agent, app_name, user_id, session_id = setup_google_adk_agent()
        runner, user_id, session_id = setup_google_adk_runner(agent, app_name, user_id, session_id)
        call_agent = create_google_adk_caller(runner, user_id, session_id)
        test_google_adk_agent(call_agent)
        
    elif choice == "3":
        print("\nRunning both frameworks for comparison...")
        print("\n" + "="*40 + " LANGGRAPH " + "="*40)
        agent1 = setup_langgraph_agent()
        test_langgraph_agent(agent1)
        
        print("\n" + "="*40 + " GOOGLE ADK " + "="*40)
        agent2, app_name, user_id, session_id = setup_google_adk_agent()
        runner, user_id, session_id = setup_google_adk_runner(agent2, app_name, user_id, session_id)
        call_agent = create_google_adk_caller(runner, user_id, session_id)
        test_google_adk_agent(call_agent)
        
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")
    
    print("\nHOMEWORK COMPLETION CHECKLIST:")
    print("- Project management tools defined")
    print("- Agent framework implemented") 
    print("- Professional prompts/instructions written")
    print("- Tools properly integrated")
    print("- Test scenarios executed")
    print("- Code documented and organized")
    print("\nGreat job! Your AI agent is ready for submission!")

if __name__ == "__main__":
    # Uncomment the line below to run interactively
    main()
    
    # For direct testing, you can also run individual components:
    print("Homework solution ready!")
    print("To test:")
    print("1. Set your API keys in the appropriate sections")
    print("2. Run main() to choose your framework")
    print("3. Or call setup functions directly for testing")
