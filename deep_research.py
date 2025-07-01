def create_deep_research_team(model_name, 
                             doc_knowledge_bases: List[Dict] = None) -> Any:
    llm = get_azure_llm()
    
    web_search = DuckDuckGoSearchRun()
    
    @tool
    def deep_web_search(query: str) -> str:
        """Perform deep web search using both backend API and DuckDuckGo for comprehensive results."""
        try:
            response = requests.post(
                f"{BACKEND_URL}/brave/search",
                json={"query": query},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                data = response.json()
                backend_result = data.get("result", "")
            else:
                backend_result = ""
            
            ddg_result = web_search.run(query)
            combined_result = f"Backend Search Results:\n{backend_result}\n\nAdditional Web Results:\n{ddg_result}"
            return combined_result
            
        except Exception as e:
            return f"Error in deep web search: {str(e)}"
    
    tools = [
        deep_web_search,
        search_knowledge_base
    ]
    
    tool_node = ToolNode(tools)
    
    # Research Planning Agent
    def research_planning_agent(state: ResearchState) -> ResearchState:
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Research Planning Specialist. Your job is to analyze user queries and create comprehensive research plans.
            
            Create a detailed research plan that includes:
            1. Key research areas to explore
            2. Specific search terms for each tool
            3. Priority order of research tasks
            4. Expected information sources
            
            Break down the query into 5-7 specific research tasks covering:
            - Current news and market information
            - Knowledge base search
            
            Format your response as a structured research plan with numbered tasks."""),
            ("human", "Create a research plan for: {query}")
        ])
        
        planning_chain = planning_prompt | llm | StrOutputParser()
        research_plan = planning_chain.invoke({"query": state["query"]})
        
        state["research_plan"] = research_plan
        state["messages"].append(f"Research Plan Created: {research_plan}")
        
        return state
    
    def research_execution_agent(state: ResearchState) -> ResearchState:
        execution_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Research Execution Specialist. Execute the research plan systematically using available tools.
            
            Execute each research task in the plan. For each tool call:
            - Extract key search terms (no complete sentences)
            - Use appropriate parameters for each tool
            - Gather comprehensive information
            
            Always include the source links for all information used.
             
            TOOL PARAMETER GUIDELINES:
            - Remove articles, prepositions, and question words
            
            Call ALL relevant tools to gather comprehensive information."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        execution_agent = create_openai_functions_agent(llm, tools, execution_prompt)
        execution_executor = AgentExecutor(
            agent=execution_agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
        
        research_input = {
            "input": f"Execute this research plan step by step:\n\n{state['research_plan']}\n\nOriginal Query: {state['query']}\n\nPlease use all relevant tools to gather comprehensive information for each task in the plan."
        }
        
        research_result = execution_executor.invoke(research_input)
        
        state["gathered_info"]["research_execution"] = research_result.get("output", "")
        state["messages"].append(f"Research Execution Completed")
        
        return state
    
    def research_analysis_agent(state: ResearchState) -> ResearchState:
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Research Analysis Specialist. Synthesize all gathered research information into a comprehensive analysis.
            
            Original Query: {query}
            Research Plan: {research_plan}
            Gathered Information: {gathered_info}
            
            Create a comprehensive analysis that includes:
            1. Executive Summary (200-300 words)
            2. Detailed Findings by Category:
            3. Key Insights and Implications
            4. Limitations and Knowledge Gaps
            5. Recommendations for Further Research
            6. Source Citations
            
            Always include the source links for all information used in the analysis.
             
            The final analysis should be at least 1000 words with proper formatting and evidence-based conclusions.
            """),
            ("human", "Analyze all the research findings and provide comprehensive analysis.")
        ])
        
        analysis_chain = analysis_prompt | llm | StrOutputParser()
        
        analysis_input = {
            "query": state["query"],
            "research_plan": state["research_plan"],
            "gathered_info": str(state["gathered_info"])
        }
        
        final_analysis = analysis_chain.invoke(analysis_input)
        
        state["final_analysis"] = final_analysis
        state["messages"].append("Comprehensive Analysis Completed")
        
        return state
    
    # Build the graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("research_planning", research_planning_agent)
    workflow.add_node("research_execution", research_execution_agent)
    workflow.add_node("research_analysis", research_analysis_agent)
    
    # Add edges
    workflow.set_entry_point("research_planning")
    workflow.add_edge("research_planning", "research_execution")
    workflow.add_edge("research_execution", "research_analysis")
    workflow.add_edge("research_analysis", END)
    
    app = workflow.compile()

    class DeepResearchWorkflow:
        def __init__(self, compiled_graph):
            self.app = compiled_graph
            
        def invoke(self, input_data):
            initial_state = ResearchState(
                query=input_data.get("input", ""),
                research_plan="",
                gathered_info={},
                final_analysis="",
                messages=[]
            )
            
            result = self.app.invoke(initial_state)
            return {
                "output": result["final_analysis"]
            }
            
        def stream(self, input_data):
            initial_state = ResearchState(
                query=input_data.get("input", ""),
                research_plan="",
                gathered_info={},
                final_analysis="",
                messages=[]
            )
            
            for chunk in self.app.stream(initial_state):
                yield chunk

        async def astream_events(self, input_data, version="v2"):
            initial_state = ResearchState(
                query=input_data.get("input", ""),
                research_plan="",
                gathered_info={},
                final_analysis="",
                messages=[]
            )
            
            async for event in self.app.astream_events(initial_state, version=version):
                yield event
    
    return DeepResearchWorkflow(app)

def create_rag_agent(url: str, model_name) -> Optional[AgentExecutor]:
    """Create RAG agent using backend for URL processing"""
    try:
        # Upload URL to backend
        doc_name, message = upload_url_to_backend(url)
        
        if not doc_name:
            print(f"Failed to upload URL to backend: {message}")
            return None
        
        llm = get_azure_llm()
        
        @tool
        def search_url_content(query: str) -> str:
            """Search the loaded URL content for relevant information."""
            try:
                result = query_document_via_backend(doc_name, query)
                return f"Relevant content from {url}:\n\n{result}"
            except Exception as e:
                return f"Error searching URL content: {str(e)}"
        
        tools = [search_url_content]
        
        system_prompt = f"""You are a specialized RAG assistant for analyzing content from: {url}

        Search through the loaded web content to answer questions. Always use the search_url_content tool to find relevant information before responding.

        When answering:
        1. Search for relevant content using the tool
        2. Provide accurate information based on the retrieved content
        3. Cite that the information comes from the analyzed URL
        4. If information is not found in the content, clearly state this

        Be helpful, accurate, and always reference the source URL in your responses.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
        setattr(agent_executor, '_doc_name', doc_name)
        
        return agent_executor
        
    except Exception as e:
        print(f"Error creating RAG agent: {str(e)}")
        return None

async def stream_agent_response_async(agent, chat_history, prompt, response_placeholder):
    full_response = ""
    
    try:
        async for event in agent.astream_events(
            {"chat_history": chat_history, "input": prompt},
            version="v2",
        ):
            kind = event["event"]
            
            if kind == "on_chain_start":
                if event["name"] == "agent":
                    tool_start_text = f"\n🤖 **Starting agent:** {event['name']}\n\n"
                    full_response += tool_start_text
                    response_placeholder.markdown(full_response + "▌")
            elif kind == "on_chain_end":
                if event["name"] == "agent":
                    agent_end_text = f"\n✅ **Agent completed**\n\n"
                    full_response += agent_end_text
                    response_placeholder.markdown(full_response + "▌")

            elif kind == "on_chat_model_start":
                if "langgraph_node" in event.get("metadata", {}):
                    node_name = event["metadata"]["langgraph_node"]
                    
                    messages = event.get("data", {}).get("input", {}).get("messages", [])
                    if messages and len(messages) > 0:
                        system_message = str(messages[0])
                        
                        if "Research Planning Specialist" in system_message:
                            agent_type = "Research Planning Specialist"
                        elif "Research Execution Specialist" in system_message:
                            agent_type = "Research Execution Specialist"
                        elif "Research Analysis Specialist" in system_message:
                            agent_type = "Research Analysis Specialist"
                        else:
                            agent_type = node_name
                        
                        model_start_text = f"\n🤖 **{agent_type} is thinking...**\n\n"
                        full_response += model_start_text
                        response_placeholder.markdown(full_response + "▌")

            elif kind == "on_chat_model_stream":
                # Only stream final content from research_analysis node
                if "langgraph_node" in event.get("metadata", {}):
                    node_name = event["metadata"]["langgraph_node"]
                    
                    # Only show streaming content for the final analysis
                    if node_name == "research_analysis":
                        content = event["data"]["chunk"].content
                        if content:
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                else:
                    # For regular agents (non-deep research)
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")

            
            elif kind == "on_tool_start":
                tool_name = event["name"]
                tool_input = event["data"].get("input", {})
                tool_start_text = f"\n🔧 **Starting tool:** {tool_name}\n **Input:** {tool_input}\n\n"
                full_response += tool_start_text
                response_placeholder.markdown(full_response + "▌")
            
            elif kind == "on_tool_end":
                tool_name = event["name"]
                tool_output = event["data"].get("output", "")
                # Truncate long outputs for display
                display_output = str(tool_output)[:200] + "..." if len(str(tool_output)) > 100 else str(tool_output)
                tool_end_text = f"\n✅ **Tool completed:** {tool_name}\n **Output:** {display_output}\n\n"
                full_response += tool_end_text
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        return full_response
    
    except Exception as e:
        st.warning("Streaming failed, falling back to regular mode...")
        response = agent.invoke({
            "chat_history": chat_history,
            "input": prompt
        })
        if isinstance(response, dict) and "output" in response:
            response = response["output"]
        response_placeholder.markdown(response)
        return response

def stream_agent_response(agent, chat_history, prompt, response_placeholder):
    return asyncio.run(stream_agent_response_async(agent, chat_history, prompt, response_placeholder))
